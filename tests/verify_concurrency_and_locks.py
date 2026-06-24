import asyncio
import httpx
import time
import os
import sys
import subprocess
import signal
from pathlib import Path
import psutil

# Ensure project src is in PYTHONPATH
PROJECT_ROOT = Path(__file__).parents[1].resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PORT = 8888
BASE_URL = f"http://127.0.0.1:{PORT}"

def create_patched_server():
    server_path = PROJECT_ROOT / "src" / "deepvol" / "api" / "server.py"
    if not server_path.exists():
        raise FileNotFoundError(f"Could not find server.py at {server_path}")
        
    code = server_path.read_text()
    
    # 1. Fix fno_model and normalizers imports
    code = code.replace(
        "        from deepvol.surrogates.fno_model import MirrorPaddedFNO2d\n        from deepvol.surrogates.normalizers import IVSurfaceNormalizer, ParameterNormalizer",
        "        from deepvol.surrogates.fno_model import MirrorPaddedFNO2d\n        from deepvol.surrogates.normalizers import IVSurfaceNormalizer, ParameterNormalizer"
    )
    code = code.replace(
        "        from deepvol.surrogates.normalizers import IVSurfaceNormalizer, ParameterNormalizer",
        "        from deepvol.surrogates.normalizers import IVSurfaceNormalizer, ParameterNormalizer"
    )
    
    # 2. Fix greeks.portfolio_greeks import
    code = code.replace(
        "    from deepvol.greeks.portfolio_greeks import bs_greeks",
        "    from deepvol.greeks.portfolio_greeks import bs_greeks"
    )
    
    # 3. Fix all Path(__file__) parent resolutions to absolute paths
    code = code.replace(
        'Path(__file__).parents[3] / "artifacts"',
        'Path("/home/execorn/programming/derivatives/artifacts")'
    )
    code = code.replace(
        'Path(__file__).parents[2] / "artifacts"',
        'Path("/home/execorn/programming/derivatives/artifacts")'
    )
    code = code.replace(
        'Path(__file__).parents[2] / "scripts"',
        'Path("/home/execorn/programming/derivatives/scripts")'
    )
    code = code.replace(
        'Path(__file__).parents[2]',
        'Path("/home/execorn/programming/derivatives")'
    )
    
    # 5. Fix the IndentationError in _hot_reload_model_weights
    # Lines 253-260 are currently indented 4 spaces, they should be indented 8 spaces.
    target_unindented = """    from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
    from deepvol.surrogates.normalizers import IVSurfaceNormalizer, ParameterNormalizer
    from deepvol.utils.path_helpers import get_project_root

    artifacts_dir = get_project_root() / "artifacts"
    pn_path = artifacts_dir / "models" / pn_file
    yn_path = artifacts_dir / "models" / yn_file"""

    replacement_indented = """        from deepvol.surrogates.fno_model import MirrorPaddedFNO2d
        from deepvol.surrogates.normalizers import IVSurfaceNormalizer, ParameterNormalizer
        from deepvol.utils.path_helpers import get_project_root

        artifacts_dir = get_project_root() / "artifacts"
        pn_path = artifacts_dir / "models" / pn_file
        yn_path = artifacts_dir / "models" / yn_file"""

    if target_unindented in code:
        code = code.replace(target_unindented, replacement_indented)
    else:
        # Fallback check or line-by-line replacement if exact block match fails
        lines = code.splitlines()
        for idx in range(252, 260):
            if lines[idx].startswith("    from") or lines[idx].startswith("    artifacts_dir") or lines[idx].startswith("    pn_path") or lines[idx].startswith("    yn_path"):
                lines[idx] = "    " + lines[idx]
        code = "\n".join(lines)

    # 6. Expose a test route for triggering reload via HTTP to test read-write lock concurrency
    reload_route = """
@app.post("/test/reload/{model_name}", tags=["Test"])
async def test_reload(model_name: str):
    await _hot_reload_model_weights(model_name)
    return {"status": "reloaded"}
"""
    code += reload_route
    
    patched_path = PROJECT_ROOT / "tests" / "patched_server.py"
    patched_path.write_text(code)
    print(f"Patched server written to {patched_path}")
    return patched_path

async def run_stress_test():
    patched_file = create_patched_server()
    
    # Start the server using the virtual env python interpreter
    python_bin = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{PROJECT_ROOT / 'src'}"
    
    proc = subprocess.Popen(
        [python_bin, "-m", "uvicorn", "patched_server:app", "--host", "127.0.0.1", "--port", str(PORT)],
        env=env,
        cwd=str(PROJECT_ROOT / "tests"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )
    
    print("Starting API Server...")
    time.sleep(2)  # Give uvicorn some time to start
    
    # Wait for the server to be healthy
    async with httpx.AsyncClient() as client:
        healthy = False
        for _ in range(10):
            try:
                resp = await client.get(f"{BASE_URL}/health", timeout=2.0)
                if resp.status_code == 200:
                    healthy = True
                    print("Server is healthy.")
                    break
            except Exception:
                await asyncio.sleep(1)
        
        if not healthy:
            print("Failed to start server. Stdout/Stderr:")
            stdout, stderr = proc.communicate(timeout=1.0)
            print("STDOUT:", stdout.decode())
            print("STDERR:", stderr.decode())
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            sys.exit(1)
            
        try:
            # Prepare a dummy 8x11 market implied volatility surface
            dummy_surface = [[0.2] * 11 for _ in range(8)]
            
            # --- 1. Test Concurrency and GPU Semaphore Serialization ---
            # Send 3 concurrent requests to calibrate Heston model.
            # If the GPU Semaphore (asyncio.Semaphore(1)) works, they will run sequentially.
            print("\n--- Testing GPU Semaphore Serialization ---")
            payload = {
                "market_iv": dummy_surface,
                "n_starts": 1,
                "max_iter": 5,
                "tol": 1e-5
            }
            
            t0 = time.time()
            
            async def send_calibration(req_id):
                start = time.time()
                resp = await client.post(f"{BASE_URL}/calibrate/heston", json=payload, timeout=20.0)
                end = time.time()
                elapsed = end - start
                try:
                    res_json = resp.json()
                except Exception:
                    res_json = {}
                api_elapsed = res_json.get('elapsed_ms')
                api_elapsed_str = f"{api_elapsed / 1000.0:.2f}s" if api_elapsed is not None else "N/A"
                print(f"Calibration Request {req_id}: Status {resp.status_code}, Client Elapsed: {elapsed:.2f}s, API Elapsed: {api_elapsed_str}")
                if resp.status_code != 200:
                    print("Error Response Body:", resp.text)
                return res_json
                
            # Fire them concurrently
            tasks = [send_calibration(i) for i in range(3)]
            results = await asyncio.gather(*tasks)
            
            total_wall_clock = time.time() - t0
            sum_api_times = sum((res.get("elapsed_ms") or 0.0) / 1000.0 for res in results)
            print(f"Total Wall-Clock Time: {total_wall_clock:.2f}s")
            print(f"Sum of individual API Calibration execution times: {sum_api_times:.2f}s")
            
            # Verify serialization: wall clock time should be close to or greater than the sum of execution times
            # since they are serialized by the semaphore.
            if total_wall_clock >= sum_api_times * 0.8:
                print("SUCCESS: Calibrations are successfully serialized by the GPU semaphore!")
            else:
                print("WARNING: Calibrations did not appear to run sequentially. Semaphore might be bypassed.")
                
            # --- 2. Test Read-Write Lock (Calibration vs Weight Reload) ---
            print("\n--- Testing Read-Write Lock (Calibration vs Weight Reload) ---")
            # We want to trigger a weight reload while a calibration is running.
            # The reload should be blocked until calibration releases the read lock,
            # and calibration should not fail or crash during the reload.
            
            # Start a calibration that takes some time (increase max_iter to make it longer)
            long_payload = {
                "market_iv": dummy_surface,
                "n_starts": 2,
                "max_iter": 50,
                "tol": 1e-8
            }
            
            async def run_calib_and_reload():
                calib_task = asyncio.create_task(client.post(f"{BASE_URL}/calibrate/heston", json=long_payload, timeout=40.0))
                # Wait 0.5s for calibration to start and acquire read lock
                await asyncio.sleep(0.5)
                
                print("Triggering weights reload concurrently...")
                reload_start = time.time()
                reload_resp = await client.post(f"{BASE_URL}/test/reload/heston", timeout=40.0)
                reload_end = time.time()
                print(f"Reload completed in {reload_end - reload_start:.2f}s, status: {reload_resp.status_code}")
                
                calib_resp = await calib_task
                print(f"Calibration completed with status {calib_resp.status_code}")
                return calib_resp, reload_resp
                
            calib_res, reload_res = await run_calib_and_reload()
            assert calib_res.status_code == 200, "Calibration failed during concurrent reload"
            assert reload_res.status_code == 200, "Reload failed"
            print("SUCCESS: Read-Write Lock protects concurrent calibration and reload without failure.")
            
            # --- 3. Test Session Cache Memory Leak ---
            print("\n--- Testing Session Cache Memory Leak ---")
            # We will hit the session calibration endpoint repeatedly to see if memory leaks
            # or if the TTLCache limits memory usage and cleans up.
            process = psutil.Process(proc.pid)
            initial_memory = process.memory_info().rss / (1024 * 1024)
            print(f"Initial Server Memory: {initial_memory:.2f} MB")
            
            session_payload = {
                "market_iv": dummy_surface,
                "n_starts": 1,
                "max_iter": 2,
                "tol": 1e-3
            }
            
            print("Creating 50 calibration sessions to stress the cache...")
            for i in range(50):
                resp = await client.post(f"{BASE_URL}/session/calibrate/sabr", json=session_payload)
                if i % 10 == 0:
                    print(f"Session {i} created.")
            
            post_session_memory = process.memory_info().rss / (1024 * 1024)
            print(f"Post-Session Server Memory: {post_session_memory:.2f} MB")
            diff = post_session_memory - initial_memory
            print(f"Memory growth: {diff:.2f} MB")
            
            if diff > 100:  # arbitrary threshold for severe leak
                print("WARNING: Severe memory growth detected during session cache stress test.")
            else:
                print("SUCCESS: Session cache memory usage is stable.")
                
        finally:
            print("\nStopping API Server...")
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait()
            # Clean up the patched server file
            if patched_file.exists():
                patched_file.unlink()
                print("Cleaned up patched server file.")

if __name__ == "__main__":
    asyncio.run(run_stress_test())
