import argparse
import sys
import numpy as np
import torch
from deepvol.calibration.interface import calibrate

def main():
    parser = argparse.ArgumentParser(description="DeepVol Volatility Model Calibration CLI")
    parser.add_argument("--surface", type=str, required=False, help="Path to CSV or NPZ containing implied volatility surface")
    parser.add_argument("--model", type=str, default="heston", choices=["heston", "sabr", "ssvi", "rbergomi", "fno", "rough_heston"], help="Model to calibrate")
    parser.add_argument("--method", type=str, default="newton", choices=["newton", "l-bfgs"], help="Calibration method")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use")
    
    args = parser.parse_args()
    
    T_grid = None
    K_grid = None
    
    # If no surface is provided, run a quick self-test or generate a dummy surface
    if args.surface is None:
        print("No input surface provided. Generating a dummy target surface for self-test...")
        # Grid of shape (8, 11)
        T_grid = np.array([0.08, 0.16, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], dtype=np.float32)
        K_grid = np.linspace(-0.5, 0.5, 11, dtype=np.float32)
        # Create a simple smile
        iv_surface = np.zeros((len(T_grid), len(K_grid)), dtype=np.float32)
        for i, t in enumerate(T_grid):
            for j, k in enumerate(K_grid):
                iv_surface[i, j] = 0.3 + 0.1 * k**2 - 0.05 * t
        iv_surface = np.clip(iv_surface, 0.05, 1.0)
    else:
        try:
            if args.surface.endswith(".csv"):
                iv_surface = np.loadtxt(args.surface, delimiter=",", dtype=np.float32)
            elif args.surface.endswith(".npz"):
                data = np.load(args.surface)
                # Look for 'iv_surface' or take the first array
                key = 'iv_surface' if 'iv_surface' in data else data.files[0]
                iv_surface = data[key].astype(np.float32)
                if 'T_grid' in data:
                    T_grid = data['T_grid'].astype(np.float32)
                if 'K_grid' in data:
                    K_grid = data['K_grid'].astype(np.float32)
            else:
                iv_surface = np.load(args.surface).astype(np.float32)
        except Exception as e:
            print(f"Error loading surface file {args.surface}: {e}", file=sys.stderr)
            sys.exit(1)
            
    print(f"Calibrating model '{args.model}' using '{args.method}' method on '{args.device}'...")
    try:
        kwargs = {}
        if T_grid is not None:
            kwargs["T_grid"] = T_grid
        if K_grid is not None:
            kwargs["K_grid"] = K_grid
            
        res = calibrate(
            market_iv_surface=iv_surface,
            model_name=args.model,
            method=args.method,
            device=args.device,
            **kwargs
        )
        print("\nCalibration succeeded!")
        print(f"Elapsed Time: {res.elapsed_time:.4f} seconds")
        print(f"Final RMSE: {res.rmse:.6f}")
        print("Calibrated Parameters:")
        print(res.parameters)
        print("Metadata:")
        for k, v in res.info.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"Calibration failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
