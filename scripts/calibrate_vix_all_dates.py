import os
import sys
import json
from datetime import date
from pathlib import Path

# Add src path
project_root = Path(__file__).parents[1]
sys.path.insert(0, str(project_root / "src"))

from market.spx_data import download_spx_chain, clean_chain, to_iv_surface
from market.vix_futures import fetch_vix_futures
from calibration.joint_calibration import calibrate_joint_multitenor

DATES = ["2020-03-16", "2022-01-24", "2024-01-02", "2024-08-05"]
S0_MAP = {
    "2020-03-16": 2400.0,
    "2022-01-24": 4400.0,
    "2024-01-02": 4700.0,
    "2024-08-05": 5200.0,
}

def main():
    results_dir = project_root / "results" / "vix_term_structure"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for date_str in DATES:
        print(f"Calibrating date: {date_str}")
        dt = date.fromisoformat(date_str)
        
        # SPX surface
        df_spx = download_spx_chain(dt, cache=True)
        df_clean = clean_chain(df_spx)
        S0 = S0_MAP[date_str]
        spx_surface = to_iv_surface(df_clean, S0, 0.05, 0.015)
        
        # VIX term structure
        vix_df = fetch_vix_futures(date_str)
        
        # Calibrate
        res = calibrate_joint_multitenor(spx_surface, vix_df, weights=(1.0, 1.0), n_restarts=3, seed=42)
        
        # Save results
        out_file = results_dir / f"{date_str}.json"
        with open(out_file, "w") as f:
            json.dump(res, f, indent=4)
        print(f"Saved results for {date_str} to {out_file}")

if __name__ == "__main__":
    main()
