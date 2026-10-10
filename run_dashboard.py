"""CLI entry point to launch the ThermalEye Interactive Dashboard."""
import os
import sys
from pathlib import Path

def main():
    print("Launching ThermalEye Phase 7 Dashboard...")
    dashboard_path = Path("src/thermaleye/dashboard/app.py")
    
    if not dashboard_path.exists():
        print(f"Error: Dashboard application not found at {dashboard_path}")
        sys.exit(1)
        
    os.system(f"streamlit run {dashboard_path}")

if __name__ == "__main__":
    main()
