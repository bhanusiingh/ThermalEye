import sys
import warnings
from pathlib import Path

# Suppress known harmless third-party version mismatch warning
warnings.filterwarnings("ignore", message=".*urllib3.*")

# Ensure src/ is on sys.path when executed directly
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import argparse

from thermaleye.ingestion.firms import run_ingestion, FirmsConfigError, FirmsAPIError
from thermaleye.config import BoundingBox, DEFAULT_INDIA_BBOX, DEFAULT_VIIRS_PRODUCTS, DEFAULT_DAY_RANGE, DEFAULT_OUTPUT_PATH

def main():
    parser = argparse.ArgumentParser(description="ThermalEye NASA FIRMS VIIRS Ingestion")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to save the output CSV")
    parser.add_argument("--date", type=str, default=None, help="Specific date to query (YYYY-MM-DD)")
    parser.add_argument("--product", type=str, action='append', help="FIRMS product name(s) (can be specified multiple times)")
    parser.add_argument("--bbox", type=str, default=None, help="Bounding box as min_lon,min_lat,max_lon,max_lat")
    parser.add_argument("--day-range", type=int, default=DEFAULT_DAY_RANGE, help="Number of days to query (1-10)")

    args = parser.parse_args()

    import re
    if args.date:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", args.date):
            print("Error: --date must be in YYYY-MM-DD format.")
            sys.exit(1)

    if args.day_range < 1 or args.day_range > 10:
        print("Error: --day-range must be between 1 and 10.")
        sys.exit(1)

    bbox = DEFAULT_INDIA_BBOX
    if args.bbox:
        try:
            coords = [float(x.strip()) for x in args.bbox.split(",")]
            bbox = BoundingBox(*coords)
        except Exception as e:
            print(f"Error: Invalid bbox format. {e}")
            sys.exit(1)

    products = args.product if args.product else DEFAULT_VIIRS_PRODUCTS
    allowed_products = {"VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "VIIRS_SNPP_NRT",
                        "VIIRS_NOAA20_SP", "VIIRS_NOAA21_SP", "VIIRS_SNPP_SP"}
    for p in products:
        if p not in allowed_products:
            print(f"Error: Unsupported product '{p}'. Supported products are: {allowed_products}")
            sys.exit(1)

    print("Initiating ThermalEye NASA FIRMS VIIRS Ingestion...")
    try:
        summary = run_ingestion(
            output_path=args.output,
            date_str=args.date,
            products=products,
            bbox=bbox,
            day_range=args.day_range
        )
        print(summary.format_report())
        if summary.failed_requests:
            print("\nWarning: Some products encountered errors (see summary above).")
    except FirmsConfigError as e:
        print(f"\nConfiguration Error: {e}")
        print("Please check your .env file or reference .env.example.")
        sys.exit(1)
    except FirmsAPIError as e:
        print(f"\nAPI Ingestion Error: {e}")
        sys.exit(2)
    except Exception as e:
        print(f"\nUnexpected Error: {e}")
        sys.exit(3)

if __name__ == "__main__":
    main()
