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

from thermaleye.ingestion.firms import main

if __name__ == "__main__":
    main()
