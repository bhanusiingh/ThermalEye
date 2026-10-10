"""ThermalEye: Satellite thermal anomaly analysis for India."""

import warnings

# Suppress harmless external warning between urllib3 and requests
warnings.filterwarnings("ignore", message=".*urllib3.*")

__version__ = "0.1.0"
