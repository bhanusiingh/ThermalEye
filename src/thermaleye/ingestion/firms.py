"""NASA FIRMS Near-Real-Time VIIRS data ingestion module."""

from dataclasses import dataclass, field
from datetime import date
import io
from pathlib import Path
import re
from typing import List, Optional, Sequence, Tuple
import warnings

# Suppress known harmless third-party version mismatch warning from requests
warnings.filterwarnings("ignore", category=UserWarning, module="requests")

import requests
import pandas as pd

from thermaleye.config import (
    BoundingBox,
    DEFAULT_INDIA_BBOX,
    DEFAULT_VIIRS_PRODUCTS,
    DEFAULT_DAY_RANGE,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_OUTPUT_PATH,
    FIRMS_API_BASE_URL,
    get_firms_map_key,
)


class FirmsIngestionError(Exception):
    """Base exception for FIRMS ingestion errors."""


class FirmsConfigError(FirmsIngestionError):
    """Raised when configuration or credentials are missing or invalid."""


class FirmsAPIError(FirmsIngestionError):
    """Raised when FIRMS API returns an error or invalid payload."""


@dataclass
class ProductFetchResult:
    """Result of fetching a single satellite product."""

    product: str
    success: bool
    row_count: int = 0
    error_message: Optional[str] = None
    data: Optional[pd.DataFrame] = None


@dataclass
class IngestionSummary:
    """Concise summary of an ingestion run."""

    products_attempted: List[str]
    successful_requests: List[str]
    failed_requests: List[Tuple[str, str]]
    total_raw_rows: int
    saved_rows: int
    columns: List[str]
    date_range: Optional[Tuple[str, str]]
    output_path: Optional[Path]

    def format_report(self) -> str:
        """Format concise human-readable summary report."""
        lines = [
            "================ FIRMS INGESTION SUMMARY ================",
            f"Products Attempted   : {', '.join(self.products_attempted)}",
            f"Successful Requests  : {', '.join(self.successful_requests) if self.successful_requests else 'None'}",
        ]
        if self.failed_requests:
            lines.append("Failed Requests      :")
            for prod, err in self.failed_requests:
                lines.append(f"  - {prod}: {err}")
        else:
            lines.append("Failed Requests      : None")

        lines.extend([
            f"Total Raw Rows       : {self.total_raw_rows}",
            f"Unique Saved Rows    : {self.saved_rows}",
            f"Available Columns    : {', '.join(self.columns) if self.columns else 'None'}",
            f"Date Range           : {f'{self.date_range[0]} to {self.date_range[1]}' if self.date_range else 'N/A'}",
            f"Output Destination   : {self.output_path if self.output_path else 'None'}",
            "=========================================================",
        ])
        return "\n".join(lines)


class FirmsClient:
    """Client for NASA FIRMS Area API."""

    # Expected column names in FIRMS VIIRS CSV responses
    EXPECTED_COLUMNS = [
        "latitude",
        "longitude",
        "bright_ti4",
        "scan",
        "track",
        "acq_date",
        "acq_time",
        "satellite",
        "instrument",
        "confidence",
        "version",
        "bright_ti5",
        "frp",
        "daynight",
    ]

    KNOWN_ERROR_PATTERNS = [
        re.compile(r"invalid\s+map_key", re.IGNORECASE),
        re.compile(r"bad\s+request", re.IGNORECASE),
        re.compile(r"transaction\s+limit\s+exceeded", re.IGNORECASE),
        re.compile(r"no\s+such\s+product", re.IGNORECASE),
        re.compile(r"access\s+denied", re.IGNORECASE),
    ]

    def __init__(
        self,
        map_key: Optional[str] = None,
        base_url: str = FIRMS_API_BASE_URL,
        timeout: int = DEFAULT_REQUEST_TIMEOUT,
    ) -> None:
        key = map_key if map_key is not None else get_firms_map_key()
        if not key:
            raise FirmsConfigError(
                "FIRMS_MAP_KEY is missing or empty. Please set it in .env or the environment."
            )
        self._map_key = key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _sanitize(self, text: str) -> str:
        """Ensure secret map key never appears in log messages or URLs."""
        if not self._map_key:
            return text
        return text.replace(self._map_key, "***REDACTED***")

    def build_request_url(
        self,
        product: str,
        bbox: BoundingBox,
        day_range: int = DEFAULT_DAY_RANGE,
        date_str: Optional[str] = None,
    ) -> str:
        """Construct the Area CSV URL for FIRMS API.

        Format: https://firms.modaps.eosdis.nasa.gov/api/area/csv/[MAP_KEY]/[SOURCE]/[AREA_COORDINATES]/[DAY_RANGE]/[DATE]
        """
        if day_range < 1 or day_range > 10:
            raise ValueError(f"day_range must be between 1 and 10, got {day_range}")

        url = f"{self.base_url}/{self._map_key}/{product}/{bbox.to_firms_str()}/{day_range}"
        if date_str:
            url = f"{url}/{date_str}"
        return url

    def parse_csv_response(self, text: str, product: str) -> pd.DataFrame:
        """Parse and validate FIRMS CSV response text into a DataFrame."""
        stripped = text.strip()

        # Check for HTML responses (typical of 404, 502, proxy errors or gateway blocks)
        if stripped.lower().startswith("<!doctype") or "<html" in stripped.lower():
            raise FirmsAPIError("Received HTML response from FIRMS API instead of expected CSV.")

        # Check for known FIRMS plain-text error messages
        for pattern in self.KNOWN_ERROR_PATTERNS:
            if pattern.search(stripped):
                cleaned_msg = self._sanitize(stripped.splitlines()[0] if stripped else "API Error")
                raise FirmsAPIError(f"FIRMS API reported error: {cleaned_msg}")

        # Empty body
        if not stripped:
            df = pd.DataFrame(columns=self.EXPECTED_COLUMNS)
            df["source_product"] = pd.Series(dtype="object")
            return df

        try:
            df = pd.read_csv(io.StringIO(stripped))
        except Exception as e:
            raise FirmsAPIError(f"Failed to parse CSV response: {e}") from e

        # Validate that returned headers look like FIRMS data
        # Check at least latitude and longitude are present
        if "latitude" not in df.columns or "longitude" not in df.columns:
            # If the CSV has 1 column and 0-1 rows, it might be an uncaught plain text error message
            if len(df.columns) <= 2 and len(df) == 0:
                first_line = self._sanitize(stripped.splitlines()[0])
                raise FirmsAPIError(f"Unrecognized response header from FIRMS API: '{first_line}'")
            raise FirmsAPIError(f"Expected latitude and longitude columns; got: {list(df.columns)}")

        # Add source product column to preserve origin
        df["source_product"] = product
        return df

    def fetch_product(
        self,
        product: str,
        bbox: BoundingBox,
        day_range: int = DEFAULT_DAY_RANGE,
        date_str: Optional[str] = None,
    ) -> ProductFetchResult:
        """Fetch thermal detection data for a single product."""
        raw_url = self.build_request_url(product, bbox, day_range, date_str)
        sanitized_url = self._sanitize(raw_url)

        try:
            response = requests.get(raw_url, timeout=self.timeout)
            if response.status_code != 200:
                sanitized_body = self._sanitize(response.text[:200])
                return ProductFetchResult(
                    product=product,
                    success=False,
                    error_message=f"HTTP {response.status_code}: {sanitized_body}",
                )

            df = self.parse_csv_response(response.text, product)
            return ProductFetchResult(
                product=product,
                success=True,
                row_count=len(df),
                data=df,
            )
        except requests.RequestException as e:
            safe_err = self._sanitize(str(e))
            return ProductFetchResult(
                product=product,
                success=False,
                error_message=f"Request network failure: {safe_err}",
            )
        except FirmsAPIError as e:
            return ProductFetchResult(
                product=product,
                success=False,
                error_message=str(e),
            )


def deduplicate_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate records without discarding legitimate distinct observations.

    Deduplication key: latitude, longitude, acq_date, acq_time, satellite.
    Different satellites (NOAA-20 vs NOAA-21) or different acquisition times/positions
    are preserved as distinct observations.
    """
    if df.empty:
        return df

    dedup_cols = ["latitude", "longitude", "acq_date", "acq_time", "satellite"]
    available_cols = [c for c in dedup_cols if c in df.columns]

    if available_cols:
        return df.drop_duplicates(subset=available_cols).reset_index(drop=True)
    return df.drop_duplicates().reset_index(drop=True)


def run_ingestion(
    map_key: Optional[str] = None,
    bbox: BoundingBox = DEFAULT_INDIA_BBOX,
    products: Sequence[str] = DEFAULT_VIIRS_PRODUCTS,
    day_range: int = DEFAULT_DAY_RANGE,
    date_str: Optional[str] = None,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    timeout: int = DEFAULT_REQUEST_TIMEOUT,
) -> IngestionSummary:
    """Execute Phase 1 FIRMS ingestion across configured products and save to CSV."""
    client = FirmsClient(map_key=map_key, timeout=timeout)

    successful_products: List[str] = []
    failed_products: List[Tuple[str, str]] = []
    dataframes: List[pd.DataFrame] = []
    total_raw_rows = 0

    for product in products:
        result = client.fetch_product(
            product=product,
            bbox=bbox,
            day_range=day_range,
            date_str=date_str,
        )
        if result.success and result.data is not None:
            successful_products.append(product)
            total_raw_rows += result.row_count
            dataframes.append(result.data)
        else:
            failed_products.append((product, result.error_message or "Unknown failure"))

    if not successful_products:
        # Every attempted request failed - raise so caller knows ingestion failed
        errors_str = "; ".join(f"{p}: {err}" for p, err in failed_products)
        raise FirmsAPIError(f"All FIRMS product queries failed: {errors_str}")

    combined_df = (
        pd.concat(dataframes, ignore_index=True)
        if dataframes
        else pd.DataFrame(columns=FirmsClient.EXPECTED_COLUMNS + ["source_product"])
    )

    deduped_df = deduplicate_observations(combined_df)
    saved_rows = len(deduped_df)

    # Automatically ensure output parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    deduped_df.to_csv(output_path, index=False)

    columns = list(deduped_df.columns)
    date_range: Optional[Tuple[str, str]] = None
    if "acq_date" in deduped_df.columns and not deduped_df.empty:
        dates = deduped_df["acq_date"].dropna().astype(str)
        if not dates.empty:
            date_range = (dates.min(), dates.max())

    summary = IngestionSummary(
        products_attempted=list(products),
        successful_requests=successful_products,
        failed_requests=failed_products,
        total_raw_rows=total_raw_rows,
        saved_rows=saved_rows,
        columns=columns,
        date_range=date_range,
        output_path=output_path,
    )

    return summary


def main() -> None:
    """CLI entrypoint for running ingestion."""
    print("Initiating ThermalEye NASA FIRMS VIIRS Ingestion (Phase 1)...")
    try:
        summary = run_ingestion()
        print(summary.format_report())
        if summary.failed_requests:
            print("\nWarning: Some products encountered errors (see summary above).")
    except FirmsConfigError as e:
        print(f"\nConfiguration Error: {e}")
        print("Please check your .env file or reference .env.example.")
        raise SystemExit(1)
    except FirmsAPIError as e:
        print(f"\nAPI Ingestion Error: {e}")
        raise SystemExit(2)
    except Exception as e:
        print(f"\nUnexpected Error: {e}")
        raise SystemExit(3)


if __name__ == "__main__":
    main()
