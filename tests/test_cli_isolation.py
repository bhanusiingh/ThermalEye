"""Tests for CLI output path isolation."""
import subprocess
import sys
from pathlib import Path

def test_firms_ingestion_cli_help():
    result = subprocess.run(
        [sys.executable, "run_firms_ingestion.py", "--help"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--output" in result.stdout
    assert "--date" in result.stdout
    assert "--product" in result.stdout
    assert "--bbox" in result.stdout
    assert "--day-range" in result.stdout

def test_temporal_grouping_cli_help():
    result = subprocess.run(
        [sys.executable, "run_temporal_grouping.py", "--help"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--output-events" in result.stdout
    assert "--output-grouped" in result.stdout

def test_sentinel2_retrieval_cli_help():
    result = subprocess.run(
        [sys.executable, "run_sentinel2_retrieval.py", "--help"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--output-dir" in result.stdout
    assert "--meta-path" in result.stdout

def test_osm_enrichment_cli_help():
    result = subprocess.run(
        [sys.executable, "run_osm_enrichment.py", "--help"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--output" in result.stdout

def test_fusion_cli_help():
    result = subprocess.run(
        [sys.executable, "run_fusion.py", "--help"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--output" in result.stdout


def test_firms_ingestion_cli_invalid_date():
    result = subprocess.run(
        [sys.executable, "run_firms_ingestion.py", "--date", "2026/10/10"],
        capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "Error: --date must be in YYYY-MM-DD format." in result.stdout

def test_firms_ingestion_cli_invalid_day_range():
    result = subprocess.run(
        [sys.executable, "run_firms_ingestion.py", "--day-range", "15"],
        capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "Error: --day-range must be between 1 and 10." in result.stdout

def test_firms_ingestion_cli_invalid_product():
    result = subprocess.run(
        [sys.executable, "run_firms_ingestion.py", "--product", "INVALID_PROD"],
        capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "Error: Unsupported product" in result.stdout

def test_firms_ingestion_cli_invalid_bbox():
    result = subprocess.run(
        [sys.executable, "run_firms_ingestion.py", "--bbox", "97.4,6.7,68.1,37.1"],
        capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "Error: Invalid bbox format" in result.stdout
