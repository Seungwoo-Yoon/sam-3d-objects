#!/usr/bin/env python3
"""
Download Google Scanned Objects (GSO) Dataset

This script downloads the Google Scanned Objects dataset, which contains over 1000
high-quality 3D scanned household items.

Dataset Info:
- Official Page: https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/
- Paper: https://arxiv.org/abs/2204.11918
- License: Creative Commons

Usage:
    python download_gso_dataset.py --output-dir ./google_scanned_objects
    python download_gso_dataset.py --output-dir ./gso --objects-list objects.txt
"""

import argparse
import os
import sys
import zipfile
import subprocess
from pathlib import Path
from typing import Optional, List
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Mirror URL for GSO dataset
GSO_MIRROR_URL = "https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/tars/google_scanned_objects.zip"

# Alternative: Official Google short link
GSO_OFFICIAL_URL = "https://goo.gle/scanned-objects"


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file from URL using wget or curl.

    Args:
        url: URL to download from
        output_path: Local path to save the file
        chunk_size: Chunk size for download (not used with wget/curl)

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Downloading from {url}")
    logger.info(f"Saving to {output_path}")

    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try using wget first (shows progress bar)
    try:
        result = subprocess.run(
            ['wget', '-c', '-O', str(output_path), url],
            check=True,
            capture_output=False
        )
        logger.info("Download completed successfully with wget")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"wget failed or not available: {e}")

    # Try using curl
    try:
        result = subprocess.run(
            ['curl', '-L', '-C', '-', '-o', str(output_path), url],
            check=True,
            capture_output=False
        )
        logger.info("Download completed successfully with curl")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"curl failed or not available: {e}")

    # Fallback to Python urllib
    try:
        import urllib.request
        logger.info("Downloading with urllib (no progress bar)...")

        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100.0 / total_size, 100.0) if total_size > 0 else 0
            logger.info(f"Progress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB)")

        urllib.request.urlretrieve(url, str(output_path), reporthook=report_progress)
        logger.info("Download completed successfully with urllib")
        return True
    except Exception as e:
        logger.error(f"urllib download failed: {e}")
        return False


def extract_zip(zip_path: Path, output_dir: Path, objects_list: Optional[List[str]] = None) -> bool:
    """
    Extract ZIP file to output directory.

    Args:
        zip_path: Path to ZIP file
        output_dir: Directory to extract to
        objects_list: Optional list of specific object names to extract

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Extracting {zip_path} to {output_dir}")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            if objects_list is None:
                # Extract all files
                zip_ref.extractall(output_dir)
                logger.info(f"Extracted all files to {output_dir}")
            else:
                # Extract only specific objects
                all_files = zip_ref.namelist()
                objects_set = set(objects_list)

                extracted_count = 0
                for file_path in all_files:
                    # Check if this file belongs to one of the requested objects
                    parts = Path(file_path).parts
                    if len(parts) > 1:
                        object_name = parts[1] if parts[0] == 'google_scanned_objects' else parts[0]
                        if object_name in objects_set:
                            zip_ref.extract(file_path, output_dir)
                            extracted_count += 1

                logger.info(f"Extracted {extracted_count} files for {len(objects_list)} objects")

        return True
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False


def verify_dataset(output_dir: Path) -> None:
    """
    Verify the downloaded dataset structure.

    Args:
        output_dir: Root directory of the dataset
    """
    logger.info("Verifying dataset structure...")

    # The extracted directory might be 'google_scanned_objects'
    gso_dir = output_dir / 'google_scanned_objects'
    if not gso_dir.exists():
        gso_dir = output_dir

    # Count object directories
    object_dirs = [d for d in gso_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    logger.info(f"Found {len(object_dirs)} object directories")

    # Check a few objects for mesh files
    mesh_formats = ['.obj', '.ply', '.stl']
    objects_with_meshes = 0

    for obj_dir in object_dirs[:10]:  # Check first 10
        has_mesh = False
        for mesh_ext in mesh_formats:
            if list(obj_dir.rglob(f'*{mesh_ext}')):
                has_mesh = True
                break
        if has_mesh:
            objects_with_meshes += 1

    logger.info(f"Verified: {objects_with_meshes}/10 sample objects have mesh files")

    if objects_with_meshes == 0:
        logger.warning("No mesh files found in sample objects. Dataset structure might be different.")
    else:
        logger.info("Dataset verification successful!")


def load_objects_list(list_file: Path) -> List[str]:
    """
    Load list of object names from a text file.

    Args:
        list_file: Path to text file with one object name per line

    Returns:
        List of object names
    """
    with open(list_file, 'r') as f:
        objects = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return objects


def main():
    parser = argparse.ArgumentParser(
        description='Download Google Scanned Objects (GSO) dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download entire dataset
  python download_gso_dataset.py --output-dir ./google_scanned_objects

  # Download specific objects only
  python download_gso_dataset.py --output-dir ./gso --objects-list my_objects.txt

  # Use official URL (may redirect)
  python download_gso_dataset.py --output-dir ./gso --use-official-url
        """
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for the dataset'
    )

    parser.add_argument(
        '--objects-list',
        type=str,
        default=None,
        help='Text file with list of specific objects to extract (one per line)'
    )

    parser.add_argument(
        '--use-official-url',
        action='store_true',
        help='Use official Google URL instead of mirror (may require manual steps)'
    )

    parser.add_argument(
        '--keep-zip',
        action='store_true',
        help='Keep the downloaded ZIP file after extraction'
    )

    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download and only extract existing ZIP file'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / 'google_scanned_objects.zip'

    # Load objects list if provided
    objects_list = None
    if args.objects_list:
        list_file = Path(args.objects_list)
        if not list_file.exists():
            logger.error(f"Objects list file not found: {list_file}")
            return 1
        objects_list = load_objects_list(list_file)
        logger.info(f"Will extract {len(objects_list)} specific objects")

    # Download dataset
    if not args.skip_download:
        url = GSO_OFFICIAL_URL if args.use_official_url else GSO_MIRROR_URL

        if args.use_official_url:
            logger.warning(
                "Using official URL. This may redirect you to Ignition Fuel website.\n"
                "You may need to download manually and place the file at:\n"
                f"  {zip_path}\n"
                "Then run again with --skip-download flag."
            )

        if zip_path.exists():
            logger.info(f"ZIP file already exists: {zip_path}")
            response = input("Re-download? (y/N): ").strip().lower()
            if response != 'y':
                logger.info("Skipping download")
            else:
                if not download_file(url, zip_path):
                    logger.error("Download failed")
                    return 1
        else:
            if not download_file(url, zip_path):
                logger.error("Download failed")
                return 1

    # Verify ZIP file exists
    if not zip_path.exists():
        logger.error(f"ZIP file not found: {zip_path}")
        logger.error("Please download the dataset manually or check the URL")
        return 1

    # Extract dataset
    if not extract_zip(zip_path, output_dir, objects_list):
        logger.error("Extraction failed")
        return 1

    # Verify dataset
    verify_dataset(output_dir)

    # Clean up ZIP file if requested
    if not args.keep_zip:
        logger.info(f"Removing ZIP file: {zip_path}")
        zip_path.unlink()

    logger.info("=" * 60)
    logger.info("GSO Dataset download complete!")
    logger.info(f"Dataset location: {output_dir}")
    logger.info("=" * 60)
    logger.info("\nTo use with FoundationPoseDataset:")
    logger.info(f"  dataset = FoundationPoseDataset(")
    logger.info(f"      data_root='/path/to/foundationpose',")
    logger.info(f"      gso_root='{output_dir.absolute()}/google_scanned_objects',")
    logger.info(f"      load_meshes=True,")
    logger.info(f"  )")

    return 0


if __name__ == '__main__':
    sys.exit(main())
