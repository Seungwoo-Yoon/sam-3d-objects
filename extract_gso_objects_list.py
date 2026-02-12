#!/usr/bin/env python3
"""
Extract list of GSO objects used in FoundationPose dataset.

This script scans the FoundationPose dataset and creates a list of all GSO objects
that are actually used. This list can then be used with download_gso_dataset.py to
download only the necessary objects.

Usage:
    python extract_gso_objects_list.py --data-root ./foundationpose --output gso_objects.txt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Set
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_object_name_from_path(prim_path: str) -> tuple[str, str] | None:
    """
    Extract object name and dataset type from prim path.

    Returns:
        Tuple of (object_name, dataset_type) or None if not found
    """
    prefixes = ['objaverse_', 'gso_', 'shapenet_', 'ycb_']

    for prefix in prefixes:
        if f'/{prefix}' in prim_path:
            parts = prim_path.split(f'/{prefix}')
            if len(parts) > 1:
                name = parts[1].split('/')[0]
                dataset_type = prefix.rstrip('_')
                return (name, dataset_type)

    if '/World/objects/' in prim_path:
        parts = prim_path.split('/World/objects/')
        if len(parts) > 1:
            name = parts[1].split('/')[0]
            return (name, 'unknown')

    return None


def scan_foundationpose_dataset(data_root: Path) -> Set[str]:
    """
    Scan FoundationPose dataset to find all GSO objects.

    Args:
        data_root: Root directory of FoundationPose dataset

    Returns:
        Set of GSO object names
    """
    gso_objects = set()

    # Find all top-level object directories
    object_dirs = sorted([d for d in data_root.iterdir()
                         if d.is_dir() and d.name.isdigit()])

    logger.info(f"Found {len(object_dirs)} object directories")

    for obj_dir in object_dirs:
        # Check states.json
        states_file = obj_dir / 'states.json'
        if not states_file.exists():
            continue

        try:
            with open(states_file, 'r') as f:
                states = json.load(f)

            # Extract GSO objects from states
            if 'objects' in states:
                for obj_name, obj_data in states['objects'].items():
                    prim_path = obj_data.get('prim_path', '')
                    if 'gso_' in prim_path:
                        # Extract just the object name
                        obj_info = extract_object_name_from_path(prim_path)
                        if obj_info and obj_info[1] == 'gso':
                            gso_objects.add(obj_info[0])

            # Also check instance segmentation mappings
            scene_dirs = sorted([d for d in obj_dir.iterdir()
                               if d.is_dir() and d.name.startswith('scene-')])

            for scene_dir in scene_dirs:
                render_dirs = sorted([d for d in scene_dir.iterdir()
                                    if d.is_dir() and d.name.startswith('RenderProduct_')])

                for render_dir in render_dirs:
                    seg_dir = render_dir / 'instance_segmentation'
                    if seg_dir.exists():
                        # Check first mapping file
                        mapping_files = sorted(seg_dir.glob('instance_segmentation_mapping_*.json'))
                        if mapping_files:
                            with open(mapping_files[0], 'r') as f:
                                id_mapping = json.load(f)

                            for prim_path in id_mapping.values():
                                if 'gso_' in prim_path:
                                    obj_info = extract_object_name_from_path(prim_path)
                                    if obj_info and obj_info[1] == 'gso':
                                        gso_objects.add(obj_info[0])

        except Exception as e:
            logger.warning(f"Error processing {obj_dir}: {e}")
            continue

    return gso_objects


def main():
    parser = argparse.ArgumentParser(
        description='Extract list of GSO objects from FoundationPose dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # Extract GSO objects list
  python extract_gso_objects_list.py --data-root ./foundationpose --output gso_objects.txt

  # Then download only those objects
  python download_gso_dataset.py --output-dir ./gso --objects-list gso_objects.txt
        """
    )

    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='Root directory of FoundationPose dataset'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='gso_objects.txt',
        help='Output text file for GSO objects list (default: gso_objects.txt)'
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        logger.error(f"Data root not found: {data_root}")
        return 1

    logger.info(f"Scanning FoundationPose dataset: {data_root}")

    # Scan dataset
    gso_objects = scan_foundationpose_dataset(data_root)

    if not gso_objects:
        logger.warning("No GSO objects found in the dataset")
        return 1

    # Sort for consistency
    gso_objects = sorted(gso_objects)

    # Save to file
    output_file = Path(args.output)
    with open(output_file, 'w') as f:
        f.write("# GSO objects used in FoundationPose dataset\n")
        f.write(f"# Total: {len(gso_objects)} objects\n")
        f.write("#\n")
        for obj_name in gso_objects:
            f.write(f"{obj_name}\n")

    logger.info(f"Found {len(gso_objects)} unique GSO objects")
    logger.info(f"Saved to: {output_file}")
    logger.info("\nSample objects:")
    for obj_name in list(gso_objects)[:10]:
        logger.info(f"  - {obj_name}")
    if len(gso_objects) > 10:
        logger.info(f"  ... and {len(gso_objects) - 10} more")

    logger.info("\nNext steps:")
    logger.info(f"  python download_gso_dataset.py --output-dir ./gso --objects-list {output_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
