"""
Utility script to prepare scene-level training data for ShortCut model.

This script helps convert your raw data into the format expected by the
scene-level training script.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Any
import json
import torch
from tqdm import tqdm


def prepare_scene_data(
    scene_id: str,
    objects: List[Dict[str, Any]],
    output_dir: Path,
    vae_encoder=None,
):
    """
    Prepare data for a single scene.

    Args:
        scene_id: Scene identifier
        objects: List of object dictionaries, each containing:
                 - geometry: 3D mesh/point cloud
                 - translation: [3] position
                 - scale: [3] scale
                 - rotation: [3, 3] rotation matrix or [6] 6D rotation
        output_dir: Output directory for this scene
        vae_encoder: VAE encoder to convert geometry to latent codes
                     (optional, if latents are already provided)

    Returns:
        Dict containing the prepared scene data
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    num_objects = len(objects)
    print(f"Processing scene {scene_id} with {num_objects} objects...")

    # Prepare latent dictionaries
    latents = {
        'shape': [],
        'translation': [],
        'scale': [],
        '6drotation_normalized': [],
    }

    # Process each object
    for i, obj in enumerate(objects):
        # 1. Encode shape to latent if encoder is provided
        if 'latent' in obj:
            # Latent already provided
            shape_latent = obj['latent']
        elif vae_encoder is not None:
            # Encode geometry to latent
            geometry = obj['geometry']
            with torch.no_grad():
                shape_latent = vae_encoder.encode(geometry)
        else:
            raise ValueError(
                f"Object {i} has no latent and no VAE encoder provided"
            )

        # 2. Get translation
        translation = torch.tensor(obj['translation'], dtype=torch.float32)
        if translation.dim() == 1:
            translation = translation.unsqueeze(0)  # [1, 3]

        # 3. Get scale
        scale = torch.tensor(obj['scale'], dtype=torch.float32)
        if scale.dim() == 1:
            scale = scale.unsqueeze(0)  # [1, 3]

        # 4. Get rotation (convert to 6D if needed)
        if '6drotation' in obj:
            rotation_6d = torch.tensor(obj['6drotation'], dtype=torch.float32)
        elif 'rotation_matrix' in obj:
            from pytorch3d.transforms import matrix_to_rotation_6d
            rotation_matrix = torch.tensor(obj['rotation_matrix'], dtype=torch.float32)
            rotation_6d = matrix_to_rotation_6d(rotation_matrix)
        else:
            raise ValueError(f"Object {i} has no rotation information")

        if rotation_6d.dim() == 1:
            rotation_6d = rotation_6d.unsqueeze(0)  # [1, 6]

        # Add to lists
        latents['shape'].append(shape_latent)
        latents['translation'].append(translation)
        latents['scale'].append(scale)
        latents['6drotation_normalized'].append(rotation_6d)

    # Stack all objects
    latents = {
        'shape': torch.stack(latents['shape']),  # [num_objects, C, tokens]
        'translation': torch.stack(latents['translation']),  # [num_objects, 1, 3]
        'scale': torch.stack(latents['scale']),  # [num_objects, 1, 3]
        '6drotation_normalized': torch.stack(latents['6drotation_normalized']),  # [num_objects, 1, 6]
    }

    # Save latents
    latents_path = output_dir / 'latents.pt'
    torch.save(latents, latents_path)
    print(f"  Saved latents to {latents_path}")

    # Save metadata
    metadata = {
        'scene_id': scene_id,
        'num_objects': num_objects,
        'object_ids': [obj.get('id', f'obj_{i}') for i, obj in enumerate(objects)],
    }
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {metadata_path}")

    return {
        'latents': latents,
        'metadata': metadata,
    }


def prepare_dataset(
    input_scenes_file: str,
    output_root: str,
    split: str = 'train',
    vae_checkpoint: str = None,
):
    """
    Prepare entire dataset for scene-level training.

    Args:
        input_scenes_file: JSON file containing scene data
        output_root: Root output directory
        split: Data split name ('train', 'val', 'test')
        vae_checkpoint: Path to VAE checkpoint (if needed for encoding)
    """
    output_root = Path(output_root)
    output_dir = output_root / split
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load VAE encoder if checkpoint provided
    vae_encoder = None
    if vae_checkpoint is not None:
        print(f"Loading VAE encoder from {vae_checkpoint}")
        # Load your VAE encoder here
        # vae_encoder = YourVAE.load_from_checkpoint(vae_checkpoint)
        # vae_encoder.eval()
        # vae_encoder.cuda()
        raise NotImplementedError(
            "Implement VAE loading for your specific model"
        )

    # Load scene data
    print(f"Loading scenes from {input_scenes_file}")
    with open(input_scenes_file, 'r') as f:
        scenes_data = json.load(f)

    # Prepare each scene
    scene_list = []
    for scene_data in tqdm(scenes_data, desc=f"Processing {split} scenes"):
        scene_id = scene_data['scene_id']
        objects = scene_data['objects']

        # Prepare this scene
        prepare_scene_data(
            scene_id=scene_id,
            objects=objects,
            output_dir=output_dir / scene_id,
            vae_encoder=vae_encoder,
        )

        scene_list.append(scene_id)

    # Save scene list
    scene_list_file = output_dir / 'scenes.txt'
    with open(scene_list_file, 'w') as f:
        for scene_id in scene_list:
            f.write(f"{scene_id}\n")

    print(f"\nDataset preparation complete!")
    print(f"  Split: {split}")
    print(f"  Num scenes: {len(scene_list)}")
    print(f"  Output dir: {output_dir}")
    print(f"  Scene list: {scene_list_file}")


def verify_scene_data(scene_dir: Path):
    """
    Verify that a scene directory has the correct format.

    Args:
        scene_dir: Path to scene directory
    """
    print(f"\nVerifying scene: {scene_dir.name}")

    # Check latents file
    latents_file = scene_dir / 'latents.pt'
    if not latents_file.exists():
        print(f"  ❌ Missing latents.pt")
        return False
    else:
        latents = torch.load(latents_file)
        print(f"  ✓ Found latents.pt")

        # Check keys
        required_keys = ['shape', 'translation', 'scale', '6drotation_normalized']
        for key in required_keys:
            if key not in latents:
                print(f"    ❌ Missing key: {key}")
                return False
            else:
                tensor = latents[key]
                print(f"    ✓ {key}: {list(tensor.shape)}")

        # Check consistency
        num_objects = latents['shape'].shape[0]
        for key in required_keys:
            if latents[key].shape[0] != num_objects:
                print(f"    ❌ Inconsistent num_objects for {key}")
                return False

        print(f"  ✓ All {num_objects} objects have consistent shapes")

    # Check metadata file
    metadata_file = scene_dir / 'metadata.json'
    if not metadata_file.exists():
        print(f"  ⚠ Missing metadata.json (optional)")
    else:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"  ✓ Found metadata.json")
        print(f"    Scene ID: {metadata.get('scene_id', 'N/A')}")
        print(f"    Num objects: {metadata.get('num_objects', 'N/A')}")

    print(f"  ✅ Scene verification passed!")
    return True


def example_scene_creation():
    """
    Create an example scene with random data for testing.
    """
    print("\n" + "=" * 80)
    print("Creating example scene for testing")
    print("=" * 80)

    # Create example scene with 5 random objects
    num_objects = 5
    objects = []

    for i in range(num_objects):
        obj = {
            'id': f'obj_{i}',
            'latent': torch.randn(256, 256),  # [C, tokens] - shape latent
            'translation': torch.randn(3).numpy().tolist(),
            'scale': torch.abs(torch.randn(3)).numpy().tolist(),
            '6drotation': torch.randn(6).numpy().tolist(),
        }
        objects.append(obj)

    # Prepare scene
    output_dir = Path('./example_data/train/scene_example')
    prepare_scene_data(
        scene_id='scene_example',
        objects=objects,
        output_dir=output_dir,
        vae_encoder=None,
    )

    # Verify
    verify_scene_data(output_dir)

    print("\n" + "=" * 80)
    print("Example scene created successfully!")
    print(f"Location: {output_dir}")
    print("You can use this as a template for your own data.")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare scene data for ShortCut training'
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Prepare dataset command
    prepare_parser = subparsers.add_parser('prepare', help='Prepare dataset')
    prepare_parser.add_argument(
        '--input_scenes',
        type=str,
        required=True,
        help='JSON file containing scene data',
    )
    prepare_parser.add_argument(
        '--output_root',
        type=str,
        required=True,
        help='Root output directory',
    )
    prepare_parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='Data split',
    )
    prepare_parser.add_argument(
        '--vae_checkpoint',
        type=str,
        default=None,
        help='Path to VAE checkpoint (if encoding is needed)',
    )

    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify scene data')
    verify_parser.add_argument(
        '--scene_dir',
        type=str,
        required=True,
        help='Path to scene directory to verify',
    )

    # Example command
    example_parser = subparsers.add_parser('example', help='Create example scene')

    args = parser.parse_args()

    if args.command == 'prepare':
        prepare_dataset(
            input_scenes_file=args.input_scenes,
            output_root=args.output_root,
            split=args.split,
            vae_checkpoint=args.vae_checkpoint,
        )
    elif args.command == 'verify':
        verify_scene_data(Path(args.scene_dir))
    elif args.command == 'example':
        example_scene_creation()
    else:
        parser.print_help()
