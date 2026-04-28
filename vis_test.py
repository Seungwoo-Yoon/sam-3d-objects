"""
reward.py의 mask_interior 로직 검증용 시각화 스크립트.

FoundationPoseDataset에서 conditionals (image, object_masks)를 가져온 뒤,
reward.py의 pointmap_coverage_reward와 동일한 방식으로 mask_interior를 계산하고
image에 적용해 각 object별 visible 영역을 저장한다.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

from foundation_pose_dataset import FoundationPoseDataset


def compute_mask_interior(mask: torch.Tensor) -> torch.Tensor:
    """
    reward.py의 pointmap_coverage_reward와 동일한 erosion 로직.
    mask: (H, W) float tensor
    returns: (H, W) bool tensor
    """
    mask_f = mask.float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    mask_interior = (
        -F.max_pool2d(-mask_f, kernel_size=3, stride=1, padding=1)
    ).squeeze() > 0.5  # (H, W) bool
    return mask_interior


def apply_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    image: (H, W, 3) float [0,1]
    mask: (H, W) bool
    returns: (H, W, 3) float - masked 영역만 표시, 나머지는 어둡게
    """
    result = image.copy()
    # result[~mask] = result[~mask] * 1.0  # 마스크 밖은 어둡게
    result[~mask] = 1.0  # 마스크 밖은 흰색으로 (원본 이미지가 어두운 경우 대비)
    return result


def visualize_sample(sample, save_dir: Path, sample_idx: int):
    cond = sample['conditionals']
    image = cond['image'].numpy()  # (H, W, 3) float [0,1]

    if 'object_masks' not in cond:
        print(f"  [skip] object_masks 없음")
        return

    masks = cond['object_masks']  # (N, H, W)
    N = masks.shape[0]

    print(f"  objects: {N}, image: {image.shape}")

    # 원본 이미지 저장
    # plt.imsave(str(save_dir / f'{sample_idx:03d}_original.png'), image)

    # 전체 mask (원본 그대로) vs mask_interior 비교
    # for obj_idx in range(N):
    #     mask_raw = masks[obj_idx]  # (H, W)

    #     # 원본 마스크
    #     raw_np = mask_raw.numpy().astype(bool)
    #     if not raw_np.any():
    #         continue  # 빈 마스크 skip

    #     # reward.py와 동일한 mask_interior
    #     mask_interior = compute_mask_interior(mask_raw)
    #     interior_np = mask_interior.numpy()

    #     # image에 각각 적용
    #     img_raw = apply_mask_to_image(image, raw_np)
    #     img_interior = apply_mask_to_image(image, interior_np)

    #     # 나란히 비교 저장
    #     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    #     axes[0].imshow(image)
    #     axes[0].set_title('Original')
    #     axes[0].axis('off')

    #     axes[1].imshow(img_raw)
    #     axes[1].set_title(f'obj{obj_idx} raw mask\n({raw_np.sum()} px)')
    #     axes[1].axis('off')

    #     axes[2].imshow(img_interior)
    #     axes[2].set_title(f'obj{obj_idx} mask_interior (eroded)\n({interior_np.sum()} px)')
    #     axes[2].axis('off')

    #     plt.tight_layout()
    #     out_path = save_dir / f'{sample_idx:03d}_obj{obj_idx:02d}_mask_compare.png'
    #     plt.savefig(str(out_path), dpi=150)
    #     plt.close(fig)

    #     print(f"    obj{obj_idx}: raw={raw_np.sum()} px  interior={interior_np.sum()} px  -> {out_path.name}")

    # 전체 interior mask overlay (모든 object 합쳐서)
    all_interior = torch.zeros(masks.shape[1], masks.shape[2], dtype=torch.bool)
    for obj_idx in range(N):
        mi = compute_mask_interior(masks[obj_idx])
        all_interior |= mi

    img_all = apply_mask_to_image(image, all_interior.numpy())
    
    # plt.imsave(str(save_dir / f'{sample_idx:03d}_all_interior.png'), img_all)

    # 전체 interior 마스크가 적용된 이미지 저장
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(img_all)
    axes[1].set_title('All Interior Masks')
    axes[1].axis('off')

    plt.tight_layout()
    out_path = save_dir / f'{sample_idx:03d}_all_interior.png'
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)

    print(f"    all interior overlay -> {out_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='/data0/intern0/seungwoo/sam-3d-objects/foundationpose_test/')
    parser.add_argument('--save-dir', default='/data0/intern0/seungwoo/sam-3d-objects/debug/vis_mask_interior')
    parser.add_argument('--num-samples', type=int, default=5, help='몇 개 샘플 시각화할지')
    parser.add_argument('--sample-idx', type=int, default=None, help='특정 샘플 인덱스 (None이면 0~num_samples)')
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = FoundationPoseDataset(
        data_root=args.data_root,
        max_objects_per_scene=32,
        load_images=True,
        load_depth=True,
        load_masks=True,
    )
    print(f"Dataset size: {len(dataset)}")

    if args.sample_idx is not None:
        indices = [args.sample_idx]
    else:
        indices = list(range(min(args.num_samples, len(dataset))))

    for idx in indices:
        print(f"\n[{idx}] {dataset.scenes[idx]['scene_id']}")
        try:
            sample = dataset[idx]
            visualize_sample(sample, save_dir, idx)
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n저장 완료: {save_dir}")


if __name__ == '__main__':
    main()
