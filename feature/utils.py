from typing import *
import numpy as np
import torch
from copy import deepcopy
import os
import utils3d

os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]
os.environ["LIDRA_SKIP_INIT"] = "true"

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]


def radical_inverse(base, n):
    val = 0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val


def halton_sequence(dim, n):
    return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]


def hammersley_sequence(dim, n, num_samples):
    return [n / num_samples] + halton_sequence(dim - 1, n)


def sphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]


def ready_gaussian_for_video_rendering(scene_gs, in_place=False, fix_alignment=False):
    if fix_alignment:
        scene_gs = _fix_gaussian_alignment(scene_gs, in_place=in_place)
    scene_gs = normalized_gaussian(scene_gs, in_place=fix_alignment)
    return scene_gs


def _fix_gaussian_alignment(scene_gs, in_place=False):
    if not in_place:
        scene_gs = deepcopy(scene_gs)

    device = scene_gs._xyz.device
    dtype = scene_gs._xyz.dtype
    scene_gs._xyz = (
        scene_gs._xyz
        @ torch.tensor(
            [
                [-1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
            ],
            device=device,
            dtype=dtype,
        ).T
    )
    return scene_gs


def normalized_gaussian(scene_gs, in_place=False, outlier_percentile=None):
    if not in_place:
        scene_gs = deepcopy(scene_gs)

    orig_xyz = scene_gs.get_xyz
    orig_scale = scene_gs.get_scaling

    active_mask = (scene_gs.get_opacity > 0.9).squeeze()
    inv_scale = (
        orig_xyz[active_mask].max(dim=0)[0] - orig_xyz[active_mask].min(dim=0)[0]
    ).max()
    norm_scale = orig_scale / inv_scale
    norm_xyz = orig_xyz / inv_scale

    if outlier_percentile is None:
        lower_bound_xyz = torch.min(norm_xyz[active_mask], dim=0)[0]
        upper_bound_xyz = torch.max(norm_xyz[active_mask], dim=0)[0]
    else:
        lower_bound_xyz = torch.quantile(
            norm_xyz[active_mask],
            outlier_percentile,
            dim=0,
        )
        upper_bound_xyz = torch.quantile(
            norm_xyz[active_mask],
            1.0 - outlier_percentile,
            dim=0,
        )

    center = (lower_bound_xyz + upper_bound_xyz) / 2
    norm_xyz = norm_xyz - center
    scene_gs.from_xyz(norm_xyz)
    scene_gs.mininum_kernel_size /= inv_scale.item()
    scene_gs.from_scaling(norm_scale)
    return scene_gs


def yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs):
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)
    extrinsics = []
    intrinsics = []
    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        fov = torch.deg2rad(torch.tensor(float(fov))).cuda()
        yaw = torch.tensor(float(yaw)).cuda()
        pitch = torch.tensor(float(pitch)).cuda()
        orig = (
            torch.tensor(
                [
                    torch.sin(yaw) * torch.cos(pitch),
                    torch.sin(pitch),
                    torch.cos(yaw) * torch.cos(pitch),
                ]
            ).cuda()
            * r
        )
        extr = utils3d.torch.extrinsics_look_at(
            orig,
            torch.tensor([0, 0, 0]).float().cuda(),
            torch.tensor([0, 1, 0]).float().cuda(),
        )
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        extrinsics.append(extr)
        intrinsics.append(intr)
    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]
    return extrinsics, intrinsics