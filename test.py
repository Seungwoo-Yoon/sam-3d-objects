import torch
import os
import uuid
import json
import imageio
import trimesh
import numpy as np
from IPython.display import Image as ImageDisplay
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_matrix

from inference import Inference, ready_gaussian_for_video_rendering, load_image, load_masks, display_image, make_scene, render_video, interactive_visualizer
from sam3d_objects.custom.utils import *

PATH = os.getcwd()
TAG = "hf"
config_path = f"{PATH}/../checkpoints/{TAG}/pipeline.yaml"
inference = Inference(config_path, compile=False)

IMAGE_PATH = f"{PATH}/images/foundationpose/image.png"
IMAGE_NAME = os.path.basename(os.path.dirname(IMAGE_PATH))

image = load_image(IMAGE_PATH)
masks = load_masks(os.path.dirname(IMAGE_PATH), extension=".png")
print(masks[0])

pointmap = np.load('../test_pointmap.npy')
pointmap = torch.from_numpy(pointmap).cuda()

print(pointmap.shape)

outputs = [inference(image, mask, seed=20, pointmap=pointmap) for mask in masks]

if 'voxel' in outputs[0]:
    for i, output in enumerate(outputs):
        pointmap_pc = output["pointmap"].reshape(-1, 3).cpu().numpy()
        sam3d_pc = output["voxel"].cpu().numpy()

        scale = output["scale"].cpu().numpy()[0, 0]
        translation = output["translation"].cpu().numpy()
        rotation = rotation_6d_to_matrix(output["6drotation_normalized"][0].cpu()).numpy()[0]

        sam3d_pc = (sam3d_pc * scale) @ rotation.T + translation

        save_multiple_pcs([sam3d_pc, pointmap_pc],
                        f"../debug/pointmap_sam3d_compare/{i}.ply",
                        colors=[(255, 0, 0, 255), (0, 255, 0, 255)])

for i in range(len(outputs)):
    glb_path = f"{PATH}/meshes/{IMAGE_NAME}/{i+1}.glb"
    os.makedirs(os.path.dirname(glb_path), exist_ok=True)
    outputs[i]['glb'].export(glb_path)

# Combine all objects into a single GLB using model output transforms
combined_meshes = []
for i, output in enumerate(outputs):
    glb = output['glb']
    if isinstance(glb, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g.copy() for g in glb.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    elif isinstance(glb, trimesh.Trimesh):
        mesh = glb.copy()
    else:
        continue

    s = output['scale'].detach().cpu().numpy().flatten()[:3]
    t = output['translation'].detach().cpu().numpy().flatten()[:3]
    R = quaternion_to_matrix(output['rotation']).detach().cpu().numpy().reshape(3, 3)

    # Build 4x4 transform (trimesh column-vector: v' = M @ v)
    # Matching PyTorch3D compose_transform: p' = (p * s) @ R + t
    M = np.eye(4)
    M[:3, :3] = R.T @ np.diag(s)
    M[:3, 3] = t

    mesh.apply_transform(M)
    combined_meshes.append(mesh)

combined = trimesh.util.concatenate(combined_meshes)
combined_glb_path = f"{PATH}/meshes/{IMAGE_NAME}/combined.glb"
os.makedirs(os.path.dirname(combined_glb_path), exist_ok=True)
combined.export(combined_glb_path)
print(f"Combined GLB saved to {combined_glb_path}")

for i, output in enumerate(outputs):
    print(f'scale: {output["scale"]}')

    min_x = output["gs"].get_xyz[:, 0].min().item()
    max_x = output["gs"].get_xyz[:, 0].max().item()
    min_y = output["gs"].get_xyz[:, 1].min().item()
    max_y = output["gs"].get_xyz[:, 1].max().item()
    min_z = output["gs"].get_xyz[:, 2].min().item()
    max_z = output["gs"].get_xyz[:, 2].max().item()


    print(f'gaussian_scale: {max_x - min_x}, {max_y - min_y}, {max_z - min_z}')
    # print(f'intrinsic: {output["intrinsic"]}')
    output["gaussian"][0].save_ply(f"{PATH}/gaussians/single/{IMAGE_NAME}_{i}.ply")

outputs_copy = outputs.copy()

scene_gs = make_scene(*outputs)

# export posed gaussian splatting (as point cloud)
scene_gs.save_ply(f"{PATH}/gaussians/{IMAGE_NAME}_posed.ply")

scene_gs = ready_gaussian_for_video_rendering(scene_gs)
# export gaussian splatting (as point cloud)
scene_gs.save_ply(f"{PATH}/gaussians/multi/{IMAGE_NAME}.ply")

with open('../test_latents.json', 'r') as f:
    test_latents = json.load(f)
    scale = test_latents['scale']
    translation = test_latents['translation']
    rotation_6d_normalized = test_latents['6drotation_normalized']

outputs = outputs_copy

for i in range(len(outputs)):
    outputs[i]['scale'] = torch.tensor(scale[i]).unsqueeze(0).cuda() * 2
    outputs[i]['translation'] = torch.tensor(translation[i]).unsqueeze(0).cuda()
    outputs[i]['rotation'] = matrix_to_quaternion(rotation_6d_to_matrix(torch.tensor(rotation_6d_normalized[i]).unsqueeze(0).cuda()))

scene_gs = make_scene(*outputs)
scene_gs = ready_gaussian_for_video_rendering(scene_gs)
scene_gs.save_ply(f"{PATH}/gaussians/multi/{IMAGE_NAME}_from_latents.ply")



video = render_video(
    scene_gs,
    r=1,
    fov=60,
    resolution=512,
)["color"]

# save video as gif
imageio.mimsave(
    os.path.join(f"{PATH}/gaussians/multi/{IMAGE_NAME}.gif"),
    video,
    format="GIF",
    duration=1000 / 30,  # default assuming 30fps from the input MP4
    loop=0,  # 0 means loop indefinitely
)

