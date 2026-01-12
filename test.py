import os
import uuid
import imageio
import numpy as np
from IPython.display import Image as ImageDisplay
import torch
from pytorch3d.transforms import rotation_6d_to_matrix

from inference import Inference, ready_gaussian_for_video_rendering, load_image, load_masks, display_image, make_scene, render_video, interactive_visualizer
from sam3d_objects.custom.utils import *

PATH = os.getcwd()
TAG = "hf"
config_path = f"{PATH}/../checkpoints/{TAG}/pipeline.yaml"
inference = Inference(config_path, compile=False)

IMAGE_PATH = f"{PATH}/images/segment1/image.jpg"
IMAGE_NAME = os.path.basename(os.path.dirname(IMAGE_PATH))

image = load_image(IMAGE_PATH)
masks = load_masks(os.path.dirname(IMAGE_PATH), extension=".png")

outputs = [inference(image, mask, seed=42) for mask in masks[1:]]

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

scene_gs = make_scene(*outputs)

# export posed gaussian splatting (as point cloud)
scene_gs.save_ply(f"{PATH}/gaussians/{IMAGE_NAME}_posed.ply")

scene_gs = ready_gaussian_for_video_rendering(scene_gs)
# export gaussian splatting (as point cloud)
scene_gs.save_ply(f"{PATH}/gaussians/multi/{IMAGE_NAME}.ply")

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

