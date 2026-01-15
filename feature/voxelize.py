from utils import *
from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import Gaussian
import numpy as np
import os

ply_path = f"../notebook/gaussians/single/segment1_6.ply"

gs = Gaussian([-1, -1, -1, 2, 2, 2]) # dummy aabb
gs.load_ply(ply_path)

vertices = gs.get_xyz.cpu().numpy()

vertices = np.clip(np.asarray(vertices), -0.5 + 1e-6, 0.5 - 1e-6)

# normalize to [0, 1]
vertices = vertices + 0.5

# occupancy grid (64^3)
grid_size = 64
occupancy_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
indices = (vertices * grid_size).astype(np.int32)
indices = np.clip(indices, 0, grid_size - 1)
occupancy_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1

# save occupancy grid as .npy file
os.makedirs("../debug/occupancy_grids", exist_ok=True)
np.save("../debug/occupancy_grids/segment1_6_occupancy_grid.npy", occupancy_grid)