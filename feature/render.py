from utils import *
from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import Gaussian
from sam3d_objects.model.backbone.tdfy_dit.utils import render_utils

from PIL import Image

num_views = 150
ply_path = f"../notebook/gaussians/single/segment1_5.ply"

yaws = []
pitchs = []
offset = (np.random.rand(), np.random.rand())

print(offset)

for i in range(num_views):
    y, p = sphere_hammersley_sequence(i, num_views, offset)
    yaws.append(y)
    pitchs.append(p)
radius = [2] * num_views
fov = [40] * num_views
views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]

gs = Gaussian([-1, -1, -1, 2, 2, 2]) # dummy aabb
gs.load_ply(ply_path)
# gs = ready_gaussian_for_video_rendering(gs)

extr, intr = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, radius, fov)
video = render_utils.render_frames(
    gs,
    extr,
    intr,
    {"resolution": 512, "bg_color": (0, 0, 0), "backend": "gsplat"},
)["color"]

for i, frame in enumerate(video):
    img = Image.fromarray(frame)
    os.makedirs("../debug/rendered_frames", exist_ok=True)
    img.save(f"../debug/rendered_frames/frame_{i:03d}.png")

extr = torch.stack(extr, dim=0)
intr = torch.stack(intr, dim=0)

# also save extrinsics and intrinsics as npy
np.save("../debug/rendered_frames/extrinsics.npy", extr.cpu().numpy())
np.save("../debug/rendered_frames/intrinsics.npy", intr.cpu().numpy())