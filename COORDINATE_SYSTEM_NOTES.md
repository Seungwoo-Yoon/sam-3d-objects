# Coordinate System Compatibility: FoundationPose ↔ SAM3D

## Summary

**FoundationPose** provides poses in **metric world coordinates**, while **SAM3D** expects poses in a **scene-normalized canonical space**. This document explains the conversion.

## Coordinate Systems

### Foundation Pose
- **Frame**: World coordinates (metric, meters)
- **Poses**: Object poses are in world space
  - `translation`: [x, y, z] in meters
  - `rotation_matrix`: 3x3 rotation matrix
  - `scale`: [sx, sy, sz] object scale factors
- **Camera**: Camera view transform in world space
- **Convention**: USD/Omniverse (column-major matrices, Y-up typically)

### SAM3D
- **Frame**: Scene-normalized canonical space
- **Poses**: Object poses relative to scene normalization
  - `instance_position_l2c`: [x, y, z] in normalized space
  - `instance_quaternion_l2c`: [w, x, y, z] rotation quaternion
  - `instance_scale_l2c`: [s] or [sx, sy, sz] object scale
  - `scene_scale`: [sx, sy, sz] scene normalization scale
  - `scene_shift`: [x, y, z] scene center offset
- **Convention**: PyTorch3D (row-major matrices)
- **"l2c"**: "local to canonical" - object pose in scene-normalized space

## Scene Normalization

SAM3D normalizes scenes using scale-shift invariance (similar to MiDaS depth normalization):

```python
# From pose_target.py: ScaleShiftInvariant.get_scale_and_shift()
shift_z = pointmap[..., -1].nanmedian()  # Median depth
shift = [0, 0, shift_z]
scale = (pointmap - shift).abs().nanmean()  # Mean absolute distance
```

This transforms object poses as:
```python
position_normalized = (position_world - scene_shift) / scene_scale
scale_normalized = scale_world / scene_scale
```

## Required Conversion for FoundationPose → SAM3D

1. **Compute scene normalization parameters**:
   ```python
   # Get all object positions in the scene
   positions = [obj['translation'] for obj in scene_objects]

   # Compute scene center and scale
   positions = np.array(positions)
   scene_shift = positions.mean(axis=0)  # or median
   scene_scale = np.abs(positions - scene_shift).mean()
   ```

2. **Transform object poses to canonical space**:
   ```python
   # Normalize translation
   position_l2c = (position_world - scene_shift) / scene_scale

   # Normalize scale
   scale_l2c = scale_world / scene_scale

   # Rotation stays the same (just convert matrix → quaternion)
   quaternion_l2c = matrix_to_quaternion(rotation_matrix)
   ```

3. **Store normalization for decoding**:
   - `scene_scale`: Used to denormalize predictions back to metric
   - `scene_shift`: Used to recenter predictions

## Pose Target Conventions

SAM3D supports multiple pose representations (see `pose_target.py`):

- **`Identity`**: Direct passthrough (no normalization)
- **`ScaleShiftInvariant`**: MiDaS-style scale-shift normalization
- **`DisparitySpace`**: Disparity-based representation (x/z, y/z)
- **`NormalizedSceneScale`**: Normalize by scene scale
- **`ApparentSize`**: Scale relative to distance

For training with FoundationPose, we likely want:
- **`ScaleShiftInvariant`**: If using depth/pointmap conditioning
- **`Identity`**: If keeping metric poses
- **`NormalizedSceneScale`**: If normalizing by scene statistics

## Camera Coordinates vs. World Coordinates

**Important**: SAM3D may expect poses in **camera coordinates**, not world coordinates!

- FoundationPose: Objects in world, camera view transform available
- SAM3D: May expect object poses relative to camera

To convert:
```python
# World to camera transform
T_cam_world = camera_view_transform  # 4x4 matrix

# Object pose in camera coordinates
T_obj_cam = T_cam_world @ T_obj_world
```

Check if poses should be in camera frame or normalized world frame!

## Action Items

Before finalizing the dataset:

1. ✅ **Verify coordinate frame**: World or camera?
   - Check if FoundationPose poses are already in camera frame
   - Or if we need to apply camera transform

2. ✅ **Determine normalization scheme**:
   - Compute `scene_scale` and `scene_shift` from object positions
   - Or use depth/pointmap if available

3. ✅ **Choose pose target convention**:
   - Match the convention used in training
   - Default: `ScaleShiftInvariant` for depth-conditioned models

4. ✅ **Test with one sample**:
   - Load sample, convert poses
   - Visualize to verify correctness
   - Check scale magnitudes (should be ~ 1.0 after normalization)

## References

- PyTorch3D Transforms: https://pytorch3d.org/docs/transforms
- MiDaS (scale-shift invariance): https://arxiv.org/pdf/1907.01341v3
- SAM3D pose_target.py: `/workspace/sam-3d-objects/sam3d_objects/data/dataset/tdfy/pose_target.py`
