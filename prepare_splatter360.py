import os
import json
import torch
import cv2
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from einops import rearrange, repeat

# Paths
ROOT_DIR = Path("/data/Splatter360")
DATA_DIR = Path("/data/extracted")
OUTPUT_DIR = ROOT_DIR / "datasets" / "real_data"
CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "depth_anything_v2_vits.pth"
DEPTH_ANYTHING_REPO = Path.home() / "Depth-Anything-V2"

# Setup Depth Anything
sys.path.append(str(DEPTH_ANYTHING_REPO))
from depth_anything_v2.dpt import DepthAnythingV2

# Setup Splatter360 utils (for Equirec2Cube)
sys.path.append(str(ROOT_DIR / "src"))
from geometry.util import Equirec2Cube

def get_rotation_matrix(quaternion, translation):
    # quaternion: [w, x, y, z] or [x, y, z, w]
    # Check normalization
    q = np.array(quaternion)
    if np.abs(np.linalg.norm(q) - 1.0) > 1e-3:
        print(f"Warning: Quaternion not normalized: {q}")
        q = q / np.linalg.norm(q)
    
    # Try to determine order. Usually w is first if not specified.
    # For scipy: [x, y, z, w]
    # Let's assume input is [w, x, y, z] -> convert to [x, y, z, w]
    q_scipy = np.array([q[1], q[2], q[3], q[0]])
    
    rot = R.from_quat(q_scipy).as_matrix()
    
    c2w = np.eye(4)
    c2w[:3, :3] = rot
    c2w[:3, 3] = translation
    return c2w

def get_cube_poses(c2w_pano):
    # c2w_pano: 4x4 matrix (camera to world) for the panoramic camera
    # We need to generate 6 poses for the cubemap faces: F, R, B, L, U, D
    
    pose_rotations = [
        R.from_euler('y', 0, degrees=True).as_matrix(),   # F (Look +Z)
        R.from_euler('y', -90, degrees=True).as_matrix(), # R (Look +X) -> Wait, if I rotate -90 around Y, I look Right.
        
        R.from_euler('y', 180, degrees=True).as_matrix(), # B (Look -Z)
        R.from_euler('y', 90, degrees=True).as_matrix(),  # L (Look -X)
        R.from_euler('x', -90, degrees=True).as_matrix(), # U (Look +Y)
        R.from_euler('x', 90, degrees=True).as_matrix(),  # D (Look -Y)
    ]
    
    c2w_cubes = []
    for rot in pose_rotations:
        c2w = np.eye(4)
        c2w[:3, :3] = c2w_pano[:3, :3] @ rot
        c2w[:3, 3] = c2w_pano[:3, 3]
        c2w_cubes.append(c2w)
        
    return np.stack(c2w_cubes) # 6x4x4

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load Depth Anything
    depth_model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    depth_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
    depth_model = depth_model.to(device).eval()
    
    # Setup Output Directories
    scene_name = "real_scene"
    test_dir = OUTPUT_DIR / "test"
    scene_dir = test_dir / scene_name
    pano_dir = scene_dir / "pano"
    pano_depth_dir = scene_dir / "pano_depth"
    cubemaps_depth_dir = scene_dir / "cubemaps_depth"
    
    pano_dir.mkdir(parents=True, exist_ok=True)
    pano_depth_dir.mkdir(parents=True, exist_ok=True)
    cubemaps_depth_dir.mkdir(parents=True, exist_ok=True)
    
    # Read Poses
    with open(DATA_DIR / "poses.json", "r") as f:
        poses_data = json.load(f)
        
    image_files = sorted(list((DATA_DIR / "images").glob("*.jpg")) + list((DATA_DIR / "images").glob("*.png")))
    
    chunk_data = []
    
    cube_h, cube_w = 256, 256
    
    equi2cube = Equirec2Cube(512, 1024, 256)
    
    # Store all frames for the scene
    all_cameras = []
    all_c2ws_cubes = []
    all_fxfycxcys = []
    
    for idx, img_path in enumerate(tqdm(image_files)):
        img_name = img_path.name
        if img_name not in poses_data:
            print(f"Skipping {img_name} - no pose found")
            continue
            
        pose_info = poses_data[img_name]
        c2w = get_rotation_matrix(pose_info["rotation_quaternion"], pose_info["translation"])
        
        # 1. Copy RGB
        dest_rgb = pano_dir / f"{idx:05d}.png"
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (1024, 512)) # Ensure correct size
        cv2.imwrite(str(dest_rgb), img)
        
        # 2. Estimate Depth
        depth = depth_model.infer_image(img) # Returns numpy array
        
        # Save Pano Depth (as 16-bit png, scaled by 1000)
        d_min, d_max = depth.min(), depth.max()
        depth_norm = (depth - d_min) / (d_max - d_min + 1e-6)
        depth_metric = depth_norm * 9.9 + 0.1 # Range [0.1, 10.0]
        
        depth_uint16 = (depth_metric * 1000).astype(np.uint16)
        cv2.imwrite(str(pano_depth_dir / f"{idx:05d}.png"), depth_uint16)
        
        # 3. Convert to Cubemap Depth
        # Equirec2Cube expects [H, W, 1] for depth
        depth_metric_expanded = depth_metric[..., np.newaxis]
        _, cube_dep = equi2cube.run(img, depth_metric_expanded)
        
        cube_dep_flat = cube_dep # [256, 256*6, 1]
        # Split into 6 faces
        faces = np.split(cube_dep_flat, 6, axis=1) # List of [256, 256, 1]
        cube_dep_tensor = torch.from_numpy(np.stack(faces, axis=0)) # [6, 256, 256, 1]
        
        # Save as .torch
        torch.save(cube_dep_tensor, cubemaps_depth_dir / f"{idx:05d}.torch")
        
        # 4. Prepare Metadata
        c2ws_cubes = get_cube_poses(c2w)
        
        # Intrinsics
        # fx = fy = W/2 = 128 (for 90 deg FOV, tan(45)=1. f = (W/2)/tan(45) = 128)
        # cx = cy = 128.
        # fxfycxcys = torch.tensor([128.0, 128.0, 128.0, 128.0]).unsqueeze(0).repeat(6, 1) # INCORRECT for dataset_hm3d.py
        fxfycxcys = torch.tensor([128.0, 128.0, 128.0, 128.0]) # [4] Correct!
        
        all_cameras.append(torch.from_numpy(c2w).float())
        all_c2ws_cubes.append(torch.from_numpy(c2ws_cubes).float())
        all_fxfycxcys.append(fxfycxcys.float())

    example_data = {
        "key": scene_name,
        "cube_shape": torch.tensor([256, 256], dtype=torch.int32),
        "cameras": torch.stack(all_cameras), # [N, 4, 4]
        "c2ws_cubes": torch.stack(all_c2ws_cubes), # [N, 6, 4, 4]
        "fxfycxcys": torch.stack(all_fxfycxcys) # [N, 4]
    }
    chunk_data.append(example_data)

    # Save Chunk
    torch.save(chunk_data, test_dir / "000000.torch")
    
    # Save Index
    index = {scene_name: "000000.torch"}
    with open(test_dir / "index.json", "w") as f:
        json.dump(index, f)
        
    print("Preprocessing complete!")

if __name__ == "__main__":
    main()