#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to visualize RGB and normals data from the camera in the normal_camera_example environment.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Visualize camera data from normal_camera_example environment")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.dict import print_dict

from normal_camera_example_env_cfg import NormalCameraExampleEnvCfg


def main():
    """Main function."""
    # create environment configuration
    env_cfg = NormalCameraExampleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.decimation = 1  # Run at full resolution for visualization
    
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    print("Environment created successfully!")
    print(f"Number of environments: {env.num_envs}")
    print(f"Environment device: {env.device}")
    
    # reset environment to get initial state
    obs, _ = env.reset()
    print("Environment reset. Taking initial camera capture...")
    
    # step the simulation once to ensure camera data is captured
    actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    obs, _, _, _, _ = env.step(actions)
    
    # access the camera
    camera = env.scene["camera"]
    print(f"Camera device: {camera.device}")
    print(f"Camera data types: {camera.data.output.keys()}")
    
    # get the camera data
    camera_data = camera.data.output
    
    if "rgb" in camera_data and "normals" in camera_data:
        # Get RGB and normals data for the first environment
        rgb_data = camera_data["rgb"][0].cpu().numpy()  # Shape: (H, W, 3)
        normals_data = camera_data["normals"][0].cpu().numpy()  # Shape: (H, W, 3)
        
        print(f"RGB data shape: {rgb_data.shape}")
        print(f"Normals data shape: {normals_data.shape}")
        print(f"RGB data range: [{rgb_data.min():.3f}, {rgb_data.max():.3f}]")
        print(f"Normals data range: [{normals_data.min():.3f}, {normals_data.max():.3f}]")
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Camera Data Visualization - Initial Frame")
        
        # RGB image
        axes[0].imshow(rgb_data)
        axes[0].set_title("RGB Image")
        axes[0].axis("off")
        
        # Normals image (convert from [-1,1] to [0,1] for visualization)
        normals_vis = (normals_data + 1.0) / 2.0
        axes[1].imshow(normals_vis)
        axes[1].set_title("Surface Normals (RGB Encoded)")
        axes[1].axis("off")
        
        plt.tight_layout()
        plt.savefig("camera_visualization.png", dpi=150, bbox_inches="tight")
        print("Visualization saved as 'camera_visualization.png'")
        
        if not args_cli.headless:
            plt.show()
        
    else:
        print("Error: Camera data does not contain both 'rgb' and 'normals'")
        print(f"Available data types: {list(camera_data.keys())}")
    
    # close the environment
    env.close()


if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()
