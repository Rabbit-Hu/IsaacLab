#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to visualize RGB and normals data from the camera in the normal_camera_example environment.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Visualize camera data from normal_camera_example environment")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from gbuffer_camera_example_env_cfg import GbufferCameraExampleEnvCfg

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.dict import print_dict


def main():
    """Main function."""
    # create environment configuration
    env_cfg = GbufferCameraExampleEnvCfg()
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
        # Get RGB, normals, and albedo data for all environments
        rgb_data = camera_data["rgb"].cpu().numpy()  # Shape: (num_envs, H, W, 3)
        normals_data = camera_data["normals"].cpu().numpy()  # Shape: (num_envs, H, W, 3)
        albedo_data = camera_data["gbuffer:albedo"].cpu().numpy()  # Shape: (num_envs, H, W, 3)
        instance_id_data = camera_data["instance_id_segmentation_fast"].cpu().numpy()  # Shape: (num_envs, H, W, 1)

        print(f"RGB data: shape={rgb_data.shape}, dtype={rgb_data.dtype}, min={rgb_data.min()}, max={rgb_data.max()}")
        print(
            f"Normals data: shape={normals_data.shape}, dtype={normals_data.dtype}, min={normals_data.min()},"
            f" max={normals_data.max()}"
        )
        print(
            f"Albedo data: shape={albedo_data.shape}, dtype={albedo_data.dtype}, min={albedo_data.min()},"
            f" max={albedo_data.max()}"
        )
        print(
            f"Instance ID data: shape={instance_id_data.shape}, dtype={instance_id_data.dtype},"
            f" min={instance_id_data.min()}, max={instance_id_data.max()}"
        )

        print(f"idToLabels: {camera.data.info.get('instance_id_segmentation_fast', [{}]).get('idToLabels', {})}")

        # Create visualization with grid layout: rows = num_envs, columns = 3 (RGB, Normals, Albedo)
        num_envs = rgb_data.shape[0]
        fig, axes = plt.subplots(num_envs, 4, figsize=(22, 5 * num_envs))
        fig.suptitle(f"Camera Data Visualization - {num_envs} Environments", fontsize=16)

        # Handle single environment case (axes won't be 2D)
        if num_envs == 1:
            axes = axes.reshape(1, -1)

        for env_idx in range(num_envs):
            # RGB image
            axes[env_idx, 0].imshow(rgb_data[env_idx])
            axes[env_idx, 0].set_title(f"Env {env_idx}: RGB Image")
            axes[env_idx, 0].axis("off")

            # Normals image (convert from [-1,1] to [0,1] for visualization)
            normals_vis = (normals_data[env_idx] + 1.0) / 2.0
            axes[env_idx, 1].imshow(normals_vis)
            axes[env_idx, 1].set_title(f"Env {env_idx}: Surface Normals")
            axes[env_idx, 1].axis("off")

            # Albedo image
            axes[env_idx, 2].imshow(albedo_data[env_idx])
            axes[env_idx, 2].set_title(f"Env {env_idx}: Albedo")
            axes[env_idx, 2].axis("off")

            # Instance ID image
            axes[env_idx, 3].imshow(instance_id_data[env_idx], cmap="jet")
            axes[env_idx, 3].set_title(f"Env {env_idx}: Instance ID Segmentation")
            axes[env_idx, 3].axis("off")

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
