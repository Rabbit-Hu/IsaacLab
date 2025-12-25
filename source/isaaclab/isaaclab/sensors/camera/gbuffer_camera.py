# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
from collections.abc import Sequence as SequenceABC
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from collections.abc import Sequence

import warp as wp

from isaaclab.utils.warp.kernels import reshape_tiled_image

from . import TiledCamera

if TYPE_CHECKING:
    from .gbuffer_camera_cfg import GBufferCameraCfg


class GBufferCamera(TiledCamera):
    """Custom camera sensor that extends TiledCamera to support G-buffer data extraction."""

    cfg: "GBufferCameraCfg"

    def __init__(self, cfg: "GBufferCameraCfg"):
        """Initialize the G-buffer camera.

        Args:
            cfg: Configuration for the G-buffer camera.
        """
        # Initialize parent class
        super().__init__(cfg)

    def __del__(self):
        """Destructor to clean up G-buffer annotators."""
        for annotator in self._gbuffer_annotators.values():
            annotator.detach(self.render_product_paths)

        super().__del__()

    def _initialize_impl(self):
        """Initialize the camera with G-buffer support."""
        # Initialize parent class
        super()._initialize_impl()

        import omni.replicator.core as rep

        print("Initializing GBufferCamera...")

        # Additional G-buffer annonators (in addition to tiled camera annotators)
        self._gbuffer_annotators = dict()
        for data_type in self.cfg.gbuffer_data_types:
            if data_type == "albedo":
                annotator = rep.AnnotatorRegistry.get_annotator(
                    "DiffuseAlbedo", device=self.device, do_array_copy=False
                )
                self._gbuffer_annotators[data_type] = annotator
            else:
                raise ValueError(f"Unsupported G-buffer data type: {data_type}")

        print(f"G-buffer annotators: {list(self._gbuffer_annotators.keys())}")

        for annotator in self._gbuffer_annotators.values():
            annotator.attach(self._render_product_paths)

        print("G-buffer annotators attached.")

        # Add G-buffer data types to self.data.output
        data_dict = self._data.output
        for data_type in self.cfg.gbuffer_data_types:
            if data_type == "albedo":
                data_dict["gbuffer:" + data_type] = torch.zeros(
                    (self._view.count, self.cfg.height, self.cfg.width, 3),
                    device=self.device,
                    dtype=torch.uint8,
                )
        self._data.output = data_dict

        print("GBufferCamera initialized successfully.")

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        super()._update_buffers_impl(env_ids)

        for data_type, annotator in self._gbuffer_annotators.items():
            output = annotator.get_data()
            if isinstance(output, dict):
                tiled_data_buffer = output["data"]
                self._data.info[data_type] = output["info"]
            else:
                tiled_data_buffer = output

            if isinstance(tiled_data_buffer, np.ndarray):
                tiled_data_buffer = wp.array(tiled_data_buffer, device=self.device)
            else:
                tiled_data_buffer = tiled_data_buffer.to(device=self.device)

            if data_type == "albedo":
                tiled_data_buffer = tiled_data_buffer[:, :, :3].contiguous()

            wp.launch(
                kernel=reshape_tiled_image,
                dim=(self._view.count, self.cfg.height, self.cfg.width),
                inputs=[
                    tiled_data_buffer.flatten(),
                    wp.from_torch(self._data.output["gbuffer:" + data_type]),  # Fixed: use correct key
                    *list(self._data.output["gbuffer:" + data_type].shape[1:]),  # Fixed: use correct key
                    self._tiling_grid_shape()[0],  # num_tiles_x
                ],
                device=self.device,
            )
