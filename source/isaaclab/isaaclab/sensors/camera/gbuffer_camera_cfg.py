"""
Custom G-Buffer Camera implementation for Isaac Lab.

This module provides a custom camera sensor that can extract G-buffer data
including material properties like albedo, roughness, and metallic values.
"""

from typing import List

from . import TiledCameraCfg
from ...utils import configclass

from .gbuffer_camera import GBufferCamera


@configclass
class GBufferCameraCfg(TiledCameraCfg):
    """Configuration for a G-buffer camera sensor."""
    
    class_type: type = GBufferCamera
    
    # Additional G-buffer data types (currently only "albedo" is supported)
    gbuffer_data_types: List[str] = ["albedo"]
    """List of G-buffer data types to capture. Available options:
    - "albedo": Diffuse albedo of materials
    """
