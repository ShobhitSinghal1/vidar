# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from functools import lru_cache

import torch
import torch.nn as nn

from vidar.arch.networks.layers.fsm.camera_utils import scale_intrinsics, invert_intrinsics
from vidar.arch.networks.layers.fsm.pose import Pose
from vidar.utils.tensor import pixel_grid
from vidar.utils.types import is_tensor, is_list


class Camera(nn.Module):
    """
    Differentiable camera class implementing reconstruction and projection
    functions for a pinhole model.
    """
    def __init__(self, I, Tcw=None,):
        """
        Initializes the Camera class

        Parameters
        ----------
        I : torch.Tensor
            Camera intrinsics tensor [4]
        Tcw : Pose or torch.Tensor
            Camera -> World pose transformation [B,4,4]
        """
        super().__init__()
        self.I = I
        if Tcw is None:
            self.Tcw = Pose.identity(len(I))
        elif isinstance(Tcw, Pose):
            self.Tcw = Tcw
        else:
            self.Tcw = Pose(Tcw)

        self.Tcw.to(self.I.device)

    def __len__(self):
        """Batch size of the camera intrinsics"""
        return len(self.I)

    def to(self, *args, **kwargs):
        """Moves object to a specific device"""
        self.I = self.I.to(*args, **kwargs)
        self.Tcw = self.Tcw.to(*args, **kwargs)
        return self

    @property
    def fx(self):
        """Focal length in x"""
        return self.K[:, 0, 0]

    @property
    def fy(self):
        """Focal length in y"""
        return self.K[:, 1, 1]

    @property
    def cx(self):
        """Principal point in x"""
        return self.K[:, 0, 2]

    @property
    def cy(self):
        """Principal point in y"""
        return self.K[:, 1, 2]

    @property
    @lru_cache()
    def Twc(self):
        """World -> Camera pose transformation (inverse of Tcw)"""
        return self.Tcw.inverse()

    def reconstruct(self, depth, frame='w', scene_flow=None, return_grid=False):
        """
        Reconstructs pixel-wise 3D points from a depth map.

        Parameters
        ----------
        depth : torch.Tensor
            Depth map for the camera [B,1,H,W]
        frame : 'w'
            Reference frame: 'c' for camera and 'w' for world
        scene_flow : torch.Tensor
            Optional per-point scene flow to be added (camera reference frame) [B,3,H,W]
        return_grid : bool
            Return pixel grid as well

        Returns
        -------
        points : torch.tensor
            Pixel-wise 3D points [B,3,H,W]
        """
        # If depth is a list, return each reconstruction
        if is_list(depth):
            return [self.reconstruct(d, frame, scene_flow, return_grid) for d in depth]
        # Dimension assertions
        assert depth.dim() == 4 and depth.shape[1] == 1, \
            'Wrong dimensions for camera reconstruction'

        # Create flat index grid [B,3,H,W]
        B, _, H, W = depth.shape
        grid = pixel_grid((H, W), B, device=depth.device, normalize=False, with_ones=True)
        flat_grid = grid.view(B, 3, -1)

        # Get inverse intrinsics
        Kinv = self.Kinv if self.hw is None else self.scaled_Kinv(depth.shape)

        # Estimate the outward rays in the camera frame
        Xnorm = (Kinv.bmm(flat_grid)).view(B, 3, H, W)
        # Scale rays to metric depth
        Xc = Xnorm * depth

        # Add scene flow if provided
        if scene_flow is not None:
            Xc = Xc + scene_flow

        # If in camera frame of reference
        if frame == 'c':
            pass
        # If in world frame of reference
        elif frame == 'w':
            Xc = self.Twc @ Xc
        # If none of the above
        else:
            raise ValueError('Unknown reference frame {}'.format(frame))
        # Return points and grid if requested
        return (Xc, grid) if return_grid else Xc

    def project(self, X, frame='w', normalize=True, return_z=False):
        """
        Projects 3D points onto the image plane

        Parameters
        ----------
        X : torch.Tensor
            3D points to be projected [B,3,H,W]
        frame : 'w'
            Reference frame: 'c' for camera and 'w' for world
        normalize : bool
            Normalize grid coordinates
        return_z : bool
            Return the projected z coordinate as well

        Returns
        -------
        points : torch.Tensor
            2D projected points that are within the image boundaries [B,H,W,2]
        """
        assert 2 < X.dim() <= 4 and X.shape[1] == 3, \
            'Wrong dimensions for camera projection'

        # Determine if input is a grid
        is_grid = X.dim() == 4
        # If it's a grid, flatten it
        X_flat = X.view(X.shape[0], 3, -1) if is_grid else X

        # Get dimensions
        hw = X.shape[2:] if is_grid else self.hw
        # Get intrinsics
        K = self.scaled_K(X.shape) if is_grid else self.K

        # Project 3D points onto the camera image plane
        if frame == 'c':
            Xc = K.bmm(X_flat)
        elif frame == 'w':
            Xc = K.bmm(self.Tcw @ X_flat)
        else:
            raise ValueError('Unknown reference frame {}'.format(frame))

        # Extract coordinates
        Z = Xc[:, 2].clamp(min=1e-5)
        XZ = Xc[:, 0] / Z
        YZ = Xc[:, 1] / Z

        # Normalize points
        if normalize and hw is not None:
            XZ = 2 * XZ / (hw[1] - 1) - 1.
            YZ = 2 * YZ / (hw[0] - 1) - 1.

        # Clamp out-of-bounds pixels
        Xmask = ((XZ > 1) + (XZ < -1)).detach()
        XZ[Xmask] = 2.
        Ymask = ((XZ > 1) + (YZ < -1)).detach()
        YZ[Ymask] = 2.

        # Stack X and Y coordinates
        XY = torch.stack([XZ, YZ], dim=-1)
        # Reshape coordinates to a grid if possible
        if is_grid and hw is not None:
            XY = XY.view(X.shape[0], hw[0], hw[1], 2)

        # If also returning depth
        if return_z:
            # Reshape depth values to a grid if possible
            if is_grid and hw is not None:
                Z = Z.view(X.shape[0], hw[0], hw[1], 1).permute(0, 3, 1, 2)
            # Otherwise, reshape to an array
            else:
                Z = Z.view(X.shape[0], -1, 1).permute(0, 2, 1)
            # Return coordinates and depth values
            return XY, Z
        else:
            # Return coordinates
            return XY

    def reconstruct_depth_map(self, depth, to_world=True):
        if to_world:
            return self.reconstruct(depth, frame='w')
        else:
            return self.reconstruct(depth, frame='c')

    def project_points(self, points, from_world=True, normalize=True, return_z=False):
        if from_world:
            return self.project(points, frame='w')
        else:
            return self.project(points, frame='c')

    def coords_from_depth(self, depth, ref_cam=None):
        if ref_cam is None:
            return self.project_points(self.reconstruct_depth_map(depth, to_world=False), from_world=True)
        else:
            return ref_cam.project_points(self.reconstruct_depth_map(depth, to_world=True), from_world=True)
