#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

from gsplat.rasterize import RasterizeGaussians
from gsplat.project_gaussians import ProjectGaussians
from gsplat.sh import SphericalHarmonics, num_sh_bases


def render_gsplat(camera,means,opacities,colors_all,scales,quats,bg_color):

    opacities_crop = opacities
    means_crop = means
    colors_crop = colors_all
    scales_crop = scales
    quats_crop = quats
    
    viewmat, projmat =camera.world_view_transform,camera.full_proj_transform 
    H,W = int(camera.image_height), int(camera.image_width)
    cx, cy = W /2 , H / 2

    # cx,cy = camera.camera_center[0], camera.camera_center[1]
    # campos=camera.camera_center,
    prefiltered=False,

    tanfovx = math.tan(camera.FoVx * 0.5)
    tanfovy = math.tan(camera.FoVy * 0.5)
    
    # focal_y = H / (2.0 * tanfovy)
    focal_y = H / (2 *tanfovy)
    focal_x = W / (2 *tanfovx)
	# focal_x = W / (2.0 * tanfovx)

    BLOCK_X, BLOCK_Y = 16, 16
    tile_bounds = (
        (W + BLOCK_X - 1) // BLOCK_X,
        (H + BLOCK_Y - 1) // BLOCK_Y,
        1,
    )

    xys, depths, radii, conics, num_tiles_hit, cov3d = ProjectGaussians.apply(
        means_crop,
        torch.exp(scales_crop),
        1,
        quats_crop / quats_crop.norm(dim=-1, keepdim=True),
        viewmat.squeeze()[:3, :],
        projmat.squeeze() @ viewmat.squeeze(),
        camera.FoVx,
        camera.FoVy,
        # focal_x,
        # focal_y,
        cx,
        cy,
        H,
        W,
        tile_bounds,
    )

    # switched off sh degree scheduling 
    sh_degree_interval: int = 1000
    sh_degree: int = 3


    # Important to allow xys grads to populate properly
    # ns-gs code
    # xys.retain_grad()
    viewdirs = means_crop.detach() 
    # - camera.camera_to_worlds.detach()[..., :3, 3]  # (N, 3)
    viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)

    n = sh_degree
    rgbs = SphericalHarmonics.apply(n, viewdirs, colors_crop)
    rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)

    rgb = RasterizeGaussians.apply(
        xys,
        depths,
        radii,
        conics,
        num_tiles_hit,
        rgbs,
        torch.sigmoid(opacities_crop),
        H,
        W,
        bg_color,
    )

    return rgb, radii, xys


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # raster_settings = GaussianRasterizationSettings(
    #     image_height=int(viewpoint_camera.image_height),
    #     image_width=int(viewpoint_camera.image_width),
    #     tanfovx=tanfovx,
    #     tanfovy=tanfovy,
    #     bg=bg_color,
    #     scale_modifier=scaling_modifier,
    #     viewmatrix=viewpoint_camera.world_view_transform,
    #     projmatrix=viewpoint_camera.full_proj_transform,
    #     sh_degree=pc.active_sh_degree,
    #     campos=viewpoint_camera.camera_center,
    #     prefiltered=False,
    #     debug=pipe.debug
    # )

    # rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    # means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    colors_precomp = None
    shs = pc.get_features
    
    rendered_image, radii, screenspace_points, = render_gsplat(
        viewpoint_camera,
        means3D,
        opacities=opacity,
        colors_all = shs, # check this
        scales = scales, 
        quats = rotations,
        bg_color = bg_color)


    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # rendered_image, radii = rasterizer(
    #     means3D = means3D,
    #     means2D = means2D,
    #     shs = shs,
    #     colors_precomp = colors_precomp,
    #     opacities = opacity,
    #     scales = scales,
    #     rotations = rotations,
    #     cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image.permute(2,0,1),
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

    return 
