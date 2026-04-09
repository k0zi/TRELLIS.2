import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from . import _C


def intrinsics_to_projection(
        intrinsics: torch.Tensor,
        near: float,
        far: float,
    ) -> torch.Tensor:
    """OpenCV intrinsics → OpenGL perspective matrix."""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = - 2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.
    return ret


def _rasterize_voxels_cpu(
    position: torch.Tensor,   # [N, 3]
    attrs: torch.Tensor,       # [N, C]
    voxel_size: float,
    view: torch.Tensor,        # [4, 4] col-major (transposed from Python)
    proj_view: torch.Tensor,   # [4, 4] col-major (transposed from Python)
    camera: torch.Tensor,      # [3]
    tan_fovx: float,
    tan_fovy: float,
    W: int,
    H: int,
) -> tuple:
    """
    Simple CPU voxel splatting renderer.

    Projects each voxel centre to screen space and draws a square
    footprint proportional to the projected voxel size.  No GPU needed.
    This is intentionally simple and serves as a functional fallback;
    quality is lower than the CUDA kernel.
    """
    N, C = attrs.shape
    device = position.device

    # proj_view was passed transposed (column-major), undo for row-vector math
    PV = proj_view.T.float()  # [4, 4] row-major

    # Homogeneous position
    ones = torch.ones(N, 1, device=device, dtype=torch.float32)
    pos_h = torch.cat([position.float(), ones], dim=-1)   # [N, 4]

    # Clip-space coordinates
    clip = pos_h @ PV.T                                   # [N, 4]

    w = clip[:, 3].clamp(min=1e-6)
    ndc_x = clip[:, 0] / w
    ndc_y = clip[:, 1] / w
    ndc_z = clip[:, 2] / w

    # Frustum culling
    vis = (ndc_x.abs() <= 1.0) & (ndc_y.abs() <= 1.0) & (ndc_z > 0) & (ndc_z < 1)
    if not vis.any():
        return (
            torch.zeros(C, H, W, device=device),
            torch.zeros(H, W, device=device),
            torch.zeros(H, W, device=device),
        )

    ndc_x = ndc_x[vis]
    ndc_y = ndc_y[vis]
    ndc_z = ndc_z[vis]
    w_vis  = w[vis]
    attrs_v = attrs[vis].float()

    # Screen-space pixel coordinates
    px = ((ndc_x + 1.0) * 0.5 * W).long().clamp(0, W - 1)
    py = ((1.0 - (ndc_y + 1.0) * 0.5) * H).long().clamp(0, H - 1)

    # Estimate projected voxel radius in pixels
    # (voxel_size / depth) * focal → pixels
    focal_px = 0.5 * W / tan_fovx
    depth = w_vis.clamp(min=1e-3)
    radius = ((voxel_size / depth) * focal_px * 0.5).long().clamp(min=0, max=16)

    # Sort back-to-front for painter's algorithm
    order = ndc_z.argsort(descending=True)
    px, py, radius, attrs_v, ndc_z = (
        px[order], py[order], radius[order], attrs_v[order], ndc_z[order]
    )

    color = torch.zeros(H, W, C, device=device)
    depth_buf = torch.zeros(H, W, device=device)
    alpha_buf = torch.zeros(H, W, device=device)

    for i in range(px.shape[0]):
        r = radius[i].item()
        cx_, cy_ = px[i].item(), py[i].item()
        d = ndc_z[i].item()
        col = attrs_v[i]

        x0, x1 = max(0, cx_ - r), min(W, cx_ + r + 1)
        y0, y1 = max(0, cy_ - r), min(H, cy_ + r + 1)

        color[y0:y1, x0:x1] = col
        depth_buf[y0:y1, x0:x1] = d
        alpha_buf[y0:y1, x0:x1] = 1.0

    return (
        color.permute(2, 0, 1),  # [C, H, W]
        depth_buf,
        alpha_buf,
    )


class VoxelRenderer:
    """
    Renderer for the Voxel representation.

    Automatically selects the CUDA kernel when available, falling back
    to a CPU software renderer on systems without CUDA (e.g. AMD iGPU).
    """

    def __init__(self, rendering_options={}) -> None:
        self.rendering_options = edict({
            "resolution": None,
            "near": 0.1,
            "far": 10.0,
            "ssaa": 1,
        })
        self.rendering_options.update(rendering_options)

    def render(
            self,
            position: torch.Tensor,
            attrs: torch.Tensor,
            voxel_size: float,
            extrinsics: torch.Tensor,
            intrinsics: torch.Tensor,
        ) -> edict:
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]

        view = extrinsics
        perspective = intrinsics_to_projection(intrinsics, near, far)
        camera = torch.inverse(view)[:3, 3]
        focalx = intrinsics[0, 0]
        focaly = intrinsics[1, 1]

        args = (
            position,
            attrs,
            voxel_size,
            view.T.contiguous(),
            (perspective @ view).T.contiguous(),
            camera,
            0.5 / focalx,
            0.5 / focaly,
            resolution * ssaa,
            resolution * ssaa,
        )

        use_cuda = (
            position.device.type == 'cuda'
            and torch.cuda.is_available()
        )
        if use_cuda:
            color, depth, alpha = _C.rasterize_voxels_cuda(*args)
        else:
            color, depth, alpha = _rasterize_voxels_cpu(*args)

        if ssaa > 1:
            color = F.interpolate(color[None], size=(resolution, resolution),
                                  mode='bilinear', align_corners=False,
                                  antialias=True).squeeze()
            depth = F.interpolate(depth[None, None], size=(resolution, resolution),
                                  mode='bilinear', align_corners=False,
                                  antialias=True).squeeze()
            alpha = F.interpolate(alpha[None, None], size=(resolution, resolution),
                                  mode='bilinear', align_corners=False,
                                  antialias=True).squeeze()

        return edict({'attr': color, 'depth': depth, 'alpha': alpha})
