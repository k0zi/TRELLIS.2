from typing import *
from tqdm import tqdm
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import trimesh
import trimesh.visual
import trimesh.proximity
import trimesh.triangles


# ---------------------------------------------------------------------------
# Embedded CPU grid-sample helper (avoids cross-package imports)
# ---------------------------------------------------------------------------

def _grid_sample_3d_cpu(
    attrs: torch.Tensor,
    coords: torch.Tensor,
    shape,
    grid: torch.Tensor,
    mode: str = 'trilinear',
) -> torch.Tensor:
    """
    CPU trilinear sampling from a sparse 3D voxel volume.

    Args:
        attrs:  [N, C]  sparse feature values
        coords: [N, 4]  integer indices (batch, z, y, x)
        shape:  (D, H, W) or (B, C, D, H, W)
        grid:   [1, K, 3] query positions in voxel space, (x, y, z) order
    Returns:
        [1, K, C] sampled features
    """
    device = attrs.device
    C = attrs.shape[-1]

    if len(shape) == 3:
        D, H, W = shape
    elif len(shape) == 5:
        _, _, D, H, W = shape
    else:
        raise ValueError(f"Unsupported shape: {shape}")

    dense = torch.zeros(1, C, D, H, W, dtype=torch.float32, device=device)
    z_idx = coords[:, 1].long().clamp(0, D - 1)
    y_idx = coords[:, 2].long().clamp(0, H - 1)
    x_idx = coords[:, 3].long().clamp(0, W - 1)
    dense[0, :, z_idx, y_idx, x_idx] = attrs.float().T

    q = grid.float()
    gx = q[..., 0] / max(W - 1, 1) * 2 - 1
    gy = q[..., 1] / max(H - 1, 1) * 2 - 1
    gz = q[..., 2] / max(D - 1, 1) * 2 - 1
    norm_grid = torch.stack([gx, gy, gz], dim=-1).unsqueeze(1).unsqueeze(1)

    sampled = F.grid_sample(
        dense, norm_grid,
        mode='bilinear', padding_mode='border', align_corners=True,
    )
    return sampled.squeeze(2).squeeze(2).permute(0, 2, 1)  # [1, K, C]


# ---------------------------------------------------------------------------
# Mesh-processing helpers (CPU, no cumesh dependency)
# ---------------------------------------------------------------------------

def _fill_holes(verts: np.ndarray, faces: np.ndarray):
    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    trimesh.repair.fill_holes(tm)
    return np.asarray(tm.vertices, np.float32), np.asarray(tm.faces, np.int32)


def _simplify(verts: np.ndarray, faces: np.ndarray, target: int):
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(
        vertex_matrix=verts.astype(np.float64),
        face_matrix=faces.astype(np.int32),
    ))
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(target))
    m = ms.current_mesh()
    return m.vertex_matrix().astype(np.float32), m.face_matrix().astype(np.int32)


def _clean(verts: np.ndarray, faces: np.ndarray, min_component_faces: int = 50):
    """Remove duplicate faces, small components, fix orientations."""
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(
        vertex_matrix=verts.astype(np.float64),
        face_matrix=faces.astype(np.int32),
    ))
    try:
        ms.meshing_remove_duplicate_faces()
    except Exception:
        pass
    try:
        ms.meshing_repair_non_manifold_edges()
    except Exception:
        pass
    try:
        ms.meshing_remove_connected_component_by_face_number(
            mincomponentsize=min_component_faces)
    except Exception:
        pass
    try:
        ms.meshing_re_orient_faces_coherentely()
    except Exception:
        pass
    m = ms.current_mesh()
    return m.vertex_matrix().astype(np.float32), m.face_matrix().astype(np.int32)


def _uv_unwrap(verts: np.ndarray, faces: np.ndarray):
    """
    UV parametrize using xatlas.
    Returns (new_verts, new_faces, uvs, vmapping) where vmapping[i] is the
    original vertex index for new vertex i – same semantics as cumesh.
    """
    import xatlas
    vmapping, indices, uvs = xatlas.parametrize(
        verts.astype(np.float32), faces.astype(np.int32)
    )
    return (
        verts[vmapping].astype(np.float32),
        indices.astype(np.int32),
        uvs.astype(np.float32),
        vmapping,
    )


def _closest_point_barycentric(
    mesh_verts: np.ndarray,
    mesh_faces: np.ndarray,
    query_points: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    For each query point find the closest point on the mesh and return
    (closest_pos, face_id, barycentric_uvw).

    Replaces cumesh.cuBVH.unsigned_distance(..., return_uvw=True).
    """
    device = query_points.device
    tm = trimesh.Trimesh(vertices=mesh_verts, faces=mesh_faces, process=False)
    pts_np = query_points.cpu().float().numpy()

    closest_np, _, face_ids_np = trimesh.proximity.closest_point(tm, pts_np)

    face_verts = mesh_verts[mesh_faces[face_ids_np]]          # [K, 3, 3]
    uvw_np = trimesh.triangles.points_to_barycentric(face_verts, closest_np)  # [K, 3]

    closest_t  = torch.from_numpy(closest_np.astype(np.float32)).to(device)
    face_ids_t = torch.from_numpy(face_ids_np.astype(np.int64)).to(device)
    uvw_t      = torch.from_numpy(uvw_np.astype(np.float32)).to(device)
    return closest_t, face_ids_t, uvw_t


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def to_glb(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    decimation_target: int = 1_000_000,
    texture_size: int = 2048,
    remesh: bool = False,
    remesh_band: float = 1,
    remesh_project: float = 0.9,
    mesh_cluster_threshold_cone_half_angle_rad=np.radians(90.0),
    mesh_cluster_refine_iterations=0,
    mesh_cluster_global_iterations=1,
    mesh_cluster_smooth_strength=1,
    verbose: bool = False,
    use_tqdm: bool = False,
):
    """
    Convert an extracted mesh to a textured GLB file.

    Performs mesh cleaning, (optional) remeshing, UV unwrapping, and
    texture baking from a sparse attribute volume.

    This implementation works on CPU + AMD iGPU (no cumesh / CUDA required).
    It uses trimesh, pymeshlab, xatlas, and nvdiffrast (OpenGL context).
    """
    import nvdiffrast.torch as dr

    # ── Normalise AABB / voxel-size / grid-size ──────────────────────────
    if isinstance(aabb, (list, tuple)):
        aabb = np.array(aabb)
    if isinstance(aabb, np.ndarray):
        aabb = torch.tensor(aabb, dtype=torch.float32, device=coords.device)
    assert isinstance(aabb, torch.Tensor) and aabb.shape == (2, 3)

    if voxel_size is not None:
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size] * 3
        if isinstance(voxel_size, (list, tuple)):
            voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray):
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=coords.device)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        assert grid_size is not None
        if isinstance(grid_size, int):
            grid_size = [grid_size] * 3
        if isinstance(grid_size, (list, tuple)):
            grid_size = np.array(grid_size)
        if isinstance(grid_size, np.ndarray):
            grid_size = torch.tensor(grid_size, dtype=torch.int32, device=coords.device)
        voxel_size = (aabb[1] - aabb[0]) / grid_size

    # ── Setup ─────────────────────────────────────────────────────────────
    device = coords.device
    if use_tqdm:
        pbar = tqdm(total=6, desc="Extracting GLB")
    if verbose:
        print(f"Original mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    verts_np = vertices.cpu().float().numpy()
    faces_np = faces.cpu().numpy().astype(np.int32)

    # ── Step 1: Fill holes ────────────────────────────────────────────────
    verts_np, faces_np = _fill_holes(verts_np, faces_np)
    if verbose:
        print(f"After filling holes: {len(verts_np)} vertices, {len(faces_np)} faces")
    if use_tqdm:
        pbar.update(1)

    # Keep a copy of the hole-filled mesh for BVH queries later
    bvh_verts_np = verts_np.copy()
    bvh_faces_np = faces_np.copy()
    if use_tqdm:
        pbar.update(1)  # "Building BVH" step

    # ── Step 2: Simplify & clean ──────────────────────────────────────────
    if use_tqdm:
        pbar.set_description("Cleaning mesh")

    if not remesh:
        verts_np, faces_np = _simplify(verts_np, faces_np, decimation_target * 3)
        if verbose:
            print(f"After initial simplification: {len(verts_np)} verts, {len(faces_np)} faces")
        verts_np, faces_np = _clean(verts_np, faces_np)
        verts_np, faces_np = _fill_holes(verts_np, faces_np)
        if verbose:
            print(f"After initial cleanup: {len(verts_np)} verts, {len(faces_np)} faces")
        verts_np, faces_np = _simplify(verts_np, faces_np, decimation_target)
        if verbose:
            print(f"After final simplification: {len(verts_np)} verts, {len(faces_np)} faces")
        verts_np, faces_np = _clean(verts_np, faces_np)
        verts_np, faces_np = _fill_holes(verts_np, faces_np)
        if verbose:
            print(f"After final cleanup: {len(verts_np)} verts, {len(faces_np)} faces")
    else:
        warnings.warn(
            "remesh=True requires cumesh (CUDA).  "
            "Falling back to simplification-only pipeline."
        )
        verts_np, faces_np = _simplify(verts_np, faces_np, decimation_target)
        verts_np, faces_np = _clean(verts_np, faces_np)

    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Mesh cleaning done.")

    # ── Step 3: UV parametrization ────────────────────────────────────────
    if use_tqdm:
        pbar.set_description("Parameterizing new mesh")
    if verbose:
        print("UV parametrization...")

    out_verts_np, out_faces_np, out_uvs_np, vmapping = _uv_unwrap(verts_np, faces_np)

    # Vertex normals for the UV-expanded mesh
    simplified_tm = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
    simplified_normals = np.asarray(simplified_tm.vertex_normals, dtype=np.float32)
    out_normals_np = simplified_normals[vmapping]

    if use_tqdm:
        pbar.update(1)
    if verbose:
        print(f"UV unwrap done: {len(out_verts_np)} verts, {len(out_faces_np)} faces")

    # ── Step 4: Texture baking (rasterise in UV space) ────────────────────
    if use_tqdm:
        pbar.set_description("Sampling attributes")
    if verbose:
        print("Sampling attributes...", end="", flush=True)

    # All tensors live on the original device; nvdiffrast GL context handles
    # the transfer to the GPU (OpenGL) automatically.
    out_verts_t  = torch.from_numpy(out_verts_np).float().to(device)
    out_faces_t  = torch.from_numpy(out_faces_np).int().to(device)
    out_uvs_t    = torch.from_numpy(out_uvs_np).float().to(device)

    use_cuda_ctx = str(device).startswith("cuda") and torch.cuda.is_available()
    ctx = dr.RasterizeCudaContext() if use_cuda_ctx else dr.RasterizeGLContext()

    # Render in UV space: UV coords become the "clip-space" xy position
    uvs_rast = torch.cat([
        out_uvs_t * 2 - 1,
        torch.zeros_like(out_uvs_t[:, :1]),
        torch.ones_like(out_uvs_t[:, :1]),
    ], dim=-1).unsqueeze(0)   # [1, V, 4]

    rast = torch.zeros((1, texture_size, texture_size, 4),
                       device=device, dtype=torch.float32)
    for i in range(0, out_faces_t.shape[0], 100_000):
        chunk = out_faces_t[i:i + 100_000]
        rast_chunk, _ = dr.rasterize(ctx, uvs_rast, chunk,
                                     resolution=[texture_size, texture_size])
        mask_chunk = rast_chunk[..., 3:4] > 0
        rast_chunk[..., 3:4] += i   # encode face-ID offset in alpha
        rast = torch.where(mask_chunk, rast_chunk, rast)

    mask = rast[0, ..., 3] > 0   # [T, T] bool

    # World-space position of each texel
    pos = dr.interpolate(out_verts_t.unsqueeze(0), rast, out_faces_t)[0][0]  # [T, T, 3]
    valid_pos = pos[mask]    # [K, 3]

    # Project back to the ORIGINAL (hole-filled) mesh for accurate attributes
    _, face_ids_t, uvw_t = _closest_point_barycentric(
        bvh_verts_np, bvh_faces_np, valid_pos)

    bvh_faces_torch = torch.from_numpy(bvh_faces_np).long().to(device)
    bvh_verts_torch = torch.from_numpy(bvh_verts_np).float().to(device)
    orig_tri_verts = bvh_verts_torch[bvh_faces_torch[face_ids_t]]  # [K, 3, 3]
    valid_pos = (orig_tri_verts * uvw_t.unsqueeze(-1)).sum(dim=1)  # [K, 3]

    # Trilinear attribute sampling from the sparse volume
    attrs_tex = torch.zeros(texture_size, texture_size,
                            attr_volume.shape[1], device=device)
    sampled = _grid_sample_3d_cpu(
        attr_volume,
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
        shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
        grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
        mode='trilinear',
    )  # [1, K, C]
    attrs_tex[mask] = sampled.squeeze(0)

    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")

    # ── Step 5: Texture post-processing & GLB construction ────────────────
    if use_tqdm:
        pbar.set_description("Finalizing mesh")
    if verbose:
        print("Finalizing mesh...", end="", flush=True)

    mask_np = mask.cpu().numpy()
    inv_mask = (~mask_np).astype(np.uint8)

    def _extract(channel: slice) -> np.ndarray:
        return np.clip(
            attrs_tex[..., channel].cpu().numpy() * 255, 0, 255
        ).astype(np.uint8)

    base_color = _extract(attr_layout['base_color'])
    metallic   = _extract(attr_layout['metallic'])
    roughness  = _extract(attr_layout['roughness'])
    alpha      = _extract(attr_layout['alpha'])

    base_color = cv2.inpaint(base_color,  inv_mask, 3, cv2.INPAINT_TELEA)
    metallic   = cv2.inpaint(metallic,    inv_mask, 1, cv2.INPAINT_TELEA)[..., None]
    roughness  = cv2.inpaint(roughness,   inv_mask, 1, cv2.INPAINT_TELEA)[..., None]
    alpha      = cv2.inpaint(alpha,       inv_mask, 1, cv2.INPAINT_TELEA)[..., None]

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(
            np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(
            np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode='OPAQUE',
        doubleSided=not remesh,
    )

    # Coordinate-system conversion (Y ↔ Z swap for GLB)
    verts_out  = out_verts_np.copy()
    norms_out  = out_normals_np.copy()
    uvs_out    = out_uvs_np.copy()

    verts_out[:, 1], verts_out[:, 2] = verts_out[:, 2].copy(), -verts_out[:, 1].copy()
    norms_out[:, 1], norms_out[:, 2] = norms_out[:, 2].copy(), -norms_out[:, 1].copy()
    uvs_out[:, 1] = 1 - uvs_out[:, 1]

    textured_mesh = trimesh.Trimesh(
        vertices=verts_out,
        faces=out_faces_np,
        vertex_normals=norms_out,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=uvs_out, material=material),
    )

    if use_tqdm:
        pbar.update(1)
        pbar.close()
    if verbose:
        print("Done")

    return textured_mesh
