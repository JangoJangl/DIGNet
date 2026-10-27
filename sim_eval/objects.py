"""
Object meshes for the simulation: loading (same as dataset_tools), mass and collision assets.

Collision modes
  coacd        CoACD convex decomposition, one USD per object with a convex hull per part (default, cached)
  convex_hull  a single convex hull (fast, but concave objects such as mugs and bowls become solid)
  sdf          Newton SDF collision of the triangle mesh (needs reasonably closed meshes)
"""
import hashlib
import os

import numpy as np

COLLISION_MODES = ('coacd', 'convex_hull', 'sdf')
MESH_PREPS = ('raw', 'graspgen')
COACD_MAX_HULLS = 24


def load_mesh(object_root, mesh_file, scale):
    """ object mesh in the frame of the grasps: the source mesh scaled by `scale` (as in dataset_tools/sources.py) """
    import trimesh
    path = mesh_file if os.path.isabs(mesh_file) else os.path.join(object_root, mesh_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"mesh {path} not found (object_root={object_root})")
    mesh = trimesh.load(path, force='mesh', skip_materials=True)
    mesh.apply_scale(scale)
    return mesh


def watertight_simplified(mesh, cache_dir, key, voxels=128, max_faces=5000):
    """
    Watertight, decimated version of a mesh, the kind of mesh the GraspGen labels were simulated on
    (GraspGen download_objects.py --simplify: Manifold, then simplify). Approximated without these tools: the surface
    is voxelized (voxels along the longest side), enclosed space is filled, the boundary is extracted with marching
    cubes and decimated to max_faces. Open shells become walls of about one voxel. The volume of the filled voxels
    is kept in metadata['solid_volume']. Cached as npz.
    """
    path = os.path.join(cache_dir, f'{key}.npz')
    import trimesh
    if os.path.exists(path):
        data = np.load(path)
        if 'solid_volume' in data:
            out = trimesh.Trimesh(data['vertices'], data['faces'])
            out.metadata['solid_volume'] = float(data['solid_volume'])
            return out
    pitch = mesh.extents.max() / voxels
    grid = mesh.voxelized(pitch).fill()
    solid_volume = float(grid.filled_count * pitch ** 3)
    out = grid.marching_cubes.apply_transform(grid.transform)  # marching cubes are in voxel indices
    if len(out.faces) > max_faces:
        import fast_simplification
        v, f = fast_simplification.simplify(out.vertices, out.faces, target_reduction=1.0 - max_faces / len(out.faces))
        out = trimesh.Trimesh(v, f)
    out.fix_normals()
    out.metadata['solid_volume'] = solid_volume
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(path, vertices=np.asarray(out.vertices, np.float64), faces=np.asarray(out.faces, np.int64),
             solid_volume=solid_volume)
    return out


def object_mass(mesh, density=400.0, min_mass=0.02, max_mass=0.5, solid_volume=None, shell=0.003):
    """
    mass from the solid volume (filled voxels of watertight_simplified, or the volume of a watertight mesh), at least
    that of a shell of the surface area with thickness `shell` (open and thin meshes), clipped to what the grippers
    are meant for. (The convex hull made flat and hollow objects far too heavy.)
    """
    if solid_volume is None:
        solid_volume = mesh.volume if mesh.is_watertight and mesh.volume > 0 else 0.0
    return float(np.clip(density * max(solid_volume, mesh.area * shell), min_mass, max_mass))


def _cache_key(mesh_file, scale, mode):
    return hashlib.md5(f'{mesh_file}|{scale:.6f}|{mode}'.encode()).hexdigest()[:16]


def coacd_parts(mesh, cache_dir, key, threshold=0.05, max_hulls=COACD_MAX_HULLS, max_faces=20000):
    """ convex parts (list of (vertices, faces)) of a mesh, cached as npz """
    path = os.path.join(cache_dir, f'{key}.npz')
    if os.path.exists(path):
        data = np.load(path)
        return [(data[f'v{i}'], data[f'f{i}']) for i in range(int(data['n']))]
    import coacd
    import trimesh
    coacd.set_log_level('error')
    if len(mesh.faces) > max_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=max_faces)
        except Exception:  # fast_simplification missing
            pass
    hulls = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=threshold, max_convex_hull=max_hulls,
                            preprocess_resolution=50, resolution=2000, mcts_iterations=100)
    parts = []
    for v, f in hulls:
        hull = trimesh.Trimesh(v, f).convex_hull
        parts.append((np.asarray(hull.vertices, np.float32), np.asarray(hull.faces, np.int32)))
    os.makedirs(cache_dir, exist_ok=True)
    arrays = {'n': len(parts)}
    for i, (v, f) in enumerate(parts):
        arrays[f'v{i}'], arrays[f'f{i}'] = v, f
    np.savez(path, **arrays)
    return parts


def pad_parts(parts, n):
    """
    Pads a convex decomposition to n parts with 1 mm tetrahedra at the centre of the largest part (inside the object).
    Newton's views need the same number of shapes in every world, so all objects of a chunk get n parts.
    """
    v, _ = max(parts, key=lambda p: np.ptp(p[0], axis=0).prod())
    tet_v = (v.mean(axis=0) + 0.001 * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])).astype(np.float32)
    tet_f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int32)
    return list(parts) + [(tet_v, tet_f)] * (n - len(parts))


def write_convex_usd(parts, path, mass, visual_mesh=None, friction=1.0, condim=3):
    """
    Rigid body USD with one convex-hull collider per part (UsdPhysics only, readable by Newton and PhysX):
    /Object (Xform, RigidBodyAPI, MassAPI) / collisions / part_i (Mesh, CollisionAPI, convexHull), optional
    visual mesh without collision.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, '/Object')
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(mass)
    material = UsdShade.Material.Define(stage, '/Object/PhysicsMaterial')
    mat_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    mat_api.CreateStaticFrictionAttr(friction)
    mat_api.CreateDynamicFrictionAttr(friction)
    mat_api.CreateRestitutionAttr(0.0)

    def add_mesh(prim_path, vertices, faces):
        mesh = UsdGeom.Mesh.Define(stage, prim_path)
        mesh.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in vertices])
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr([int(i) for i in np.asarray(faces).reshape(-1)])
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        return mesh

    UsdGeom.Scope.Define(stage, '/Object/collisions')
    for i, (v, f) in enumerate(parts):
        mesh = add_mesh(f'/Object/collisions/part_{i:02d}', v, f)
        UsdGeom.Imageable(mesh).CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
        if condim != 3:  # MuJoCo contact dimension (4: torsional friction), the larger one of a pair is used
            mesh.GetPrim().CreateAttribute('mjc:condim', Sdf.ValueTypeNames.Int).Set(int(condim))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, 'physics')
    if visual_mesh is not None:
        add_mesh('/Object/visual', visual_mesh.vertices, visual_mesh.faces)
    stage.GetRootLayer().Save()
    return path


def prepare_object(record, object_root, cache_dir, collision='coacd', density=400.0, mesh_prep='raw', condim=3):
    """
    record: dict with mesh_file and scale. Returns dict(mesh=trimesh, mass, usd=path or None) and writes the USD
    for the coacd mode (cached by mesh file, scale and mode). mesh_prep='graspgen' simulates a watertight, decimated
    version of the mesh (watertight_simplified). condim is written to the colliders of the coacd USD.
    """
    if collision not in COLLISION_MODES:
        raise ValueError(f'collision must be one of {COLLISION_MODES}')
    if mesh_prep not in MESH_PREPS:
        raise ValueError(f'mesh_prep must be one of {MESH_PREPS}')
    mesh = load_mesh(object_root, record['mesh_file'], float(record['scale']))
    key = _cache_key(record['mesh_file'], float(record['scale']), 'graspgen')
    prepared = watertight_simplified(mesh, cache_dir, key)
    mass = object_mass(mesh, density, solid_volume=prepared.metadata['solid_volume'])
    if mesh_prep == 'graspgen':
        mesh = prepared
    usd = None
    if collision == 'coacd':
        key = _cache_key(record['mesh_file'], float(record['scale']), collision if mesh_prep == 'raw' else
                         f'{collision}-{mesh_prep}')
        usd = os.path.join(cache_dir, f'{key}.usda' if condim == 3 else f'{key}-condim{condim}.usda')
        if not os.path.exists(usd):
            parts = pad_parts(coacd_parts(mesh, cache_dir, key), COACD_MAX_HULLS)
            write_convex_usd(parts, usd, mass, visual_mesh=mesh, condim=condim)
    return {'mesh': mesh, 'mass': mass, 'usd': usd}
