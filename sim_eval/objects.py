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


def load_mesh(object_root, mesh_file, scale):
    """ object mesh in the frame of the grasps: the source mesh scaled by `scale` (as in dataset_tools/sources.py) """
    import trimesh
    path = mesh_file if os.path.isabs(mesh_file) else os.path.join(object_root, mesh_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"mesh {path} not found (object_root={object_root})")
    mesh = trimesh.load(path, force='mesh', skip_materials=True)
    mesh.apply_scale(scale)
    return mesh


def object_mass(mesh, density=400.0, min_mass=0.02, max_mass=1.0):
    """ mass from the volume (convex hull volume for open meshes), clipped to what the grippers are meant for """
    volume = mesh.volume if mesh.is_watertight and mesh.volume > 0 else mesh.convex_hull.volume
    return float(np.clip(density * volume, min_mass, max_mass))


def _cache_key(mesh_file, scale, mode):
    return hashlib.md5(f'{mesh_file}|{scale:.6f}|{mode}'.encode()).hexdigest()[:16]


def coacd_parts(mesh, cache_dir, key, threshold=0.05, max_hulls=24, max_faces=20000):
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


def write_convex_usd(parts, path, mass, visual_mesh=None, friction=1.0):
    """
    Rigid body USD with one convex-hull collider per part (UsdPhysics only, readable by Newton and PhysX):
    /Object (Xform, RigidBodyAPI, MassAPI) / collisions / part_i (Mesh, CollisionAPI, convexHull), optional
    visual mesh without collision.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
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
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, 'physics')
    if visual_mesh is not None:
        add_mesh('/Object/visual', visual_mesh.vertices, visual_mesh.faces)
    stage.GetRootLayer().Save()
    return path


def prepare_object(record, object_root, cache_dir, collision='coacd', density=400.0):
    """
    record: dict with mesh_file and scale. Returns dict(mesh=trimesh, mass, usd=path or None) and writes the USD
    for the coacd mode (cached by mesh file, scale and mode).
    """
    if collision not in COLLISION_MODES:
        raise ValueError(f'collision must be one of {COLLISION_MODES}')
    mesh = load_mesh(object_root, record['mesh_file'], float(record['scale']))
    mass = object_mass(mesh, density)
    usd = None
    if collision == 'coacd':
        key = _cache_key(record['mesh_file'], float(record['scale']), collision)
        usd = os.path.join(cache_dir, f'{key}.usda')
        if not os.path.exists(usd):
            parts = coacd_parts(mesh, cache_dir, key)
            write_convex_usd(parts, usd, mass, visual_mesh=mesh)
    return {'mesh': mesh, 'mass': mass, 'usd': usd}
