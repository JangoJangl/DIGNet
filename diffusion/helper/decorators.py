"""
File for all decorators used in the project
"""
import trimesh


def o3d_check(method):
    def inner(obj, *args):
        if isinstance(obj, trimesh.Trimesh):
            obj = obj.as_open3d
            obj.compute_vertex_normals()
        return method(obj, *args)
    return inner
