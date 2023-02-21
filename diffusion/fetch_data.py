import os
import acronym_tools as data
import numpy as np
import trimesh

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACRONYM_DIR = 'dataset/acronym'
num_grasps = 50


def load_data():
    pass


def main():
    """
            reads acronym data and returns gripperconfiguration c and Object-mesh o according to a global coordinate system

            Returns:
                {[np.ndarray], [np.ndarray], [np.ndarray]} -- (contact_point,normal_vector,gripper_width) Gripperconfiguration c
                [np.nd.array]?                             -- object meshes
            """

    grasps = {}
    #list all grasping file paths
    grasps['path'] = [os.path.join(BASE_DIR, ACRONYM_DIR, 'grasps', g) for g in os.listdir(os.path.join(BASE_DIR, ACRONYM_DIR, 'grasps'))]
    grasps['transform'] = []
    grasps['success'] = []
    idx = []
    for i, f in enumerate(grasps['path'][1:1000]):

        try:
            # load object mesh
            grasps['obj_mesh'] = data.load_mesh(f, mesh_root_dir=os.path.join(BASE_DIR, ACRONYM_DIR))
            # get transformations and quality of all simulated grasps
            t, s = data.load_grasps(f)
            grasps['transform'].append(t)
            grasps['success'].append(s)
        except:
            idx.append(i)


    print(idx)
    print(len(grasps['path']), len(grasps['transform']))


if __name__ == "__main__":
    main()
