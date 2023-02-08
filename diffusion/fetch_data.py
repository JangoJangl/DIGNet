import os
import acronym_tools as data
import numpy as np

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

    #list all grasping file paths
    grasps = [os.path.join(BASE_DIR, ACRONYM_DIR, 'grasps', g) for g in os.listdir(os.path.join(BASE_DIR, ACRONYM_DIR, 'grasps'))]
    for f in grasps:
        # load object mesh
        #obj_mesh = data.load_mesh(f, mesh_root_dir=args.mesh_root)

        # get transformations and quality of all simulated grasps
        T, success = data.load_grasps(f)

        # create visual markers for grasps
        successful_grasps = [
            data.create_gripper_marker(color=[0, 255, 0]).apply_transform(t)
            for t in T[np.random.choice(np.where(success == 1)[0], num_grasps)]
        ]
        failed_grasps = [
            data.create_gripper_marker(color=[255, 0, 0]).apply_transform(t)
            for t in T[np.random.choice(np.where(success == 0)[0], num_grasps)]
        ]

        #trimesh.Scene([obj_mesh] + successful_grasps + failed_grasps).show()

if __name__ == "__main__":
    main()
