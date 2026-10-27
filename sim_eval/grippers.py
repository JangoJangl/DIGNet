"""
Flying grippers for the simulation: geometry, URDF assets and the mapping of a grasp to the gripper root pose.

Grasps come in the GraspGen / ACRONYM Panda convention (dataset_tools/common.py): origin at the gripper base link,
approach +z, closing along x, finger tips at tcp_depth = 0.11217.

franka  Franka Hand, two prismatic fingers, URDF built from gripper/models/panda_gripper (already in the grasp
        convention frame), so a grasp is used as it is.
so101   SO-101 (LeRobot) gripper: one jaw rotates against a fixed jaw. Built from the gripper links of
        so101_new_calib.urdf (TheRobotStudio/SO-ARM100, Simulation/SO101). The Panda grasp is transferred by
        keeping approach and closing axis and the contact depth, and placing the fixed jaw on the +x contact of
        the object (see so101_root_pose), so the jaw closes without pushing the object around.

The numbers below for the SO-101 were measured on the URDF meshes (frame gripper_frame_link: approach +z, fixed
jaw on +x, moving jaw opens towards -x; see knowledgebase/state/sim-grasp-validation.md).
"""
import os
from dataclasses import dataclass, field

import numpy as np

from sim_eval.transforms import invert, rpy, translation

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PANDA_TCP_DEPTH = 0.11217     # base link -> point between the finger tips
PANDA_CONTACT_DEPTH = 0.1034  # base link -> centre of the finger pads (pad_z of dataset_tools/common.py)
PANDA_HALF_OPENING = 0.04

# SO-101, in gripper_frame_link
SO101_LINK_TO_FRAME = rpy(0.0, np.pi, 0.0, (-0.0079, -0.000218121, -0.0981274))  # gripper_frame_joint of the URDF
SO101_CONTACT_Z = -0.010     # pad centre, the jaw tips end at z = +0.005
SO101_FIXED_JAW_X = 0.002    # inner surface of the fixed jaw at the pad centre
SO101_JAW_LIMITS = (-0.174533, 1.74533)
SO101_JAW_JOINT = rpy(1.5708, -5.24284e-08, -1.41553e-15, (0.0202, 0.0188, -0.0234))  # joint 'gripper', in gripper_link
# inner surface x of the moving jaw at SO101_CONTACT_Z over the joint angle (above 0.75 rad the jaw has left the pad
# depth, i.e. 0.75 rad is the widest opening that still closes along the pads)
SO101_JAW_TABLE_Q = np.array([-0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75])
SO101_JAW_TABLE_X = np.array([-0.0079, -0.0145, -0.0210, -0.0275, -0.0340, -0.0408, -0.0479, -0.0557, -0.0619, -0.0716])

FIXED_JAW_CLEARANCE = 0.002   # gap between fixed jaw and object surface at the start
MOVING_JAW_CLEARANCE = 0.008  # gap between moving jaw and object surface at the start


@dataclass
class GripperSpec:
    name: str
    joint_names: list           # actuated finger joints, all closed by a position target with limited effort
    open_pos: list              # default pre-grasp joint positions (per grasp for the SO-101)
    closed_target: list
    effort_limit: float         # N per finger (prismatic) or N m (revolute)
    stiffness: float
    damping: float
    min_closed_gap: float       # joint value (sum for franka) below which the gripper closed on nothing
    root_to_tcp: np.ndarray = field(repr=False)  # tcp frame (approach z, closing x) in the root link frame
    urdf: str = ''
    close_speed: float = 0.0    # the closing target moves at this speed (m/s or rad/s), 0: it jumps to closed


def franka_spec(effort_limit=70.0):
    """
    Franka Hand: 70 N continuous grasp force, fingers 0..0.04 m each. The closing target moves at 0.1 m/s to beyond
    closed, so the grip force reaches the effort limit for thin objects too (higher stiffness is unstable with the
    15 g fingers at dt 1/240 s).
    """
    return GripperSpec(name='franka', joint_names=['panda_finger_joint1', 'panda_finger_joint2'],
                       open_pos=[0.04, 0.04], closed_target=[0.0, 0.0], effort_limit=effort_limit,
                       stiffness=3000.0, damping=100.0, min_closed_gap=0.002,
                       root_to_tcp=translation(z=PANDA_TCP_DEPTH), close_speed=0.1)


def so101_spec(effort_limit=1.5):
    """ SO-101 jaw with a STS3215 servo, effort_limit in N m (measure the real gripper, 1.5 is a guess) """
    return GripperSpec(name='so101', joint_names=['gripper'], open_pos=[0.75], closed_target=[SO101_JAW_LIMITS[0]],
                       effort_limit=effort_limit, stiffness=20.0, damping=0.5, min_closed_gap=SO101_JAW_LIMITS[0] + 0.05,
                       root_to_tcp=SO101_LINK_TO_FRAME)


SPECS = {'franka': franka_spec, 'so101': so101_spec}


# ------------------------------------------------------------------------------------------- grasp transfer

def closing_line_extent(mesh, grasp, depth=PANDA_CONTACT_DEPTH, half_range=PANDA_HALF_OPENING + 0.005,
                        offsets=((0.0, 0.0), (0.008, 0.0), (-0.008, 0.0), (0.0, 0.006), (0.0, -0.006))):
    """
    Object surface between the Panda fingers along the closing axis: (a, b) = smallest and largest x of the
    mesh surface on lines parallel to x through the pad centre (y, z offsets in the grasp frame), limited to
    |x| <= half_range. Returns (nan, nan) if the lines miss the object.
    """
    origins, directions = [], []
    for dy, dz in offsets:
        origins.append(grasp[:3, :3] @ np.array([-half_range, dy, depth + dz]) + grasp[:3, 3])
        directions.append(grasp[:3, 0])
    locations, ray_ids, _ = mesh.ray.intersects_location(np.array(origins), np.array(directions), multiple_hits=True)
    if len(locations) == 0:
        return np.nan, np.nan
    x = (locations - grasp[:3, 3]) @ grasp[:3, 0]
    x = x[(x >= -half_range) & (x <= half_range)]
    if len(x) == 0:
        return np.nan, np.nan
    return float(x.min()), float(x.max())


def so101_jaw_angle(x):
    """ joint angle at which the moving jaw's inner surface is at x (clipped to the table) """
    return float(np.interp(-x, -SO101_JAW_TABLE_X, SO101_JAW_TABLE_Q))


def so101_root_pose(grasp, extent):
    """
    Panda grasp (base link pose in the object frame) and closing_line_extent -> SO-101 gripper_link pose in the
    object frame, pre-grasp jaw angle and whether the object fits between the jaws.
    The fixed jaw is put FIXED_JAW_CLEARANCE outside the +x surface b, the moving jaw is opened until it is
    MOVING_JAW_CLEARANCE outside the -x surface a.
    """
    a, b = extent
    if not np.isfinite(a):  # the Panda fingers would close on nothing: keep the centre, still simulate
        a, b = 0.0, 0.0
    shift = b + FIXED_JAW_CLEARANCE - SO101_FIXED_JAW_X
    frame = grasp @ translation(z=PANDA_CONTACT_DEPTH) @ translation(x=shift, z=-SO101_CONTACT_Z)
    # moving jaw surface in gripper_frame_link coordinates
    a_frame = a - shift
    q_open = so101_jaw_angle(a_frame - MOVING_JAW_CLEARANCE)
    fits = bool(a_frame - MOVING_JAW_CLEARANCE >= SO101_JAW_TABLE_X[-1])
    return frame @ invert(SO101_LINK_TO_FRAME), q_open, fits


def root_poses(gripper, grasps, mesh=None):
    """
    Grasps [K, 4, 4] (Panda convention, object frame) -> root link poses [K, 4, 4] in the object frame,
    pre-grasp joint positions [K, J] and a fits flag [K] (object narrower than the maximum opening).
    """
    spec = SPECS[gripper]()
    grasps = np.asarray(grasps, dtype=np.float64)
    if gripper == 'franka':
        fits = np.ones(len(grasps), dtype=bool)
        if mesh is not None:
            ext = np.array([closing_line_extent(mesh, g) for g in grasps])
            fits = ~np.isfinite(ext[:, 0]) | (ext[:, 1] - ext[:, 0] <= 2 * PANDA_HALF_OPENING)
        return grasps.copy(), np.tile(spec.open_pos, (len(grasps), 1)), fits
    if mesh is None:
        raise ValueError('the SO-101 transfer needs the object mesh')
    poses, q_open, fits = [], [], []
    for g in grasps:
        pose, q, f = so101_root_pose(g, closing_line_extent(mesh, g))
        poses.append(pose)
        q_open.append([q])
        fits.append(f)
    return np.array(poses), np.array(q_open), np.array(fits)


def convex_halfspaces(meshes):
    """ convex meshes -> list of (normals [F, 3], offsets [F]) with n . p <= offset inside """
    out = []
    for m in meshes:
        hull = m.convex_hull
        normals = np.asarray(hull.face_normals)
        out.append((normals, np.einsum('ij,ij->i', normals, hull.triangles[:, 0])))
    return out


def colliding(points, halfspaces, margin=0.001):
    """ any of the points (gripper root frame) inside any convex part grown by margin """
    for normals, offsets in halfspaces:
        if (points @ normals.T <= offsets + margin).all(axis=1).any():
            return True
    return False


def back_off(pose, root_to_tcp, object_points, halfspaces, step=0.002, max_offset=0.02, margin=0.001):
    """
    Moves a gripper root pose (object frame) back along the approach axis in steps until the convex collision parts
    (root frame, at the pre-grasp opening) are free of the object surface points. Returns the pose and the offset;
    the offset is nan if the gripper still collides at max_offset (the pose at max_offset is returned).
    """
    for offset in np.arange(0.0, max_offset + 1e-9, step):
        moved = pose @ root_to_tcp @ translation(z=-offset) @ invert(root_to_tcp)
        points = (invert(moved) @ np.c_[object_points, np.ones(len(object_points))].T).T[:, :3]
        if not colliding(points, halfspaces, margin):
            return moved, float(offset)
    return moved, float('nan')


def transform_halfspaces(halfspaces, t):
    """ half-spaces of convex parts moved by the transform t """
    return [(normals @ t[:3, :3].T, offsets + normals @ t[:3, :3].T @ t[:3, 3]) for normals, offsets in halfspaces]


def so101_collision_halfspaces(asset_dir):
    """
    Convex collision parts of the SO-101 gripper (written by so101_urdf): half-spaces of the gripper_link parts and
    a function jaw angle -> half-spaces of all parts, in the gripper_link frame
    """
    import glob
    import trimesh
    files = sorted(glob.glob(os.path.join(asset_dir, 'so101_collision', '*.stl')))
    body = convex_halfspaces([trimesh.load(f) for f in files if not os.path.basename(f).startswith('jaw')])
    jaw = convex_halfspaces([trimesh.load(f) for f in files if os.path.basename(f).startswith('jaw')])
    return lambda q: body + transform_halfspaces(jaw, SO101_JAW_JOINT @ rpy(0.0, 0.0, q))


# ------------------------------------------------------------------------------------------------ URDF assets

def _box_inertia(mass, size):
    x, y, z = size
    return mass / 12.0 * np.array([y * y + z * z, x * x + z * z, x * x + y * y])


def _inertial(mass, com, size):
    ixx, iyy, izz = _box_inertia(mass, size)
    return (f'<inertial><origin xyz="{com[0]} {com[1]} {com[2]}" rpy="0 0 0"/><mass value="{mass}"/>'
            f'<inertia ixx="{ixx:.3e}" ixy="0" ixz="0" iyy="{iyy:.3e}" iyz="0" izz="{izz:.3e}"/></inertial>')


def franka_urdf(out_dir):
    """ Franka Hand in the grasp convention frame: hand.stl at the base link, fingers on prismatic joints along ±x """
    mesh_dir = os.path.join(REPO, 'gripper', 'models', 'panda_gripper')
    hand, finger = os.path.join(mesh_dir, 'hand.stl'), os.path.join(mesh_dir, 'finger.stl')
    urdf = f"""<?xml version="1.0"?>
<robot name="franka_hand">
  <link name="panda_hand">
    {_inertial(0.73, (0, 0, 0.02), (0.2, 0.06, 0.09))}
    <visual><geometry><mesh filename="{hand}"/></geometry></visual>
    <collision><geometry><mesh filename="{hand}"/></geometry></collision>
  </link>
  <link name="panda_leftfinger">
    {_inertial(0.015, (0.013, 0, 0.027), (0.026, 0.021, 0.054))}
    <visual><origin xyz="0 0 0" rpy="0 0 3.14159265"/><geometry><mesh filename="{finger}"/></geometry></visual>
    <collision><origin xyz="0 0 0" rpy="0 0 3.14159265"/><geometry><mesh filename="{finger}"/></geometry></collision>
  </link>
  <link name="panda_rightfinger">
    {_inertial(0.015, (-0.013, 0, 0.027), (0.026, 0.021, 0.054))}
    <visual><geometry><mesh filename="{finger}"/></geometry></visual>
    <collision><geometry><mesh filename="{finger}"/></geometry></collision>
  </link>
  <joint name="panda_finger_joint1" type="prismatic">
    <parent link="panda_hand"/><child link="panda_leftfinger"/>
    <origin xyz="0 0 0.0584" rpy="0 0 0"/><axis xyz="1 0 0"/>
    <limit lower="0.0" upper="0.04" effort="100" velocity="0.2"/>
  </joint>
  <joint name="panda_finger_joint2" type="prismatic">
    <parent link="panda_hand"/><child link="panda_rightfinger"/>
    <origin xyz="0 0 0.0584" rpy="0 0 0"/><axis xyz="-1 0 0"/>
    <limit lower="0.0" upper="0.04" effort="100" velocity="0.2"/>
  </joint>
</robot>
"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'franka_hand.urdf')
    with open(path, 'w') as f:
        f.write(urdf)
    return path


def _convex_parts(mesh, cache_prefix, max_hulls=16):
    """ CoACD convex decomposition, written as STL files <cache_prefix>_<i>.stl (cached) """
    import glob
    import trimesh
    parts = sorted(glob.glob(f'{cache_prefix}_*.stl'))
    if parts:
        return parts
    import coacd
    coacd.set_log_level('error')
    hulls = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=0.05, max_convex_hull=max_hulls,
                            preprocess_resolution=40, resolution=1000, mcts_iterations=60)
    for i, (v, f) in enumerate(hulls):
        trimesh.Trimesh(v, f).convex_hull.export(f'{cache_prefix}_{i:02d}.stl')
    return sorted(glob.glob(f'{cache_prefix}_*.stl'))


def so101_urdf(so101_dir, out_dir):
    """
    Gripper only URDF from so101_new_calib.urdf: gripper_link (wrist roll follower with the fixed jaw and the
    gripper servo) as the root and the moving jaw on the revolute joint 'gripper'. The visual meshes are kept; the
    collision meshes are convex decompositions (a single convex hull of the fixed jaw part would fill the space
    between the jaws).
    so101_dir: folder with so101_new_calib.urdf and assets/ (SO-ARM100/Simulation/SO101)
    """
    import trimesh
    assets = os.path.join(so101_dir, 'assets')
    os.makedirs(os.path.join(out_dir, 'so101_collision'), exist_ok=True)
    follower_origin = rpy(-3.14159, -5.55112e-17, 0.0, (8.32667e-17, -0.000218214, 0.000949706))
    servo_origin = rpy(-1.5708, -5.19179e-17, -1.66533e-16, (0.0077, 0.0001, -0.0234))
    jaw_origin = translation(z=0.0189)

    def collisions(name, origin, prefix):
        mesh = trimesh.load(os.path.join(assets, name), force='mesh')
        mesh.apply_transform(origin)  # parts in link coordinates, no origin needed in the URDF
        parts = _convex_parts(mesh, os.path.join(out_dir, 'so101_collision', prefix))
        return '\n    '.join(f'<collision><geometry><mesh filename="{p}"/></geometry></collision>' for p in parts)

    def visual(name, origin_xyz, origin_rpy):
        return (f'<visual><origin xyz="{origin_xyz}" rpy="{origin_rpy}"/>'
                f'<geometry><mesh filename="{os.path.join(assets, name)}"/></geometry></visual>')

    urdf = f"""<?xml version="1.0"?>
<robot name="so101_gripper">
  <link name="gripper_link">
    <inertial><origin xyz="0.000213627 0.000245138 -0.025187" rpy="0 0 0"/><mass value="0.087"/>
      <inertia ixx="2.75087e-05" ixy="-3.35241e-07" ixz="-5.7352e-06" iyy="4.33657e-05" iyz="-5.17847e-08" izz="3.45059e-05"/></inertial>
    {visual('sts3215_03a_v1.stl', '0.0077 0.0001 -0.0234', '-1.5708 -5.19179e-17 -1.66533e-16')}
    {visual('wrist_roll_follower_so101_v1.stl', '8.32667e-17 -0.000218214 0.000949706', '-3.14159 -5.55112e-17 0')}
    {collisions('wrist_roll_follower_so101_v1.stl', follower_origin, 'follower')}
    {collisions('sts3215_03a_v1.stl', servo_origin, 'servo')}
  </link>
  <link name="moving_jaw_so101_v1_link">
    <inertial><origin xyz="-0.00157495 -0.0300244 0.0192755" rpy="0 0 0"/><mass value="0.012"/>
      <inertia ixx="6.61427e-06" ixy="-3.19807e-07" ixz="-5.90717e-09" iyy="1.89032e-06" iyz="-1.09945e-07" izz="5.28738e-06"/></inertial>
    {visual('moving_jaw_so101_v1.stl', '0 0 0.0189', '0 0 0')}
    {collisions('moving_jaw_so101_v1.stl', jaw_origin, 'jaw')}
  </link>
  <joint name="gripper" type="revolute">
    <origin xyz="0.0202 0.0188 -0.0234" rpy="1.5708 -5.24284e-08 -1.41553e-15"/>
    <parent link="gripper_link"/><child link="moving_jaw_so101_v1_link"/>
    <axis xyz="0 0 1"/>
    <limit effort="10" velocity="10" lower="{SO101_JAW_LIMITS[0]}" upper="{SO101_JAW_LIMITS[1]}"/>
  </joint>
</robot>
"""
    path = os.path.join(out_dir, 'so101_gripper.urdf')
    with open(path, 'w') as f:
        f.write(urdf)
    return path
