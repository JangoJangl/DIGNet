"""
Executes grasps of an episode file with a flying gripper in Isaac Lab 3 on the Newton (MuJoCo-Warp) backend.
Isaac Sim is not needed (kit-less Isaac Lab, see sim_eval/README.md).

One environment per grasp. The gripper root is fixed in its environment and the object is placed relative to it,
so that the gripper is at the grasp pose. Gravity is off; instead the test applies an inertial load to the object:

  settle  0.10 s  fingers open, no load. The object moving more than 5 mm means the open gripper collides with it
  close   0.60 s  position target "closed" with limited effort (Franka 70 N per finger, SO-101 1.5 N m)
  load    6 x 0.40 s  force m * g * load_g along +z, -z, +x, -x, +y, -y of the gripper (pull out, push in, along
          the closing axis and along the pads): equivalent to lifting and shaking the gripper with that acceleration

success = 1 if the object is still between the fingers at the end (centre of mass moved less than 2 cm relative to
the gripper since closing and the fingers did not close completely), else 0. Grasps that collide at the start
count as 0 (--keep-colliding to judge them like the others).

--protocol graspdatagen follows the simulation that produced the GraspGen labels (NVlabs/GraspDataGen,
scripts/graspgen/grasp_sim.py): object mass 1 kg, dt 1/250 s, close 1.0 s, then 5 tugs of 0.5 s with m * g along
[0, 0, 1], [0, ±2, 1], [±2, 0, 1] of their gripper frame (their y is the closing axis, our x), success = both
fingers still touch the object at the end. The finger drive stays ours (their USD drive has stiffness 0).

  python -m sim_eval.run_newton --episodes ep.npz --gripper franka --object-root ~/objaverse --chunk 0 \
      --objects-per-chunk 16 --out results/franka_000.csv [--viz newton]
"""
import argparse
import csv
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# Kit-less URDF import: the importer registers its PhysX schemas only when it is imported, after Isaac Lab has built
# the USD schema registry ('Cannot find a valid schema for PhysxJointAPI'). OpenUSD reads this path when it builds it.
import importlib.util  # noqa: E402
_isaacsim = importlib.util.find_spec('isaacsim')
if _isaacsim is not None:
    for _root in _isaacsim.submodule_search_locations:
        _schemas = os.path.join(_root, 'asset', 'importer', 'utils', 'impl', 'schemas')
        if os.path.isdir(_schemas):
            _path = os.environ.get('PXR_PLUGINPATH_NAME')
            os.environ['PXR_PLUGINPATH_NAME'] = os.pathsep.join(filter(None, (_path, _schemas)))

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--episodes', required=True)
parser.add_argument('--gripper', choices=('franka', 'so101'), required=True)
parser.add_argument('--object-root', required=True, help='root of the mesh_file paths (Objaverse download)')
parser.add_argument('--so101-dir', default=os.environ.get('SO101_DIR', ''),
                    help='SO-ARM100/Simulation/SO101 with so101_new_calib.urdf and assets/')
parser.add_argument('--cache-dir', default=os.path.expanduser('~/.cache/dignet_sim'))
parser.add_argument('--chunk', type=int, default=0)
parser.add_argument('--objects-per-chunk', type=int, default=16)
parser.add_argument('--out', required=True, help='CSV with one row per grasp')
parser.add_argument('--collision', choices=('coacd', 'convex_hull', 'sdf'), default='coacd')
parser.add_argument('--friction', type=float, default=1.0)
parser.add_argument('--density', type=float, default=400.0, help='object density in kg/m^3 (mass clipped to 0.02..1 kg)')
parser.add_argument('--effort', type=float, default=None, help='grasp effort (N per finger / N m), default per gripper')
parser.add_argument('--load-g', type=float, default=1.0, help='load as a multiple of g')
parser.add_argument('--dt', type=float, default=1.0 / 240.0)
parser.add_argument('--substeps', type=int, default=2)
parser.add_argument('--contact-ke', type=float, default=1e5,
                    help='contact stiffness in N/m (the Newton default 2500 lets the fingers sink in by centimetres)')
parser.add_argument('--contact-kd', type=float, default=1e3, help='contact damping in N s/m')
parser.add_argument('--iterations', type=int, default=100)
parser.add_argument('--cone', choices=('pyramidal', 'elliptic'), default='elliptic')
parser.add_argument('--impratio', type=float, default=100.0,
                    help='friction to normal impedance ratio, high values stop objects creeping out along the pads')
parser.add_argument('--keep-colliding', action='store_true')
parser.add_argument('--mesh-prep', choices=('raw', 'graspgen'), default='raw',
                    help='graspgen: simulate a watertight, decimated mesh like the GraspGen labels (see objects.py)')
parser.add_argument('--protocol', choices=('dignet', 'graspdatagen'), default='dignet')
parser.add_argument('--env-spacing', type=float, default=1.0)
parser.add_argument('--video', default=None, help='record the run headless to this mp4 (Newton GL, half speed)')
add_launcher_args(parser)
args = parser.parse_args()
if args.video:  # EGL offscreen rendering also when a display is set (pyglet decides at import)
    import pyglet
    pyglet.options['headless'] = True

import numpy as np  # noqa: E402

from sim_eval.episodes import load_episodes  # noqa: E402
from sim_eval.grippers import SPECS, franka_urdf, root_poses, so101_urdf  # noqa: E402
from sim_eval.objects import prepare_object  # noqa: E402
from sim_eval.transforms import invert, matrix_from_pose7, pose7_from_matrix  # noqa: E402

PHASES = [('settle', 0.10, None), ('close', 0.60, None)] + [
    (f'load{name}', 0.40, axis) for name, axis in (('+z', (0, 0, 1)), ('-z', (0, 0, -1)), ('+x', (1, 0, 0)),
                                                   ('-x', (-1, 0, 0)), ('+y', (0, 1, 0)), ('-y', (0, -1, 0)))]
# GraspDataGen tugs, (x, y, z) in our gripper frame (closing axis x)
GRASPDATAGEN_PHASES = [('close', 1.0, None)] + [
    (f'tug{i}', 0.5, axis) for i, axis in enumerate(((0, 0, 1), (2, 0, 1), (-2, 0, 1), (0, 2, 1), (0, -2, 1)))]
FINGER_LINKS = {'franka': ('panda_leftfinger', 'panda_rightfinger'),
                'so101': ('gripper_link', 'moving_jaw_so101_v1_link')}
COLLISION_SHIFT = 0.005
HELD_SHIFT = 0.02
GRIPPER_HEIGHT = 0.5


def chunk_episodes(episodes, chunk, per_chunk):
    """ object indices and grasp rows of one chunk """
    n_obj = len(episodes['objects'])
    objects = list(range(chunk * per_chunk, min((chunk + 1) * per_chunk, n_obj)))
    rows = np.flatnonzero(np.isin(episodes['object_index'], objects))
    return objects, rows


def object_spawn_cfg(sim_utils, prepared, friction, collision):
    from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
    material = RigidBodyMaterialBaseCfg(static_friction=friction, dynamic_friction=friction)
    mass = sim_utils.MassCfg(mass=prepared['mass'])
    if collision == 'coacd':
        return sim_utils.UsdFileCfg(usd_path=prepared['usd'], mass_props=mass, physics_material=material)
    if collision == 'sdf':
        from isaaclab_newton.sim.schemas import NewtonSDFCollisionCfg
        mesh_collision = [sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name='sdf'),
                          NewtonSDFCollisionCfg(sdf_max_resolution=128)]
    else:
        mesh_collision = sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name='convexHull')
    # vertex/face arrays, not TrimeshObjectCfg: configclass validation recurses into the trimesh object
    mesh = sim_utils.MeshFileCfg.TriangleMeshCfg(vertices=np.asarray(prepared['mesh'].vertices, np.float32),
                                                 faces=np.asarray(prepared['mesh'].faces, np.int32))
    return sim_utils.MeshFileCfg(mesh=mesh,
                                 rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(), mass_props=mass,
                                 collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                                 mesh_collision_props=mesh_collision, physics_material=material)


def finger_contacts(fingers, num_envs, tol=1e-3):
    """ [num_envs, 2] bool: finger link fingers[i] (fingers[1] may be a child of fingers[0]) touches the object """
    from isaaclab_newton.physics import NewtonManager
    model, contacts = NewtonManager.get_model(), NewtonManager.get_contacts()
    body_q = matrix_from_pose7(NewtonManager.get_state_0().body_q.numpy().astype(np.float64))
    shape_body, shape_world = model.shape_body.numpy(), model.shape_world.numpy()
    role = np.array([1 if f'/{fingers[1]}/' in label else 0 if f'/{fingers[0]}/' in label
                     else 2 if '/Object/' in label else -1 for label in model.shape_label])
    n = int(contacts.rigid_contact_count.numpy()[0])
    s0, s1 = contacts.rigid_contact_shape0.numpy()[:n], contacts.rigid_contact_shape1.numpy()[:n]
    p0, p1 = contacts.rigid_contact_point0.numpy()[:n], contacts.rigid_contact_point1.numpy()[:n]
    normal = contacts.rigid_contact_normal.numpy()[:n]
    thickness = contacts.rigid_contact_margin0.numpy()[:n] + contacts.rigid_contact_margin1.numpy()[:n]
    out = np.zeros((num_envs, 2), dtype=bool)
    pairs = ((role[s0] == 2) & np.isin(role[s1], (0, 1))) | ((role[s1] == 2) & np.isin(role[s0], (0, 1)))
    for c in np.flatnonzero(pairs):
        w0 = body_q[shape_body[s0[c]]] @ np.append(p0[c], 1.0)
        w1 = body_q[shape_body[s1[c]]] @ np.append(p1[c], 1.0)
        if np.dot(normal[c], w1[:3] - w0[:3]) - thickness[c] <= tol:
            finger = s0[c] if role[s1[c]] == 2 else s1[c]
            out[shape_world[finger], role[finger]] = True
    return out


def env_object_assignment(scene, num_envs):
    """ chunk-local object index of every environment, read from the clone plan of the scene """
    plan = scene.clone_plan
    spawn = scene.cfg.object.spawn  # the scene's copy of the config, the plan refers to its spawners
    object_spawns = getattr(spawn, 'assets_cfg', None) or [spawn]
    ids = {id(cfg): i for i, cfg in enumerate(object_spawns)}
    topo = plan.topology
    layout, members, starts = (np.asarray(topo.world_prototype_layout), np.asarray(topo.world_prototypes),
                               np.asarray(topo.world_prototype_starts))
    out = np.full(num_envs, -1)
    for env in range(num_envs):
        world = layout[env]
        for proto in members[starts[world + 1]:starts[world + 2]]:
            spawn = getattr(plan.asset_cfgs[proto], 'spawn', None)
            children = getattr(spawn, 'assets_cfg', None) or [spawn]
            for child in children:
                if id(child) in ids:
                    out[env] = ids[id(child)]
    if len(object_spawns) == 1:
        out[:] = 0
    if (out < 0).any():
        raise RuntimeError('could not read the object of every environment from the clone plan')
    return out


def main():
    episodes = load_episodes(args.episodes)
    k = int(episodes['meta']['grasps_per_object'])
    objects, rows = chunk_episodes(episodes, args.chunk, args.objects_per_chunk)
    if not objects:
        print(f'chunk {args.chunk} is empty')
        return
    spec = SPECS[args.gripper]() if args.effort is None else SPECS[args.gripper](effort_limit=args.effort)
    asset_dir = os.path.join(args.cache_dir, 'grippers')
    if args.gripper == 'franka':
        urdf = franka_urdf(asset_dir)
    else:
        if not os.path.exists(os.path.join(args.so101_dir, 'so101_new_calib.urdf')):
            sys.exit('--so101-dir must point to SO-ARM100/Simulation/SO101')
        urdf = so101_urdf(args.so101_dir, asset_dir)

    # object assets and the gripper root pose / pre-grasp opening of every grasp (CPU, before the simulation)
    prepared, grasp_rows = {}, {}
    for obj in objects:
        prepared[obj] = prepare_object(episodes['objects'][obj], args.object_root,
                                       os.path.join(args.cache_dir, 'objects'), args.collision, args.density,
                                       args.mesh_prep)
        if args.protocol == 'graspdatagen':
            prepared[obj]['mass'] = 1.0
        idx = rows[episodes['object_index'][rows] == obj]
        poses, q_open, fits = root_poses(args.gripper, episodes['grasps'][idx], prepared[obj]['mesh'])
        grasp_rows[obj] = list(zip(idx, poses, q_open, fits))
    num_envs = len(objects) * k

    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg

    material = RigidBodyMaterialBaseCfg(static_friction=args.friction, dynamic_friction=args.friction)
    physics = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(solver='newton', integrator='implicitfast', njmax=1200, nconmax=300,
                                   impratio=args.impratio, cone=args.cone, update_data_interval=1,
                                   iterations=args.iterations, ls_iterations=50, use_mujoco_contacts=False,
                                   ccd_iterations=35),
        collision_cfg=NewtonCollisionPipelineCfg(rigid_contact_max=1000 * num_envs),
        default_shape_cfg=NewtonShapeCfg(mu=args.friction, ke=args.contact_ke, kd=args.contact_kd),
        num_substeps=args.substeps)
    dt = 1.0 / 250.0 if args.protocol == 'graspdatagen' else args.dt
    sim_cfg = sim_utils.SimulationCfg(dt=dt, device=args.device, gravity=(0.0, 0.0, 0.0), physics=physics,
                                      physics_material=material)
    if args.video:
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg
        sim_cfg.visualizer_cfgs = NewtonGLVisualizerCfg(headless=True, window_width=1280, window_height=720,
                                                        enable_picking=False, show_collision=True)

    with launch_simulation(sim_cfg, args):
        import torch

        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg, RigidObjectCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.sim import SimulationContext
        from isaaclab.utils import configclass, instantiate

        object_spawns = [object_spawn_cfg(sim_utils, prepared[o], args.friction, args.collision) for o in objects]
        drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=spec.stiffness, damping=spec.damping))

        @configclass
        class GraspSceneCfg(InteractiveSceneCfg):
            gripper = ArticulationCfg(
                prim_path='{ENV_REGEX_NS}/Gripper',
                spawn=sim_utils.UrdfFileCfg(
                    asset_path=urdf, usd_dir=os.path.join(asset_dir, 'usd'), fix_base=True, joint_drive=drive,
                    physics_material=material),
                init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, GRIPPER_HEIGHT), rot=(0.0, 0.0, 0.0, 1.0),
                                                           joint_pos=dict(zip(spec.joint_names, spec.open_pos))),
                actuators={'fingers': ImplicitActuatorCfg(joint_names_expr=spec.joint_names, stiffness=spec.stiffness,
                                                          damping=spec.damping, joint_effort_limit=spec.effort_limit)})
            object = RigidObjectCfg(
                prim_path='{ENV_REGEX_NS}/Object',
                spawn=object_spawns[0] if len(object_spawns) == 1 else sim_utils.MultiAssetSpawnerCfg(
                    assets_cfg=object_spawns),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, GRIPPER_HEIGHT)))

        sim = SimulationContext(sim_cfg)
        sim.set_camera_view([1.0, 1.0, 1.2], [0.0, 0.0, GRIPPER_HEIGHT])
        scene = instantiate(GraspSceneCfg(num_envs=num_envs, env_spacing=args.env_spacing, replicate_physics=True))
        sim.reset()
        gripper, obj = scene['gripper'], scene['object']
        device = sim.device

        # grasp of every environment: the next unused grasp of the object the environment got
        env_obj = env_object_assignment(scene, num_envs)
        queues = {i: list(grasp_rows[o]) for i, o in enumerate(objects)}
        assigned = [queues[i].pop(0) for i in env_obj]
        grasp_idx = np.array([a[0] for a in assigned])
        root_in_obj = np.stack([a[1] for a in assigned])
        q_open = np.stack([a[2] for a in assigned])
        fits = np.array([a[3] for a in assigned])

        joint_ids, _ = gripper.find_joints(spec.joint_names, preserve_order=True)
        gripper_w = matrix_from_pose7(gripper.data.root_link_pose_w.torch.cpu().double().numpy())
        object_w = gripper_w @ invert(root_in_obj)
        tcp_rot = torch.as_tensor((gripper_w @ spec.root_to_tcp)[:, :3, :3], dtype=torch.float32, device=device)

        obj.write_root_pose_to_sim_index(root_pose=torch.as_tensor(pose7_from_matrix(object_w), dtype=torch.float32,
                                                                   device=device))
        obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros(num_envs, 6, device=device))
        q_open_t = torch.as_tensor(q_open, dtype=torch.float32, device=device)
        gripper.write_joint_position_to_sim_index(position=q_open_t, joint_ids=joint_ids)
        gripper.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q_open_t), joint_ids=joint_ids)
        gripper.actuators.target_command.set_position_index(value=q_open_t, joint_ids=joint_ids)
        scene.reset()
        mass = obj.data.body_mass.torch.reshape(num_envs, -1)[:, :1].to(torch.float32)
        closed = torch.tensor(spec.closed_target, dtype=torch.float32, device=device).expand(num_envs, -1)
        com = lambda: obj.data.root_com_pose_w.torch[:, :3].clone()  # noqa: E731
        start_com, ref_com = com(), None
        shifts, first_fail = {}, np.full(num_envs, '', dtype=object)
        sim_dt = sim.get_physics_dt()
        frames, frame_every = [], max(1, int(round(1.0 / (60.0 * sim_dt))))  # 60 frames per simulated second
        if args.video:
            roots = gripper.data.root_link_pose_w.torch[:, :3].cpu().numpy()
            centre, size = roots.mean(axis=0), np.ptp(roots[:, :2], axis=0).max() + args.env_spacing
            sim.set_camera_view((centre + size * np.array([0.0, -0.75, 0.5])).tolist(), centre.tolist())
        phases = PHASES if args.protocol == 'dignet' else GRASPDATAGEN_PHASES
        for name, duration, axis in phases:
            if name == 'close':
                gripper.actuators.target_command.set_position_index(value=closed, joint_ids=joint_ids)
            if axis is not None:
                direction = tcp_rot @ torch.tensor(axis, dtype=torch.float32, device=device)
                force = (mass * 9.81 * args.load_g) * direction
                obj.permanent_wrench_composer.set_forces_and_torques_index(
                    forces=force[:, None, :], torques=torch.zeros_like(force)[:, None, :], is_global=True)
            for i in range(max(1, int(round(duration / sim_dt)))):
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim_dt)
                if args.video and i % frame_every == 0:
                    frames.append(sim.visualizers[0].render_rgb_array())
            now = com()
            if name == 'settle':
                shifts['initial'] = torch.linalg.norm(now - start_com, dim=1).cpu().numpy()
            elif name == 'close':
                ref_com = now
            else:
                s = torch.linalg.norm(now - ref_com, dim=1).cpu().numpy()
                shifts[name] = s
                first_fail[(s > HELD_SHIFT) & (first_fail == '')] = name

        joint_pos = gripper.data.joint_pos.torch[:, joint_ids].cpu().numpy()
        gap = joint_pos.sum(axis=1) if args.gripper == 'franka' else joint_pos[:, 0]
        final_shift = shifts[phases[-1][0]]
        collided = shifts.get('initial', np.zeros(num_envs)) > COLLISION_SHIFT
        touching = finger_contacts(FINGER_LINKS[args.gripper], num_envs).sum(axis=1)
        held = (final_shift < HELD_SHIFT) & (gap > spec.min_closed_gap)
        if args.protocol == 'graspdatagen':
            held = touching == 2
        success = held & (~collided | args.keep_colliding)
        first_fail[(gap <= spec.min_closed_gap) & (first_fail == '')] = 'closed_empty'

        if args.video:
            import imageio
            os.makedirs(os.path.dirname(os.path.abspath(args.video)), exist_ok=True)
            imageio.mimwrite(args.video, frames, fps=30, macro_block_size=1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['grasp', 'gripper', 'object_index', 'object_id', 'obs_index', 'label', 'success',
                         'initial_collision', 'fits', 'first_failed_phase', 'final_shift', 'final_joint',
                         'fingers_touching'])
        for env in range(num_envs):
            g = int(grasp_idx[env])
            o = int(episodes['object_index'][g])
            writer.writerow([g, args.gripper, o, episodes['objects'][o]['object_id'], int(episodes['obs_index'][g]),
                             int(episodes['label'][g]), int(success[env]), int(collided[env]), int(fits[env]),
                             first_fail[env], f'{final_shift[env]:.4f}', f'{gap[env]:.4f}',
                             int(touching[env])])
    print(f'chunk {args.chunk}: {num_envs} grasps, success rate {success.mean():.3f} -> {args.out}')


if __name__ == '__main__':
    main()
