# Grasp validation in Isaac Lab (Newton, no Isaac Sim)

Executes DIGNet grasps on the validation objects with a flying **Franka Hand** and a flying **SO-101 (LeRobot)**
gripper and records per grasp `success = 1` if the object is still in the gripper after closing and a lift/shake
load, else `0`. Background and design: `knowledgebase/state/sim-grasp-validation.md`.

**Status:** the geometry, the grasp transfer, the export and the reports are unit tested on CPU
(`tests/test_sim_eval.py`). `run_newton.py` is written against the Isaac Lab 3.0.0 API but has not run on a GPU
yet. Start with the calibration run below.

## 1. Install Isaac Lab 3 kit-less (Newton backend, Python 3.12)

Isaac Lab 3.0 ships Newton as the default physics backend and installs without Isaac Sim. Headless Newton needs a
CUDA GPU from Turing on (RTX 3060 is fine) and driver >= 580.65; the Isaac Sim VRAM / RTX requirements do not apply.

```bash
git clone -b v3.0.0 https://github.com/isaac-sim/IsaacLab.git ~/robot/IsaacLab && cd ~/robot/IsaacLab
uv venv --python 3.12 --seed env_isaaclab && source env_isaaclab/bin/activate
uv pip install cmake                   # otherwise isaaclab.sh wants 'sudo apt-get install cmake'
./isaaclab.sh -i                       # default install: Newton, no Isaac Sim
# URDF import without Isaac Sim: the 'importers' extra, with the overrides so the base install is not downgraded
uv pip install "isaacsim-asset-isolated==6.2.0.0" "tinyobjloader==2.0.0rc13" \
    --overrides tools/wheel_builder/uv-overrides.txt --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match
uv pip install trimesh coacd fast-simplification rtree huggingface_hub wandb imageio imageio-ffmpeg pytest
```

Do not install `usd-core`: `usd-exchange` (pulled in by Isaac Lab) ships its own `pxr` and both overwrite each other's
files (`import usdex.core` then fails with "initialization failed"; fix: `uv pip uninstall usd-core` and
`uv pip install --reinstall usd-exchange==3.0.1`). `scripts/tutorials/00_sim/create_empty.py` is no check here, it uses
PhysX and asks for Isaac Sim; run a small chunk of `run_newton.py` instead. `run_newton.py` puts the importer's PhysX
schemas on `PXR_PLUGINPATH_NAME` itself (without it the URDF import fails with "Cannot find a valid schema for
PhysxJointAPI"). On a crash while loading the stage (OpenUSD < 26.05 with many colliders per body) set
`PXR_WORK_THREAD_LIMIT=1`.

Also needed: the Objaverse meshes of the GraspGen objects (`mesh_file` in the dataset is relative to this root,
same as for `dataset_tools`) and the SO-101 model `SO-ARM100/Simulation/SO101` (URDF + STL).

## 2. Calibrate with dataset grasps (do this first)

Dataset grasps carry success labels from GraspGen's own simulation, so the simulated success of label-1 and label-0
grasps shows whether contact, friction and effort are set sensibly:

```bash
python -m sim_eval.export_grasps dataset --paths Thetatec/graspgen-unified-full --split validation \
    --max-objects 16 --grasps-per-object 32 --out sim_results/gt/episodes.npz
python -m sim_eval.evaluate --episodes sim_results/gt/episodes.npz --grippers franka \
    --object-root ~/objaverse --out-dir sim_results/gt --python ~/robot/IsaacLab/env_isaaclab/bin/python
```

A video of a few grasps (headless, Newton GL via EGL, half speed, collision shapes are shown):

```bash
python -m sim_eval.export_grasps dataset --paths Thetatec/graspgen-unified-full --split validation \
    --max-objects 2 --grasps-per-object 8 --out sim_results/video/episodes.npz
~/robot/IsaacLab/env_isaaclab/bin/python -m sim_eval.run_newton --episodes sim_results/video/episodes.npz \
    --gripper franka --object-root ~/objaverse --env-spacing 0.5 --out sim_results/video/franka.csv \
    --video sim_results/video/franka.mp4
```

`summary.json` then has `success_pos` (should be high) and `success_neg` (should be low) for the Franka.
Things to tune if they are off: `--friction`, `--effort`, `--load-g`, `--collision coacd|sdf|convex_hull`,
`--dt`, `--substeps`, `--contact-ke/--contact-kd`, `--cone`, `--impratio`. With `use_mujoco_contacts=False` MJWarp takes
the shape `ke`/`kd` as contact stiffness/damping in N/m; Newton's default `ke = 2500` lets the fingers sink in by
centimetres at 70 N, hence `--contact-ke 1e5`. A pyramidal cone or a low `impratio` lets the objects creep out along
the pads (defaults: elliptic, 100). Object mass: density (400 kg/m³) x solid volume of the voxel-filled mesh, at least a 3 mm shell of the surface
(open/thin meshes), clipped to 0.02..0.5 kg; `--mass` sets one mass for all objects. The Franka closing target moves
at 0.1 m/s to beyond closed (stiffness 3000 N/m), so the grip force reaches 70 N for thin objects too (with a
position target at closed the force was stiffness x remaining gap, ~5 N on a 5 mm plate; stiffness >= 1e4 is
unstable with the 15 g fingers). `--mesh-prep graspgen` simulates watertight, decimated meshes like the ones the
GraspGen labels were made on (voxelization + marching cubes + fast-simplification instead of the Manifold tools).

Calibration, Franka, 8 validation objects x 32 dataset grasps (success of label-1 / label-0 grasps, agreement):

| variant | label 1 | label 0 | agreement |
|---|---|---|---|
| hull mass (<= 1 kg), stiffness 2000, target jumps to closed | 44 % | 23 % | 0.60 |
| new mass, stiffness 2000 | 53 % | 26 % | 0.63 |
| uniform 0.2 kg, stiffness 2000 | 47 % | 22 % | 0.62 |
| **new mass, stiffness 3000, ramp 0.1 m/s (default)** | 51-53 % | 20 % | 0.65-0.66 |
| uniform 0.2 kg, stiffness 3000, ramp | 41 % | 13 % | 0.63 |
| new mass, stiffness 2000, mesh-prep graspgen + sdf | 54 % | 30 % | 0.62 |
| new mass, stiffness 3000, ramp, mesh-prep graspgen + sdf | 47 % | 30 % | 0.58 |
| stiffness 1e5, ramp (unstable) | 0 % | 0 % | 0.48 |

SO-101 with the defaults: 6 % / 8 %, 43 % of the grasps collide at the start (wrist and servo body).

## 3. Evaluate a checkpoint

```bash
# DIGNet environment: predictions on the validation split (data paths from the run config)
python -m sim_eval.export_grasps model --run Thetatec/dignet-flow-pn2-20m:runs/<run> --weights last \
    --split validation --max-objects 64 --grasps-per-object 32 --out sim_results/<run>/episodes.npz
# Isaac Lab environment: both grippers
python -m sim_eval.evaluate --episodes sim_results/<run>/episodes.npz --grippers franka so101 \
    --object-root ~/objaverse --so101-dir ~/robot/SO-ARM100/Simulation/SO101 --out-dir sim_results/<run>
```

Or let it follow the training (every new `runs/<run>/last/training_state.pth` on the Hub):

```bash
python -m sim_eval.watch_checkpoints --repo Thetatec/dignet-flow-pn2-20m --run <run> \
    --object-root ~/objaverse --so101-dir ~/robot/SO-ARM100/Simulation/SO101 \
    --export-python ~/envs/dignet/bin/python --sim-python ~/robot/IsaacLab/env_isaaclab/bin/python \
    --wandb-run <entity>/<project>/<run_id>
```

wandb gets `sim/success_franka`, `sim/success_so101`, `sim/success_fits_so101`, `sim/collision_*` on the axis
`sim/step` (= training step). While the training run is active the values go to a run `<name>-sim` in the same
group, because two processes must not write the same run.

## Outputs

`results.csv`, one row per grasp and gripper: `success` (1/0), `initial_collision`, `fits` (object narrower than the
opening), `first_failed_phase` (`load+z` ... or `closed_empty`), `final_shift` (m), `final_joint`, `label`
(dataset label, -1 for predictions). `summary.json`: rates per gripper.
