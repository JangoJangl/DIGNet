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
./isaaclab.sh -i                       # default install: Newton, no Isaac Sim
# URDF import without Isaac Sim needs the 'importers' extra, follow the install guide's
# "importers" section (it pins overrides so the base install is not downgraded)
uv pip install trimesh coacd fast-simplification rtree huggingface_hub wandb
```

Check: `python scripts/tutorials/00_sim/create_empty.py` runs without Isaac Sim. On a crash while loading the stage
(OpenUSD < 26.05 with many colliders per body) set `PXR_WORK_THREAD_LIMIT=1`.

Also needed: the Objaverse meshes of the GraspGen objects (`mesh_file` in the dataset is relative to this root,
same as for `dataset_tools`) and the SO-101 model `SO-ARM100/Simulation/SO101` (URDF + STL).

## 2. Calibrate with dataset grasps (do this first)

Dataset grasps carry success labels from GraspGen's own simulation, so the simulated success of label-1 and label-0
grasps shows whether contact, friction and effort are set sensibly:

```bash
python -m sim_eval.export_grasps dataset --paths Thetatec/graspgen-unified-full --split validation \
    --max-objects 16 --grasps-per-object 32 --out sim_results/gt/episodes.npz
python -m sim_eval.evaluate --episodes sim_results/gt/episodes.npz --grippers franka \
    --object-root ~/objaverse --out-dir sim_results/gt -- --viz newton
```

`summary.json` then has `success_pos` (should be high) and `success_neg` (should be low) for the Franka.
Things to tune if they are off: `--friction`, `--effort`, `--load-g`, `--collision coacd|sdf|convex_hull`,
`--dt`, `--substeps`, `--cone elliptic --impratio 10`. Use `--objects-per-chunk` to fit the GPU memory (6 GB:
start with 8 objects x 32 grasps = 256 environments).

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
