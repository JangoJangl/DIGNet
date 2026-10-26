"""
Grasp validation in simulation (Isaac Lab 3 with the Newton physics backend, no Isaac Sim needed).

export_grasps.py  DIGNet predictions (or dataset grasps) of a split -> episode file (grasps in the mesh frame)
run_newton.py     executes one chunk of an episode file with a flying Franka or SO-101 gripper (Isaac Lab)
evaluate.py       runs all chunks for both grippers and writes results.csv / summary.json
watch_checkpoints.py  polls the Hugging Face checkpoints of a run and logs sim/success_* to its wandb run
"""
