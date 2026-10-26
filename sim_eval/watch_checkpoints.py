"""
Watches the Hugging Face checkpoints of a training run and evaluates every new one in simulation: export the
validation predictions, run both grippers, log sim/success_franka and sim/success_so101 to wandb.
Meant to run next to the training (e.g. on the RTX machine while Colab trains); stop it with Ctrl+C.

  python -m sim_eval.watch_checkpoints --repo Thetatec/dignet-flow-pn2-20m --run <wandb run name> \
      --object-root ~/objaverse --so101-dir ~/robot/SO-ARM100/Simulation/SO101 \
      --wandb-run kevin-jandl/<project>/<run_id> --sim-python ~/robot/IsaacLab/env_isaaclab/bin/python

The training uploads <run dir>/last/training_state.pth every checkpoint_every_n_steps (runs/<run>/last/ in the
repo). A checkpoint is processed once; finished steps are listed in <out-dir>/done.txt.
"""
import argparse
import os
import subprocess
import sys
import time

from sim_eval.episodes import load_episodes
from sim_eval.evaluate import evaluate
from sim_eval.report import log_to_wandb

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def checkpoint_version(api, repo_id, path):
    """ blob id of the file in the repo, None if it does not exist yet """
    infos = api.get_paths_info(repo_id, [path])
    return infos[0].blob_id if infos else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo', required=True, help='HF model repo of the checkpoints')
    parser.add_argument('--run', required=True, help='run name, the checkpoints are in runs/<run>/ of the repo')
    parser.add_argument('--weights', choices=('last', 'best'), default='last')
    parser.add_argument('--object-root', required=True)
    parser.add_argument('--so101-dir', default=os.environ.get('SO101_DIR', ''))
    parser.add_argument('--grippers', nargs='+', default=['franka', 'so101'])
    parser.add_argument('--out-dir', default='sim_results')
    parser.add_argument('--split', default='validation')
    parser.add_argument('--max-objects', type=int, default=64)
    parser.add_argument('--grasps-per-object', type=int, default=32)
    parser.add_argument('--objects-per-chunk', type=int, default=16)
    parser.add_argument('--export-python', default=sys.executable, help='interpreter of the DIGNet environment')
    parser.add_argument('--sim-python', default=sys.executable, help='interpreter of the Isaac Lab environment')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--wandb-run', default=None, help='entity/project/run_id of the training run')
    parser.add_argument('--interval', type=float, default=600.0, help='seconds between checks')
    parser.add_argument('--once', action='store_true', help='process the current checkpoint and exit')
    args, extra = parser.parse_known_args(argv)
    extra = [a for a in extra if a != '--']

    from huggingface_hub import HfApi
    api = HfApi()
    path = f"runs/{args.run}/{'last/training_state.pth' if args.weights == 'last' else 'model_state_dict.pth'}"
    run_out = os.path.join(args.out_dir, args.run)
    os.makedirs(run_out, exist_ok=True)
    done_file = os.path.join(run_out, 'done.txt')
    done = set(open(done_file).read().split()) if os.path.exists(done_file) else set()
    while True:
        version = checkpoint_version(api, args.repo, path)
        if version and version not in done:
            work = os.path.join(run_out, version[:12])
            os.makedirs(work, exist_ok=True)
            episodes = os.path.join(work, 'episodes.npz')
            if not os.path.exists(episodes):
                cmd = [args.export_python, '-m', 'sim_eval.export_grasps', 'model', '--run',
                       f'{args.repo}:runs/{args.run}', '--weights', args.weights, '--split', args.split,
                       '--max-objects', str(args.max_objects), '--grasps-per-object', str(args.grasps_per_object),
                       '--device', args.device, '--out', episodes]
                print(' '.join(cmd), flush=True)
                subprocess.run(cmd, cwd=REPO, check=True)
            step = load_episodes(episodes)['meta'].get('step') or 0
            summary, results = evaluate(episodes, args.grippers, work, args.object_root, args.objects_per_chunk,
                                        args.sim_python, args.so101_dir, extra)
            print(f'step {step}:', {g: round(s['success'], 3) for g, s in summary.items()}, flush=True)
            if args.wandb_run:
                log_to_wandb(summary, step, run_path=args.wandb_run, results_csv=results)
            done.add(version)
            with open(done_file, 'a') as f:
                f.write(version + '\n')
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
