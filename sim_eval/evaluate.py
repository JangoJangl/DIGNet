"""
Runs all chunks of an episode file for the chosen grippers (one fresh Isaac Lab process per chunk, see
run_newton.py) and writes <out-dir>/results.csv and <out-dir>/summary.json.

  python -m sim_eval.evaluate --episodes ep.npz --grippers franka so101 --object-root ~/objaverse \
      --so101-dir ~/robot/SO-ARM100/Simulation/SO101 --out-dir sim_results/step_0040000 \
      [--wandb-run entity/project/run_id --step 40000] [-- --viz newton --collision sdf]

Arguments after '--' go to run_newton.py. --python selects the interpreter of the Isaac Lab environment if this
script runs in another one.
"""
import argparse
import csv
import glob
import os
import subprocess
import sys

from sim_eval.episodes import load_episodes
from sim_eval.report import log_to_wandb, read_results, summarize, write_summary

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_chunks(episodes_path, gripper, out_dir, object_root, objects_per_chunk, python=sys.executable, so101_dir='',
               extra=(), skip_done=True):
    episodes = load_episodes(episodes_path)
    n_chunks = -(-len(episodes['objects']) // objects_per_chunk)
    chunk_dir = os.path.join(out_dir, 'chunks')
    os.makedirs(chunk_dir, exist_ok=True)
    for chunk in range(n_chunks):
        out = os.path.join(chunk_dir, f'{gripper}_{chunk:04d}.csv')
        if skip_done and os.path.exists(out):
            continue
        cmd = [python, '-m', 'sim_eval.run_newton', '--episodes', episodes_path, '--gripper', gripper,
               '--object-root', object_root, '--chunk', str(chunk), '--objects-per-chunk', str(objects_per_chunk),
               '--out', out, *(['--so101-dir', so101_dir] if so101_dir else []), *extra]
        print(' '.join(cmd), flush=True)
        result = subprocess.run(cmd, cwd=REPO)
        if result.returncode != 0:
            print(f'chunk {chunk} ({gripper}) failed with exit code {result.returncode}', flush=True)
    return sorted(glob.glob(os.path.join(chunk_dir, f'{gripper}_*.csv')))


def merge(csv_files, path):
    rows = [r for f in csv_files for r in read_results(f)]
    if not rows:
        raise RuntimeError('no simulation results')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def evaluate(episodes_path, grippers, out_dir, object_root, objects_per_chunk=16, python=sys.executable,
             so101_dir='', extra=()):
    files = []
    for gripper in grippers:
        files += run_chunks(episodes_path, gripper, out_dir, object_root, objects_per_chunk, python, so101_dir, extra)
    results = os.path.join(out_dir, 'results.csv')
    summary = summarize(merge(files, results))
    write_summary(summary, os.path.join(out_dir, 'summary.json'), load_episodes(episodes_path)['meta'])
    return summary, results


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    extra = argv[argv.index('--') + 1:] if '--' in argv else []
    argv = argv[:argv.index('--')] if '--' in argv else argv
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--episodes', required=True)
    parser.add_argument('--grippers', nargs='+', default=['franka', 'so101'])
    parser.add_argument('--object-root', required=True)
    parser.add_argument('--so101-dir', default=os.environ.get('SO101_DIR', ''))
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--objects-per-chunk', type=int, default=16)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--wandb-run', default=None, help='entity/project/run_id of the training run')
    parser.add_argument('--step', type=int, default=None)
    args = parser.parse_args(argv)
    summary, results = evaluate(args.episodes, args.grippers, args.out_dir, args.object_root, args.objects_per_chunk,
                                args.python, args.so101_dir, extra)
    for gripper, stats in summary.items():
        print(gripper, {k: v for k, v in stats.items() if k != 'failed_phase'})
    if args.wandb_run:
        step = args.step if args.step is not None else load_episodes(args.episodes)['meta'].get('step') or 0
        log_to_wandb(summary, step, run_path=args.wandb_run, results_csv=results)


if __name__ == '__main__':
    main()
