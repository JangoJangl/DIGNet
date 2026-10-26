"""
Summaries of the simulation results (results.csv of evaluate.py) and logging to wandb.

Metrics per gripper (sim/<metric>_<gripper>):
  success          share of grasps with the object still in the gripper (the number to compare checkpoints)
  success_fits     the same over grasps whose object fits the gripper opening (SO-101: narrower than ~7 cm)
  collision        share of grasps where the open gripper already hits the object
  success_pos / success_neg   dataset exports only: simulated success of grasps labelled 1 / 0, i.e. how well the
                   simulation agrees with the labels of the dataset
"""
import csv
import json

import numpy as np


def read_results(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def summarize(rows):
    out = {}
    for gripper in sorted({r['gripper'] for r in rows}):
        g = [r for r in rows if r['gripper'] == gripper]
        success = np.array([int(r['success']) for r in g])
        fits = np.array([int(r['fits']) for r in g], dtype=bool)
        label = np.array([int(r['label']) for r in g])
        objects = np.array([int(r['object_index']) for r in g])
        per_object = [success[objects == o].mean() for o in np.unique(objects)]
        stats = {
            'success': float(success.mean()),
            'success_fits': float(success[fits].mean()) if fits.any() else float('nan'),
            'fits': float(fits.mean()),
            'collision': float(np.mean([int(r['initial_collision']) for r in g])),
            'object_success': float(np.mean(per_object)),
            'num_grasps': int(len(g)),
            'num_objects': int(len(per_object)),
        }
        if (label >= 0).any():
            for name, value in (('pos', 1), ('neg', 0)):
                sel = label == value
                stats[f'success_{name}'] = float(success[sel].mean()) if sel.any() else float('nan')
            sel = label >= 0
            stats['label_agreement'] = float(np.mean(success[sel] == label[sel]))
        phases = [r['first_failed_phase'] for r in g if r['first_failed_phase']]
        stats['failed_phase'] = {p: phases.count(p) for p in sorted(set(phases))}
        out[gripper] = stats
    return out


def write_summary(summary, path, meta=None):
    with open(path, 'w') as f:
        json.dump({'meta': meta or {}, 'grippers': summary}, f, indent=2)


def wandb_metrics(summary):
    """ flat dict sim/<metric>_<gripper> for wandb.log """
    metrics = {}
    for gripper, stats in summary.items():
        for key, value in stats.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                metrics[f'sim/{key}_{gripper}'] = value
    return metrics


def log_to_wandb(summary, step, run_path=None, project=None, entity=None, results_csv=None):
    """
    Logs the metrics against the training step on the axis sim/step.
    run_path 'entity/project/run_id' of the training run: a finished run is resumed, so the curves sit in the
    training run itself. While the training still runs a second writer would conflict, so a separate run
    '<name>-sim' in the same project and group is used instead.
    """
    import wandb
    if run_path:
        entity, project, run_id = run_path.split('/')
        train = wandb.Api().run(run_path)
        if train.state in ('finished', 'crashed', 'failed', 'killed'):
            run = wandb.init(entity=entity, project=project, id=run_id, resume='allow')
        else:
            run = wandb.init(entity=entity, project=project, id=f'{run_id}-sim', name=f'{train.name}-sim',
                             group=train.group or train.name, job_type='sim_eval', resume='allow')
    else:
        run = wandb.init(entity=entity, project=project, job_type='sim_eval')
    run.define_metric('sim/step')
    run.define_metric('sim/*', step_metric='sim/step')
    metrics = wandb_metrics(summary)
    run.log({**metrics, 'sim/step': step})
    if results_csv:
        rows = read_results(results_csv)
        run.log({'sim/results': wandb.Table(columns=list(rows[0].keys()), data=[list(r.values()) for r in rows]),
                 'sim/step': step})
    run.finish()
    return metrics
