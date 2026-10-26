"""
Converts ACRONYM, GraspGen or GraspNet-1Billion into the unified format (dataset_tools/common.py) and writes a
Hugging Face DatasetDict (train / validation / test) to disk.

    # ACRONYM, full and partial clouds
    python -m dataset_tools.build_dataset --source acronym --cloud full    --root /data/acronym --output /data/unified/acronym_full
    python -m dataset_tools.build_dataset --source acronym --cloud partial --root /data/acronym --output /data/unified/acronym_partial --views 4
    # GraspGen (HF repo nvidia/PhysicalAI-Robotics-GraspGen + Objaverse meshes from its download_objaverse.py)
    python -m dataset_tools.build_dataset --source graspgen --cloud full --root /data/PhysicalAI-Robotics-GraspGen \
        --object-root /data/objaverse_graspgen --gripper franka_panda --output /data/unified/graspgen_full
    # GraspNet-1Billion: real scenes are partial; full clouds come from the 88 object models
    python -m dataset_tools.build_dataset --source graspnet1b --cloud partial --root /data/graspnet --camera kinect --output /data/unified/graspnet_partial

Run from the repository root. --preview N saves a figure with N samples next to the dataset,
--push-to-hub REPO uploads it (needs HF_TOKEN or huggingface-cli login).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset_tools import sources  # noqa: E402
from dataset_tools.common import features, seed_of  # noqa: E402
from dataset_tools.mesh_views import full_row, partial_rows  # noqa: E402

CLOUD_TYPES = {'acronym': ('full', 'partial'), 'graspgen': ('full', 'partial'), 'graspnet1b': ('full', 'partial')}


def mesh_rows(records, args):
    from helper.rendering import DepthRenderer
    renderer = DepthRenderer()
    for record in records:
        rng = np.random.default_rng(seed_of(args.seed, record.source, record.object_id))
        if args.cloud == 'full':
            yield full_row(record, args.num_points, args.max_grasps, rng)
        else:
            yield from partial_rows(record, args.views, args.num_points, args.max_grasps, rng, renderer)


def generate(jobs, args):
    """ jobs: list of work items (files, shards or scene ids) for this worker """
    for job in jobs:
        try:
            if args.source == 'acronym':
                yield from mesh_rows([sources.acronym_record(args.root, job)], args)
            elif args.source == 'graspgen':
                split = sources.graspgen_splits(args.root, args.gripper)
                for row in mesh_rows(sources.graspgen_records(job, args.object_root, args.gripper, args.max_objects), args):
                    row['split'] = split.get(row['object_id'], row['split'])
                    yield row
            elif args.cloud == 'full':
                yield from sources.graspnet_full_rows(args.root, [job], args.num_points, args.max_grasps, args.gripper, args.seed)
            else:
                yield from sources.graspnet_partial_rows(args.root, [job], args.camera, range(0, 256, args.ann_stride),
                                                         args.num_points, args.max_grasps, args.gripper, seed=args.seed)
        except Exception as e:  # broken meshes or files are skipped, not fatal for a multi hour conversion
            print(f"skip {job}: {type(e).__name__}: {e}")


def jobs_for(args):
    if args.source == 'acronym':
        return sources.acronym_files(args.root, args.max_objects)
    if args.source == 'graspgen':
        return sources.graspgen_shards(args.root, args.gripper)
    if args.cloud == 'full':
        return list(range(88))[:args.max_objects]
    return [s for s in range(*args.scenes) if os.path.isdir(os.path.join(args.root, 'scenes', f'scene_{s:04d}'))]


def build(args):
    from datasets import Dataset, DatasetDict
    jobs = jobs_for(args)
    print(f"{args.source} / {args.cloud}: {len(jobs)} work items")
    num_proc = args.num_proc if len(jobs) >= args.num_proc > 1 else None
    full = Dataset.from_generator(generate, gen_kwargs={'jobs': jobs, 'args': args}, features=features(), num_proc=num_proc,
                                  cache_dir=os.path.join(args.output, '.cache'))
    splits = {name: full.filter(lambda s, n=name: s == n, input_columns='split') for name in ('train', 'validation', 'test')}
    dataset = DatasetDict({name: d for name, d in splits.items() if len(d)})  # empty splits do not reload
    dataset.save_to_disk(args.output)
    print(dataset)
    return dataset


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', required=True, choices=list(CLOUD_TYPES))
    p.add_argument('--cloud', required=True, choices=['full', 'partial'])
    p.add_argument('--root', required=True, help='dataset root in its original layout')
    p.add_argument('--output', required=True)
    p.add_argument('--object-root', help='GraspGen: folder with the Objaverse meshes and map_uuid_to_path.json')
    p.add_argument('--gripper', default='franka_panda', help='GraspGen: franka_panda or robotiq_2f_140 (suction is not supported)')
    p.add_argument('--max-objects', type=int, default=None)
    p.add_argument('--views', type=int, default=4, help='rendered views per object (partial, mesh sources)')
    p.add_argument('--num-points', type=int, default=None, help='default 2048 full, 1024 partial')
    p.add_argument('--max-grasps', type=int, default=500, help='per object / view, half successful where possible')
    p.add_argument('--camera', default='kinect', choices=['kinect', 'realsense'], help='GraspNet-1Billion camera')
    p.add_argument('--scenes', type=int, nargs=2, default=(0, 190), help='GraspNet-1Billion scene id range')
    p.add_argument('--ann-stride', type=int, default=16, help='GraspNet-1Billion: every n-th of the 256 images per scene')
    p.add_argument('--num-proc', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--preview', type=int, default=0, help='save a figure with this many samples next to the dataset')
    p.add_argument('--push-to-hub', default=None, help='HF dataset repo id, uploads only if given')
    p.add_argument('--public', action='store_true', help='create the HF repo public (default private)')
    args = p.parse_args(argv)
    if args.num_points is None:
        args.num_points = 2048 if args.cloud == 'full' else 1024
    if args.source == 'graspgen' and not args.object_root:
        p.error('--object-root is required for GraspGen')
    if args.gripper not in ('franka_panda', 'robotiq_2f_140'):
        p.error('only antipodal grippers franka_panda and robotiq_2f_140 are supported')
    return args


def main(argv=None):
    args = parse_args(argv)
    dataset = build(args)
    if args.preview:
        from dataset_tools.preview import save_preview
        print('preview:', save_preview(dataset, os.path.join(args.output, 'preview.png'), args.preview))
    if args.push_to_hub:
        dataset.push_to_hub(args.push_to_hub, private=not args.public, token=os.environ.get('HF_TOKEN'))
        print(f"uploaded to https://huggingface.co/datasets/{args.push_to_hub}")


if __name__ == '__main__':
    main()
