"""
Episode file: the grasps to execute, one npz per checkpoint (or per dataset export).

  grasps       [M, 4, 4] float32  gripper base link pose in the object (mesh) frame, Panda convention
  object_index [M] int32          row in the object table
  label        [M] int8           dataset success label, -1 for model predictions
  obs_index    [M] int32          observation (dataset row) the grasp was predicted from
  objects      json: [{object_id, mesh_file, scale, source}]
  observations json: [{sample_id, cloud_type, object_index}]
  meta         json: checkpoint, step, split, ...

Every object has the same number of grasps (grasps_per_object), so one simulation chunk with n objects has
n * grasps_per_object environments in contiguous blocks (see run_newton.py).
"""
import json
import os

import numpy as np


def save_episodes(path, grasps, object_index, label, obs_index, objects, observations, meta):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, grasps=np.asarray(grasps, np.float32), object_index=np.asarray(object_index, np.int32),
                        label=np.asarray(label, np.int8), obs_index=np.asarray(obs_index, np.int32),
                        objects=json.dumps(objects), observations=json.dumps(observations), meta=json.dumps(meta))


def load_episodes(path):
    data = np.load(path)
    return {'grasps': data['grasps'].astype(np.float64), 'object_index': data['object_index'],
            'label': data['label'], 'obs_index': data['obs_index'], 'objects': json.loads(str(data['objects'])),
            'observations': json.loads(str(data['observations'])), 'meta': json.loads(str(data['meta']))}


def cloud_to_mesh_frame(grasps, camera_pose, centre=None, tcp_depth=None):
    """
    Grasps in the frame of a dataset cloud -> mesh frame. Model predictions are finger centre poses in the
    centred cloud frame (pass centre, the mean of the cloud the model saw, and tcp_depth); dataset grasps are base
    link poses in the cloud frame (no centre, no tcp_depth).
    """
    grasps = np.asarray(grasps, dtype=np.float64).copy()
    if centre is not None:
        grasps[:, :3, 3] += np.asarray(centre, dtype=np.float64)
    if tcp_depth is not None:
        back = np.eye(4)
        back[2, 3] = -tcp_depth
        grasps = grasps @ back
    return np.asarray(camera_pose, dtype=np.float64) @ grasps


def balanced_choice(n_available, k, rng):
    """ exactly k indices out of n_available (with replacement if there are fewer) """
    return rng.choice(n_available, k, replace=n_available < k)


class EpisodeBuilder:
    """ collects grasps per observation, one object table entry per (mesh_file, scale) """

    def __init__(self, grasps_per_object):
        self.k = grasps_per_object
        self.objects, self.observations, self._object_ids = [], [], {}
        self.grasps, self.object_index, self.label, self.obs_index = [], [], [], []

    def object_for(self, row):
        key = (row['mesh_file'], round(float(row['scale']), 6))
        if key not in self._object_ids:
            self._object_ids[key] = len(self.objects)
            self.objects.append({'object_id': str(row['object_id']), 'mesh_file': str(row['mesh_file']),
                                 'scale': float(row['scale']), 'source': str(row['source'])})
        return self._object_ids[key]

    def add(self, row, grasps_mesh, labels):
        obj = self.object_for(row)
        obs = len(self.observations)
        self.observations.append({'sample_id': str(row['sample_id']), 'cloud_type': str(row['cloud_type']),
                                  'object_index': obj})
        self.grasps.append(np.asarray(grasps_mesh))
        self.object_index.append(np.full(len(grasps_mesh), obj))
        self.label.append(np.asarray(labels))
        self.obs_index.append(np.full(len(grasps_mesh), obs))

    def finish(self, rng):
        """ exactly grasps_per_object grasps per object, drawn evenly from its observations """
        grasps, object_index = np.concatenate(self.grasps), np.concatenate(self.object_index)
        label, obs_index = np.concatenate(self.label), np.concatenate(self.obs_index)
        keep = []
        for obj in range(len(self.objects)):
            idx = np.flatnonzero(object_index == obj)
            keep.append(idx[np.sort(balanced_choice(len(idx), self.k, rng))])
        keep = np.concatenate(keep)
        return grasps[keep], object_index[keep], label[keep], obs_index[keep]

    def save(self, path, meta, rng):
        grasps, object_index, label, obs_index = self.finish(rng)
        meta = dict(meta, grasps_per_object=self.k)
        save_episodes(path, grasps, object_index, label, obs_index, self.objects, self.observations, meta)
        return path
