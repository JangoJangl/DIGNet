import torch
import numpy as np
from helper.transform import wrist_to_finger_center
import os




def graspnet_loss(pred, target, obj):
    """
    Computes the graspnet loss

    Arguments:
        pred {np.array} -- predicted grasp 4x4 matrix
        target {np.array} -- target grasp 4x4 matrix

    Returns:
        [np.array] -- graspnet loss
    """

    # create transformation matrix of prediction and target
    # t_pred = p3d.transforms.Transform3d(pred)
    # t_target = p3d.transforms.Transform3d(target)

    #  load graspnet control points
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    control_points = torch.from_numpy(np.load(os.path.join(ROOT_DIR, 'gripper_control_points/panda.npy'))).double().cuda()

    w2f = torch.from_numpy(wrist_to_finger_center()).double().cuda()  # translate COS to finger center <--- Bullshit???
    T_pred = torch.matmul(pred, w2f)
    T_target = torch.matmul(target, w2f)

    points_pred = torch.empty((pred.shape[0], 0, control_points.shape[1])).double().cuda()
    points_target = torch.empty((pred.shape[0], 0, control_points.shape[1])).double().cuda()


    #  transform graspnet control points with prediction/target transformation
    for c in control_points:
        points_pred = torch.cat((points_pred, torch.matmul(T_pred, c).unsqueeze(1)), dim=1)
        points_target = torch.cat((points_target, torch.matmul(T_target, c).unsqueeze(1)), dim=1)

    #  visualize control points for one grasp
    # Visualizer.visualize_cp_tar_pred(obj, target, pred, points_target, points_pred)

    #  compute loss with cumulative distance between transformed control points
    #  tensors have to be of shape (batch, control_points_size, 1, 4) last value is always 1, therefore it can be ignored
    loss = torch.cdist(points_pred.unsqueeze(-2), points_target.unsqueeze(-2), p=2).squeeze()

    # return mean over all control points distances
    return loss.sum(dim=-1)/control_points.shape[0]


def chamfer_loss(pred, target):
    """
    Computes the chamfer loss

    Arguments:
        pred {np.array} -- predicted grasp 4x4 matrix
        target {np.array} -- target grasp 4x4 matrix

    Returns:
        [np.array] -- chamfer loss
    """

    import pytorch3d as p3d
    # trnasform grasp representations into 4x4 matrices

    # create transformation matrix of prediction and target
    t_pred = p3d.transforms.Transform3d(pred)
    t_target = p3d.transforms.Transform3d(target)

    #  load graspnet control points
    control_points = np.load('control_points.pt')
    #  transform graspnet control points with prediction transformation
    points_pred = t_pred.transform_points(control_points)
    #  transform graspnet control points with target transformation
    points_target = t_target.transform_points(control_points)
    #  compute loss with cumulative distance between transformed control points
    #  tensors have to be of shape (control_points_size, 1, 3)
    loss = p3d.loss.chamfer_distance(points_pred, points_target, p=2)

    return loss


def _load_control_points(device, dtype):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    control_points = np.load(os.path.join(ROOT_DIR, 'gripper_control_points/panda.npy'))
    return torch.as_tensor(control_points, device=device, dtype=dtype)


def transform_control_points(grasps, control_points=None):
    """
    Transforms the gripper control points with a batch of grasp poses

    Arguments:
        grasps {torch.Tensor} -- [N, 4, 4] grasp poses
        control_points {torch.Tensor} -- [P, 4] homogeneous control points, panda fingers by default

    Returns:
        [torch.Tensor] -- [N, P, 3] control points in the object frame
    """
    if control_points is None:
        control_points = _load_control_points(grasps.device, grasps.dtype)
    return torch.einsum('nij,pj->npi', grasps, control_points)[..., :3]


def pairwise_gripper_distance(x, y, symmetric=False, chunk_size=1024):
    """
    Mean euclidean distance between corresponding control points for every pair of grasps

    Arguments:
        x {torch.Tensor} -- [N, P, 3] transformed control points
        y {torch.Tensor} -- [M, P, 3] transformed control points
        symmetric {bool} -- a parallel jaw gripper rotated by 180 degree around its approach axis is the
                            same grasp, if true the fingers of y are swapped and the smaller distance is used

    Returns:
        [torch.Tensor] -- [N, M] distance matrix
    """
    if symmetric:
        # control points are stored as (left, right) finger pairs, swapping the pairs flips the gripper
        y_flipped = y.view(y.shape[0], -1, 2, 3).flip(dims=[2]).view_as(y)
    rows = []
    for start in range(0, x.shape[0], chunk_size):
        x_chunk = x[start:start + chunk_size]
        d = torch.linalg.vector_norm(x_chunk[:, None] - y[None], dim=-1).mean(dim=-1)
        if symmetric:
            d_flipped = torch.linalg.vector_norm(x_chunk[:, None] - y_flipped[None], dim=-1).mean(dim=-1)
            d = torch.minimum(d, d_flipped)
        rows.append(d)
    return torch.cat(rows, dim=0)


@torch.no_grad()
def trafo_chamfer_distance(x, y, direction='bi', coverage=False, symmetric=False):
    """Chamfer distance between two sets of grasp poses, measured on the gripper control points

    Parameters
    ----------
    x: torch.Tensor [n_grasps_x, 4, 4]
        first set of grasps (predictions)
    y: torch.Tensor [n_grasps_y, 4, 4]
        second set of grasps (ground truth)
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every grasp in y to x
            'x_to_y':  computes average minimal distance from every grasp in x to y
            'bi': compute both
    coverage: bool
        additionally return the per grasp minimal distances of the chosen direction
    symmetric: bool
        treat grasps that only differ by a 180 degree rotation around the approach axis as identical
    Returns
    -------
    chamfer_dist: float
        mean over the per grasp minimal distances, the distance between two grasps is the mean
        euclidean distance between their corresponding control points
    """
    y = y.to(device=x.device, dtype=x.dtype)
    points_x = transform_control_points(x)
    points_y = transform_control_points(y)
    dist_matrix = pairwise_gripper_distance(points_x, points_y, symmetric=symmetric)  # [N_x, N_y]

    if direction == 'y_to_x':
        dist = dist_matrix.min(dim=0).values
        chamfer_dist = dist.mean().item()
    elif direction == 'x_to_y':
        dist = dist_matrix.min(dim=1).values
        chamfer_dist = dist.mean().item()
    elif direction == 'bi':
        dist = dist_matrix.min(dim=0).values
        chamfer_dist = dist.mean().item() + dist_matrix.min(dim=1).values.mean().item()
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_to_x\', \'x_to_y\', \'bi\'")
    if coverage:
        return chamfer_dist, dist.cpu().numpy()
    else:
        return chamfer_dist


def gripper_dist(x, y):

    x = x.reshape((-1, 3))
    y = y.reshape((-1, 3))
    return np.mean(np.linalg.norm((x-y), axis=1))



if __name__ == "__main__":
    pass
