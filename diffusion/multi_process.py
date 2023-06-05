from multiprocessing import Process

import gym_obstacle_avoidance

import gym
import numpy as np

import random
import torch


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def eval_model(model, n_trajectories, obs_mean, obs_std, action_mean, action_std,
               robot_c_pos, mode_encoding, successes, pid):
    env = gym.make('obsavoid-v0')
    env.start()

    random.seed(pid)
    torch.manual_seed(pid)
    np.random.seed(pid)
    for i in range(n_trajectories):
        print(f'Rollout {i}')
        obs = env.reset()

        pred_action = env.robot_state()
        fixed_z = pred_action[2:]
        done = False
        c_pos = [env.robot.current_c_pos]
        while not done:
            obs = np.concatenate((pred_action[:2], obs))
            offset = obs[:2].copy()
            obs = (obs - obs_mean) / obs_std
            # try:  # torch
            #     obs = torch.from_numpy(obs.reshape(1, len(obs)).astype(np.float32))
            #     pred_action = to_numpy(model(obs).reshape(
            #         -1, )) * action_std + action_mean + offset  # obs[:, :2].reshape(-1,).cpu().detach().numpy()
            # except:  # numpy
            obs = obs.reshape(1, len(obs))
            pred_action = model.sample_mean(obs).reshape(-1, ) * action_std + action_mean + offset

            pred_action = np.concatenate((pred_action, fixed_z, [0, 1, 0, 0]), axis=0)

            obs, reward, done, info = env.step(pred_action)

            c_pos.append(env.robot.current_c_pos)
            try:
                for _ in range(35):
                    env.scene.next_step()
            except:
                done = True

        # c_pos = torch.from_numpy(np.array(c_pos))[:, :2]
        c_pos = torch.tensor(np.array(c_pos))[:, :2]
        robot_c_pos[pid * n_trajectories + i, :c_pos.shape[0], :] = c_pos
        mode_encoding[pid * n_trajectories + i, :] = torch.tensor(info[0])
        successes[pid * n_trajectories + i] = torch.tensor(info[1])


def multiproc_sim(model, n_trajectories, obs_mean, obs_std, action_mean, action_std, n_cores=15):
    assert n_trajectories % n_cores == 0

    robot_c_pos = torch.zeros([n_trajectories, 250 + 1, 2]).share_memory_()
    mode_encoding = torch.zeros([n_trajectories, 9]).share_memory_()
    successes = torch.zeros(n_trajectories).share_memory_()

    p_list = []
    if n_cores > 1:
        for i in range(n_cores):
            p = Process(
                target=eval_model,
                kwargs={
                    "model": model,
                    "n_trajectories": n_trajectories // n_cores,
                    "obs_mean": obs_mean,
                    "obs_std": obs_std,
                    "action_mean": action_mean,
                    "action_std": action_std,
                    "robot_c_pos": robot_c_pos,
                    "mode_encoding": mode_encoding,
                    "successes": successes,
                    "pid": i,
                },
            )
            print("Start {}".format(i))
            p.start()
            p_list.append(p)
        [p.join() for p in p_list]
    else:
        eval_model(model, n_trajectories, obs_mean, obs_std, action_mean, action_std, robot_c_pos, mode_encoding,
                   successes, 0)

    entropy = gym.make('obsavoid-v0').mode_decoding(mode_encoding[successes == 1].numpy())[1]

    return successes, entropy, robot_c_pos, mode_encoding