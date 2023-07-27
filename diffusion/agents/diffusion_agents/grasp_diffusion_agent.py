from collections import deque
import os
import logging
from typing import Optional

from omegaconf import DictConfig
import hydra
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import wandb
import einops
from sklearn.neighbors import KernelDensity

from agents.base_agent import BaseAgent
from agents.ema_helper.ema import ExponentialMovingAverage
from agents.scaler.scaler_class import Scaler
import helper.losses as losses
from helper.visualize import Visualizer
from helper.collision import collision_detection, collision_detection_pc_mesh, not_in_gripper_detection_pc_mesh
import torch.nn.functional as F

# A logger for this file
log = logging.getLogger(__name__)


class GraspDiffusionAgent(BaseAgent):


    #TODO: Use local Embedding as well
    def __init__(
            self,
            embedder: DictConfig,
            model: DictConfig,
            optimization: DictConfig,
            device: str,
            max_train_steps: int,
            n_timesteps: int,
            eval_every_n_steps: int,
            use_ema: bool,
            decay: float,
            update_ema_every_n_steps: int,
            window_size: int,
            evaluation: bool = True,
            visualize_paths: list = [],
    ):
        super().__init__(model, embedder, optimization, device, max_train_steps, eval_every_n_steps)
        self.validation_loss = losses.trafo_chamfer_distance # TODO: instantiate loss function by hydra
        self.ema_helper = ExponentialMovingAverage(self.model.get_params(), decay, self.device)
        self.use_ema = use_ema
        self.decay = decay
        self.n_timesteps = n_timesteps
        self.update_ema_every_n_steps = update_ema_every_n_steps
        # here all the parameters required for the GPT variant
        self.window_size = window_size
        self.evaluation = evaluation
        self.visualize_paths = visualize_paths

    # What exactly is scaler doing?
    def set_processor(self, processor):
        """
        Define the scaler from the Workspace class used to scale embedding and output data if necessary
        """
        self.processor = processor

    def set_bounds(self, size_bounds=7):
        """
        Define the bounds for the sampler class
        """

        bounds = np.concatenate([np.ones(size_bounds)]) # Bounds are in between of 1 and -1 because of unit variance scaling
        self.model.max_input = torch.from_numpy(bounds).to(self.device)
        self.model.min_input = torch.from_numpy(-bounds).to(self.device)
        # translation_bound = np.ones(3) * scale_factor
        # rotation_bound = np.ones(6) * rotation_factor
        # bounds = np.concatenate([translation_bound, rotation_bound])
        # self.model.max_input = torch.from_numpy(bounds).to(self.device)
        # self.model.min_input = torch.from_numpy(-bounds).to(self.device)

    def train_agent(
            self,
            train_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader
    ):

        best_test_loss = 1e2
        generator = iter(train_loader)
        print(self.max_train_steps)
        for step in tqdm(range(self.max_train_steps)):
            # run a test batch every n steps
            if not self.steps % self.eval_every_n_steps:
                diversity_loss, accuracy_loss, coverage, gripper_check_loss = self. evaluate_agent(test_loader)
                log.info("Step {}: Mean diversity loss is {}".format(step, diversity_loss))
                log.info("Step {}: Mean accuracy loss is {}".format(step, accuracy_loss))
                if (diversity_loss+accuracy_loss) < best_test_loss:
                    best_test_loss = diversity_loss+accuracy_loss
                    self.store_model_weights(self.working_dir)
                    log.info('New best test loss. Stored weights have been updated!')

            try:
                batch_loss = self.train_step(*next(generator)) #preprocess data + train step
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                generator = iter(train_loader)
                batch_loss = self.train_step(*next(generator))

            # log.info loss every 100 steps
            if not self.steps % 100:
                log.info("Step {}: Mean batch loss {}".format(step, batch_loss))
            wandb.log(
                {
                    "train_loss": batch_loss,
                    "diversity_loss": diversity_loss,
                    "accuracy_loss": accuracy_loss,
                    "chamfer_loss": best_test_loss,
                    "coverage": coverage,
                    "gripper_check_loss": gripper_check_loss,
                }
        )
        self.store_model_weights(self.working_dir)

    def train_step(self, points: torch.Tensor, grasp: torch.Tensor) -> float:

        # Set the model to train mode.
        self.model.train()
        self.embedder.train()
        # Compute the embedding.
        global_embedding = self.embedder(points)
        # stack subbatches to a batch
        embedding = torch.repeat_interleave(global_embedding, grasp.size(dim=1), dim=0)
        grasp = torch.flatten(grasp, start_dim=0, end_dim=1)

        # flatten data
        grasp = torch.flatten(grasp, start_dim=1)  # dim0 is batch size
        embedding = torch.flatten(embedding, start_dim=1)  # dim0 is batch size


        # embedding = self.scaler.scale_input(embedding)
        # grasp = self.scaler.scale_output(grasp)

        # Compute the loss.
        if self.processor.representation == 'rot_6d_representation':
            loss = self.model.loss(grasp, embedding, gram_schmidt=True)
        else:
            loss = self.model.loss(grasp, embedding)
        # Before the backward pass, zero all the network gradients
        self.optimizer.zero_grad()
        # Backward pass: compute gradient of the loss with respect to parameters
        loss.backward()
        # Calling the step function to update the parameters
        self.optimizer.step()
        self.steps += 1
        # update the ema model
        # if self.steps % self.update_ema_every_n_steps == 0:
        #     self.ema_helper.update(self.model.parameters())
        return loss


    def evaluate_agent(self, test_loader, path=None):
        test_div_loss = []
        test_acc_loss = []
        test_gripper_check_loss = []
        test_coverage = []
        for i, tensors in enumerate(test_loader):
            gripper_check = True if i < 6 else False
            points, grasp = tensors
            # (points, grasp), success = test_loader.dataset.get_screwdriver()
            obj = test_loader.dataset.processor.current_obj

            inv_fct = self.processor.postprocess
            div_loss, acc_loss, coverage, gripper_check_loss = self.evaluate(points, grasp, inv_fct, test_loader.dataset.processor.current_obj, i, gripper_check=gripper_check, path=path)

            test_div_loss.append(div_loss)
            test_acc_loss.append(acc_loss)
            test_coverage.append(coverage)
            if gripper_check_loss is not None:
                test_gripper_check_loss.append(gripper_check_loss)
        avrg_test_div_loss = np.mean(test_div_loss)
        avrg_test_acc_loss = np.mean(test_acc_loss)
        avrg_test_coverage = np.mean(test_coverage)
        avrg_test_gripper_check_loss = np.mean(test_gripper_check_loss)
        return avrg_test_div_loss, avrg_test_acc_loss, avrg_test_coverage, avrg_test_gripper_check_loss

    @torch.no_grad()
    def evaluate(self, points: torch.Tensor, grasp: torch.Tensor, inv_fct, obj, i, gripper_check=False, path=None) -> float:

        # scale data if necessarry, otherwise the scaler will return unchanged values
        # embedding = self.scaler.scale_input(embedding)
        # grasp = self.scaler.scale_output(grasp)


        # Set the model to eval mode.
        self.embedder.eval()
        self.model.eval()

        global_embedding = self.embedder(points)
        # stack subbatches to a batch
        global_embedding = torch.repeat_interleave(global_embedding, grasp.size(dim=1), dim=0)
        grasp = torch.flatten(grasp, start_dim=0, end_dim=1)

        # flatten data
        grasp = torch.flatten(grasp, start_dim=1)  # dim0 is batch size
        global_embedding = torch.flatten(global_embedding, start_dim=1)  # dim0 is batch size


        # post = self.processor.postprocess(grasp)
        # Visualizer.render_images(obj, post[:100])

        if self.processor.representation_name == 'rotation_6d_representation':
            model_pred = self.model(global_embedding, gram_schmidt=True)
        else:
            model_pred = self.model(global_embedding)

        # apply inverse function
        model_pred = inv_fct(model_pred)
        grasp = inv_fct(grasp)
        #filtering grasp proposals
        if gripper_check: # or obj.metadata['file_name'] in self.visualize_paths:
            if len(model_pred) < 40:
                mps = model_pred
            else:
                mps = model_pred[:40]
            idx_collsion = collision_detection_pc_mesh(mps, torch.squeeze(points, dim=0).cpu().numpy()[:, :3])
            idx_not_in_gripper = not_in_gripper_detection_pc_mesh(mps, torch.squeeze(points, dim=0).cpu().numpy()[:, :3])
            idx = torch.cat([torch.tensor(idx_collsion), torch.tensor(idx_not_in_gripper)]).unique()
            gripper_check_loss = len(idx) / len(mps)

            # Visualizer.render_image(obj, mps, bad_idx=idx, path=f"{path}/{i}.png")
        else:
            gripper_check_loss = None


        diversity_loss, dist = self.validation_loss(model_pred, grasp, 'y_to_x', coverage=True)
        # _, t_dist = self.validation_loss(model_pred[..., :3, 3], grasp[..., :3, 3], 'y_to_x', coverage=True)
        accuracy_loss = self.validation_loss(model_pred, grasp, 'x_to_y')
        coverage = torch.sum(torch.tensor(dist) < 0.03) / len(dist)
        # t_coverage = torch.sum(t_dist < 0.05) / len(t_dist) #Todo: adjust chamfer dist to this
        return diversity_loss, accuracy_loss, coverage, np.array(gripper_check_loss) #, t_coverage  # return batch mean control point distance

    @torch.no_grad()
    def predict(self, points: torch.Tensor, inv_fct) -> torch.Tensor:
        # scale data if necessarry, otherwise the scaler will return unchanged values

        # embedding = torch.from_numpy(embedding).float().to(self.device).unsqueeze(0)

        # embedding = self.scaler.scale_input(embedding)
        self.embedder.eval()
        self.model.eval()



        global_embedding, local_embedding = self.embedder(points)

        if self.diffusion_kde:
            # Generate multiple action samples from the diffusion model as usual (these can be done in parallel).
            # Fit a simple kernel-density estimator (KDE) over all samples, and score the likelihood of each.
            # Select the action with the highest likelihood.
            # https://openreview.net/pdf?id=Pv1GPQzRrC8

            # repeat state and goal tensor before passing to the model
            embedding = torch.repeat_interleave(global_embedding, self.diffusion_kde_samples, dim=0)
            # generate multiple samples
            model_pred = self.model(embedding)
                                                    # if len(x_0.size()) == 3:
                                                    #     x_0 = x_0[:, -1, :]
            # apply kde
            x_kde = model_pred.detach().cpu()
            kde = KernelDensity(kernel='gaussian', bandwidth=0.4).fit(x_kde)
            kde_prob = kde.score_samples(x_kde)
            max_index = kde_prob.argmax(axis=0)
            # take prediction with the highest likelihood
            model_pred = model_pred[max_index]
        # do default model evaluation
        else:
            model_pred = self.model(global_embedding)
            model_pred = inv_fct(model_pred)

        return model_pred  # .cpu().numpy()

    def visualize_trajectory_forward(self, loader, num_obj=1, num_samples=10):
        for i, tensors in enumerate(loader):
            if i >= num_obj:
                break
            points, grasp = tensors
            # (points, grasp), success = test_loader.dataset.get_screwdriver()
            inv_fct = loader.dataset.processor.postprocess
            self.embedder.eval()
            self.model.eval()
            grasp = torch.squeeze(grasp)
            noise = torch.randn_like(grasp).to(self.device)
            for t_step in range(self.model.n_timesteps):

                t = torch.ones(len(grasp), dtype=torch.int64).to(self.device) * t_step

                g_t = self.model.q_sample(grasp, t, noise)


                if self.processor.representation_name == 'rotation_6d_representation':
                    t, a1, a2 = g_t[..., :3], g_t[..., 3:6], g_t[..., 6:]
                    b1 = F.normalize(a1, dim=-1)
                    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
                    b2 = F.normalize(b2, dim=-1)
                    g_t = torch.cat((t, b1, b2), dim=-1)

                Visualizer.visualize(loader.dataset.processor.current_obj, inv_fct(g_t))


    def visualize_trajectory_backward(self, test_loader, num_obj=1, num_samples=10):
        """
        Visualize the trajectory of the denoising process
        :param obj:
        :return:
        """
        for i, tensors in enumerate(test_loader):
            points, grasp = tensors
            # (points, grasp), success = test_loader.dataset.get_screwdriver()
            inv_fct = test_loader.dataset.processor.postprocess
            self.embedder.eval()
            self.model.eval()

            global_embedding, local_embedding = self.embedder(torch.unsqueeze(points, dim=0))
            # stack subbatches to a batch
            global_embedding = torch.repeat_interleave(global_embedding, num_trajectories, dim=0)

            # flatten data
            global_embedding = torch.flatten(global_embedding, start_dim=1)  # dim0 is batch size
            batch_size = global_embedding.shape[0]
            shape = (batch_size, grasp.size(dim=-1))
            model_pred, trajectories = self.model.p_sample_loop(global_embedding, shape, return_diffusion=True)
            # apply inverse function
            colors = [[1 - i / trajectories.size(dim=1), i / trajectories.size(dim=1), 0] for i in range(trajectories.size(dim=1))]
            for traj in trajectories:
                traj = inv_fct(traj)
                Visualizer.visualize(test_loader.dataset.processor.current_obj, traj, colors=colors)

    def visualize_diffused_distribution(self, loader, num_steps=5, num_obj=1, num_samples=100):

        for i, tensors in enumerate(loader):
            if i >= num_obj:
                break
            points, grasp = tensors
            # (points, grasp), success = test_loader.dataset.get_screwdriver()
            inv_fct = loader.dataset.processor.postprocess
            self.embedder.eval()
            self.model.eval()
            grasp = torch.squeeze(grasp)
            noise = torch.randn_like(grasp).to(self.device)
            for t_step in range(num_steps):

                t = torch.ones(len(grasp), dtype=torch.int64).to(self.device) * t_step

                g_t = self.model.q_sample(grasp, t, noise)


                if self.processor.representation_name == 'rotation_6d_representation':
                    t, a1, a2 = g_t[..., :3], g_t[..., 3:6], g_t[..., 6:]
                    b1 = F.normalize(a1, dim=-1)
                    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
                    b2 = F.normalize(b2, dim=-1)
                    g_t = torch.cat((t, b1, b2), dim=-1)

                Visualizer.visualize(loader.dataset.processor.current_obj, inv_fct(g_t))


    def load_pretrained_model(self, weights_path: str, **kwargs) -> None:
        """
        Method to load a pretrained model weights inside self.model
        """
        self.embedder.load_state_dict(torch.load(os.path.join(weights_path, "embedder_state_dict.pth")))
        self.model.load_state_dict(torch.load(os.path.join(weights_path, "model_state_dict.pth")))
        log.info('Loaded pre-trained model parameters')

    def store_model_weights(self, store_path: str) -> None:
        """
        Store the model weights inside the store path as model_weights.pth
        """
        torch.save(self.embedder.state_dict(), os.path.join(store_path, "embedder_state_dict.pth"))
        torch.save(self.model.state_dict(), os.path.join(store_path, "model_state_dict.pth"))
        log.info('Stored model parameters')
