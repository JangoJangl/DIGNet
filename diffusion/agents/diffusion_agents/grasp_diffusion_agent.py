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
from helper.visualize import GRIPPER_MESHES_AVAILABLE, Visualizer
from helper.plotting import grasp_figure, wandb_object3d, COLORS
from helper.hub import HubUploader
from helper.collision import collision_detection, collision_detection_pc_mesh, not_in_gripper_detection_pc_mesh
import torch.nn.functional as F

# A logger for this file
log = logging.getLogger(__name__)


class GraspDiffusionAgent(BaseAgent):


    #TODO: Use local Embedding as well
    def __init__(
            self,
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
            num_eval_samples: int = 100,
            max_eval_objects: Optional[int] = None,
            coverage_threshold: float = 0.03,
            symmetric_metric: bool = True,
            visualize_at: tuple = (0.25, 0.5, 0.75, 1.0),
            num_vis_objects: int = 4,
            hub_repo_id: Optional[str] = None,
            hub_private: bool = True,
            checkpoint_every_n_steps: Optional[int] = None,
    ):
        super().__init__(model, optimization, device, max_train_steps, eval_every_n_steps)
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
        # number of grasps generated per test object, coverage and precision are only meaningful for many samples
        self.num_eval_samples = num_eval_samples
        self.max_eval_objects = max_eval_objects
        self.coverage_threshold = coverage_threshold
        # parallel jaw grasps that differ by a 180 degree flip around the approach axis count as identical
        self.symmetric_metric = symmetric_metric
        # fractions of max_train_steps at which predictions on the first test objects are plotted to wandb
        self.visualize_at = visualize_at
        self.num_vis_objects = num_vis_objects
        # full training state (model, optimizer, EMA, step) for resuming, default: at every evaluation
        self.checkpoint_every_n_steps = checkpoint_every_n_steps or eval_every_n_steps
        self.best_test_loss = 1e2
        run_name = wandb.run.name if wandb.run is not None else os.path.basename(self.working_dir)
        self.hub = HubUploader.from_config(hub_repo_id, hub_private, run_name)
        params = {f"params/{name}": sum(p.numel() for p in module.parameters())
                  for name, module in self.model.named_children()}
        params["params/total"] = sum(p.numel() for p in self.model.get_params())
        log.info(f"Parameter counts: {params}")
        if wandb.run is not None:
            wandb.run.summary.update(params)

    # What exactly is scaler doing?
    def set_processor(self, processor):
        """
        Define the scaler from the Workspace class used to scale embedding and output data if necessary
        """
        self.processor = processor
        self.model.inv_fct = processor.postprocess

    def set_bounds(self, size_bounds=7):
        """
        Define the bounds for the sampler class
        """

        bounds = np.concatenate([np.ones(size_bounds)]) # Bounds are in between of 1 and -1 because of unit variance scaling
        # unscaled log map rotations lie in [-pi, pi], clamping them to [-1, 1] would cut off all rotations > 1 rad
        processor = getattr(self, 'processor', None)
        if processor is not None and getattr(processor.scaler, '__name__', '') == 'no_scale' \
                and processor.representation_name in ('se3_log_map_representation', 'se3_pseudo_log_map_representation'):
            bounds[3:6] = np.pi
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

        if not GRIPPER_MESHES_AVAILABLE:
            log.warning("alr_sim not installed: gripper_check_loss (collision metric) is not computed")
        vis_steps = {max(1, int(round(f * self.max_train_steps))) for f in self.visualize_at}
        generator = iter(train_loader)
        start = self.steps  # > 0 after resume_training
        for step in tqdm(range(start, self.max_train_steps), initial=start, total=self.max_train_steps):
            # run a test batch every n steps
            if not self.steps % self.eval_every_n_steps:
                chamfer_loss = self.log_evaluation(test_loader)
                if chamfer_loss < self.best_test_loss:
                    self.best_test_loss = chamfer_loss
                    self.store_model_weights(self.working_dir)
                    log.info('New best test loss. Stored weights have been updated!')
                    if self.hub is not None:
                        self.hub.upload(self.working_dir, f"best checkpoint, step {self.steps}")

            try:
                batch_loss = self.train_step(*next(generator)) #preprocess data + train step
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                generator = iter(train_loader)
                batch_loss = self.train_step(*next(generator))

            # log.info loss every 100 steps
            if not self.steps % 100:
                log.info("Step {}: Mean batch loss {}".format(step, batch_loss))
            wandb.log({"train_loss": float(batch_loss)}, step=self.steps)
            if self.steps in vis_steps:
                self.log_visualizations(test_loader)
            if not self.steps % self.checkpoint_every_n_steps:
                self.save_training_state(os.path.join(self.working_dir, 'last'))
                if self.hub is not None:
                    self.hub.upload(self.working_dir, f"training state, step {self.steps}")

        # final evaluation and checkpoint, the best checkpoint stays in the run directory
        self.log_evaluation(test_loader)
        final_dir = os.path.join(self.working_dir, 'final')
        os.makedirs(final_dir, exist_ok=True)
        self.store_model_weights(final_dir)
        self.save_training_state(final_dir)
        if self.hub is not None:
            self.hub.upload(self.working_dir, f"final checkpoint, step {self.steps}")

    def log_evaluation(self, test_loader):
        diversity_loss, accuracy_loss, coverage, gripper_check_loss = self.evaluate_agent(test_loader)
        chamfer_loss = diversity_loss + accuracy_loss
        log.info("Step {}: Mean diversity loss is {}".format(self.steps, diversity_loss))
        log.info("Step {}: Mean accuracy loss is {}".format(self.steps, accuracy_loss))
        wandb.log(
            {
                "diversity_loss": diversity_loss,
                "accuracy_loss": accuracy_loss,
                "chamfer_loss": chamfer_loss,
                "coverage": coverage,
                "gripper_check_loss": gripper_check_loss,
            }, step=self.steps
        )
        return chamfer_loss

    @torch.no_grad()
    def log_visualizations(self, test_loader, num_samples=50):
        """ Plots ground truth (blue) and sampled grasps (green) for the first test objects and logs them to wandb """
        if self.use_ema:
            self.ema_helper.store(self.model.get_params())
            self.ema_helper.copy_to(self.model.get_params())
        try:
            self.model.eval()
            samples, objects = [], {}
            gram_schmidt = self.processor.representation_name == 'rotation_6d_representation'
            for i, (points, grasp) in enumerate(test_loader):
                if i >= self.num_vis_objects:
                    break
                pred = self.processor.postprocess(self.model(points, gram_schmidt=gram_schmidt, num_samples=num_samples))
                gt = self.processor.postprocess(torch.flatten(torch.flatten(grasp, 0, 1), 1))
                cloud = points[0, :, :3]
                samples.append({'points': cloud, 'gt': gt, 'pred': pred, 'title': f"object {i}"})
                objects[f"val_3d/object_{i}"] = wandb_object3d(cloud, [(gt[:20], COLORS['gt']), (pred[:20], COLORS['pred'])])
            progress = int(round(100 * self.steps / self.max_train_steps))
            fig = grasp_figure(samples, ncols=len(samples), views=((25, 45), (80, -90)))
            os.makedirs(os.path.join(self.working_dir, 'visualizations'), exist_ok=True)
            fig.savefig(os.path.join(self.working_dir, 'visualizations', f"step_{self.steps:07d}_{progress:03d}pct.png"), dpi=100)
            wandb.log({"val_grasps": wandb.Image(fig, caption=f"{progress} % of training, blue: ground truth, green: sampled"),
                       **objects}, step=self.steps)
            import matplotlib.pyplot as plt
            plt.close(fig)
        finally:
            if self.use_ema:
                self.ema_helper.restore(self.model.get_params())


    def train_step(self, points: torch.Tensor, grasp: torch.Tensor) -> float:

        # Set the model to train mode.
        self.model.train()
        # Compute the embedding.
        # stack subbatches to a batch
        # embedding = torch.repeat_interleave(global_embedding, grasp.size(dim=1), dim=0)
        grasp = torch.flatten(grasp, start_dim=0, end_dim=1)

        # flatten data
        grasp = torch.flatten(grasp, start_dim=1)  # dim0 is batch size
        # embedding = torch.flatten(embedding, start_dim=1)  # dim0 is batch size


        # embedding = self.scaler.scale_input(embedding)
        # grasp = self.scaler.scale_output(grasp)

        # Compute the loss.
        if self.processor.representation == 'rot_6d_representation':
            loss = self.model.loss(grasp, points, gram_schmidt=True)
        else:
            loss = self.model.loss(grasp, points)
        # Before the backward pass, zero all the network gradients
        self.optimizer.zero_grad()
        # Backward pass: compute gradient of the loss with respect to parameters
        loss.backward()
        # Calling the step function to update the parameters
        self.optimizer.step()
        self.steps += 1
        # update the ema model
        if self.use_ema and self.steps % self.update_ema_every_n_steps == 0:
            self.ema_helper.update(self.model.get_params())
        return loss.detach()


    def evaluate_agent(self, test_loader, path=None):
        # evaluate with the averaged weights, the raw weights are restored afterwards
        if self.use_ema:
            self.ema_helper.store(self.model.get_params())
            self.ema_helper.copy_to(self.model.get_params())
        try:
            return self._evaluate_agent(test_loader, path)
        finally:
            if self.use_ema:
                self.ema_helper.restore(self.model.get_params())

    def _evaluate_agent(self, test_loader, path=None):
        test_div_loss = []
        test_acc_loss = []
        test_gripper_check_loss = []
        test_coverage = []
        for i, tensors in enumerate(test_loader):
            if self.max_eval_objects is not None and i >= self.max_eval_objects:
                break
            gripper_check = True if i < 5 else False
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
        avrg_test_gripper_check_loss = np.mean(test_gripper_check_loss) if test_gripper_check_loss else float('nan')
        return avrg_test_div_loss, avrg_test_acc_loss, avrg_test_coverage, avrg_test_gripper_check_loss

    def evaluate_agent_sizes(self, test_loader):
        areas = torch.tensor(test_loader.dataset.areas)
        areas_s, area_idcs = torch.sort(areas)
        maxs = torch.tensor(test_loader.dataset.maxs)
        maxs_s, max_idcs = torch.sort(maxs)
        variances = torch.tensor(test_loader.dataset.variances)
        variances_s, var_idcs = torch.sort(variances)
        smallest = 0.025
        small = 0.05
        medium = 0.1
        large = 0.2
        largest = 1
        smallest_idx = (areas < smallest).nonzero()
        small_idx = ((areas >= smallest) & (areas < small)).nonzero()
        medium_idx = ((areas >= small) & (areas < medium)).nonzero()
        large_idx = ((areas >= medium) & (areas < large)).nonzero()
        largest_idx = (areas >= large).nonzero()
        for i, size in zip(var_idcs, variances_s):
            points, grasp = test_loader.dataset[i]
            # (points, grasp), success = test_loader.dataset.get_screwdriver()
            obj = test_loader.dataset.processor.current_obj

            inv_fct = self.processor.postprocess
            div_loss, acc_loss, coverage, gripper_check_loss = self.evaluate(torch.unsqueeze(points, dim=0), torch.unsqueeze(grasp, dim=0), inv_fct,
                                                                             test_loader.dataset.processor.current_obj,
                                                                             i, gripper_check=True)
            wandb.log(
                {
                    "object_area": areas[i],
                    "object_max": maxs[i],
                    "object_variance": variances[i],
                    "diversity_loss": div_loss,
                    "accuracy_loss": acc_loss,
                    "chamfer_loss": div_loss + acc_loss,
                    "coverage": coverage,
                    "gripper_check_loss": gripper_check_loss,
                }
            )

    @torch.no_grad()
    def evaluate(self, points: torch.Tensor, grasp: torch.Tensor, inv_fct, obj, i, gripper_check=False, path=None) -> float:

        # scale data if necessarry, otherwise the scaler will return unchanged values
        # embedding = self.scaler.scale_input(embedding)
        # grasp = self.scaler.scale_output(grasp)


        # Set the model to eval mode.
        self.model.eval()
        # # stack subbatches to a batch
        # global_embedding = torch.repeat_interleave(global_embedding, grasp.size(dim=1), dim=0)
        grasp = torch.flatten(grasp, start_dim=0, end_dim=1)

        # flatten data
        grasp = torch.flatten(grasp, start_dim=1)  # dim0 is batch size
        # global_embedding = torch.flatten(global_embedding, start_dim=1)  # dim0 is batch size


        # post = self.processor.postprocess(grasp)
        # Visualizer.render_images(obj, post[:100])

        gram_schmidt = self.processor.representation_name == 'rotation_6d_representation'
        model_pred = self.model(points, gram_schmidt=gram_schmidt, num_samples=self.num_eval_samples)

        # apply inverse function
        model_pred = inv_fct(model_pred)
        grasp = inv_fct(grasp)
        #filtering grasp proposals
        if gripper_check and GRIPPER_MESHES_AVAILABLE:  # needs the Panda meshes of alr_sim
            if len(model_pred) < 100:
                mps = model_pred
            else:
                mps = model_pred[torch.randint(len(model_pred), (100,))]
            idx_collsion = collision_detection_pc_mesh(mps, torch.squeeze(points, dim=0).cpu().numpy()[:, :3])
            idx_not_in_gripper = not_in_gripper_detection_pc_mesh(mps, torch.squeeze(points, dim=0).cpu().numpy()[:, :3])
            idx = torch.cat([torch.tensor(idx_collsion), torch.tensor(idx_not_in_gripper)]).unique()
            gripper_check_loss = len(idx) / len(mps)

            # Visualizer.render_image(obj, mps, bad_idx=idx, path=f"{path}/{i}.png")
        else:
            gripper_check_loss = None


        # diversity: mean distance of every ground truth grasp to its closest prediction (recall side)
        # accuracy: mean distance of every prediction to its closest ground truth grasp (precision side)
        diversity_loss, dist = self.validation_loss(model_pred, grasp, 'y_to_x', coverage=True,
                                                     symmetric=self.symmetric_metric)
        accuracy_loss = self.validation_loss(model_pred, grasp, 'x_to_y', symmetric=self.symmetric_metric)
        coverage = float(np.mean(np.asarray(dist) < self.coverage_threshold))
        return diversity_loss, accuracy_loss, coverage, gripper_check_loss

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

    def visualize_trajectory_forward(self, loader, num_obj=1, num_samples=1):
        for i, tensors in enumerate(loader):
            if i >= num_obj:
                break
            points, grasp = tensors
            # (points, grasp), success = test_loader.dataset.get_screwdriver()
            inv_fct = loader.dataset.processor.postprocess
            self.embedder.eval()
            self.model.eval()
            grasp = torch.squeeze(grasp)
            grasp = grasp[:num_samples]
            list_gt = torch.clone(grasp)
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
                list_gt = torch.cat((list_gt, g_t), dim=0)

            Visualizer.visualize(loader.dataset.processor.current_obj, inv_fct(list_gt))


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

            global_embedding = self.embedder(points)
            # stack subbatches to a batch
            global_embedding = torch.repeat_interleave(global_embedding, num_samples, dim=0)

            # flatten data
            global_embedding = torch.flatten(global_embedding, start_dim=1)  # dim0 is batch size
            batch_size = global_embedding.shape[0]
            shape = (batch_size, grasp.size(dim=-1))
            model_pred, trajectories = self.model.p_sample_loop(global_embedding, shape, return_diffusion=True)
            # apply inverse function
            colors = [[1 - i / trajectories.size(dim=1), i / trajectories.size(dim=1), 0] for i in range(trajectories.size(dim=1))]
            scene = Visualizer.visualize(test_loader.dataset.processor.current_obj, inv_fct(trajectories.view(-1,7)))

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
        self.model.load_state_dict(torch.load(os.path.join(weights_path, "model_state_dict.pth")))
        log.info('Loaded pre-trained model parameters')

    def save_training_state(self, store_path: str) -> None:
        """ model, optimizer, EMA, step and best metric in store_path/training_state.pth, see resume_training """
        os.makedirs(store_path, exist_ok=True)
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'ema': self.ema_helper.state_dict() if self.use_ema else None,
            'steps': self.steps,
            'best_test_loss': self.best_test_loss,
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy_rng': np.random.get_state(),
        }
        tmp = os.path.join(store_path, 'training_state.pth.tmp')
        torch.save(state, tmp)
        os.replace(tmp, os.path.join(store_path, 'training_state.pth'))  # never leave a half written file
        log.info(f'Stored training state at step {self.steps}')

    def resume_training(self, source: str) -> None:
        """
        Continue a run from a training_state.pth: a local file or folder, or '<hf_repo_id>:<path in repo>'
        (e.g. user/dignet-flow20m:runs/<run>/last/training_state.pth, needs HF_TOKEN for private repos).
        """
        if ':' in source and not os.path.exists(source):
            from huggingface_hub import hf_hub_download
            repo_id, filename = source.split(':', 1)
            source = hf_hub_download(repo_id, filename)
        if os.path.isdir(source):
            source = os.path.join(source, 'training_state.pth')
        state = torch.load(source, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        if self.use_ema and state.get('ema') is not None:
            self.ema_helper.load_state_dict(state['ema'])
        self.steps = state['steps']
        self.best_test_loss = state['best_test_loss']
        torch.set_rng_state(state['torch_rng'].cpu())
        if state.get('cuda_rng') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
        np.random.set_state(state['numpy_rng'])
        log.info(f'Resumed training from {source} at step {self.steps}')

    def store_model_weights(self, store_path: str) -> None:
        """
        Store the model weights inside the store path as model_weights.pth
        """
        torch.save(self.model.state_dict(), os.path.join(store_path, "model_state_dict.pth"))
        if self.use_ema:
            torch.save(self.ema_helper.state_dict(), os.path.join(store_path, "ema_state_dict.pth"))
        log.info('Stored model parameters')
