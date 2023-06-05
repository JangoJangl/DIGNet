# generative_learning_from_play

## Installtion Guide

First create a conda environment using the following command

```bash
sh install.sh
```

During this process two additional packages will be installed:

- [BeT](https://github.com/notmahi/bet)
- [Relay Policy Learning](https://github.com/google-research/relay-policy-learning)

Since your conda environment overrides the PYTHONPATH, we have to do add the module differently in order for python to find it, create a file nano <path to your anaconda installation>/envs/<your environment>/lib/python3.8/site-packages/generative-learning-additional-packages.pth (the name of the pth file can be chosen freely) with the content:

```
<path to your relay-policy-learning directory>
<path to your relay-policy-learning directory>/adept_envs
<path to your relay-policy-learning directory>/adept_envs/adept_envs
```

For the Bet Repo you need to adept the following line in the training and evaluation script:
```python
os.environ['ASSET_PATH'] = "/path/to/bet"
```

**Dataset**

Next, you need to download the dataset for the relay kitchen and the block push environment from  [link](https://osf.io/q3dx2/) from the repo [play-to-policy](https://github.com/jeffacce/play-to-policy), unzip and put anywhere.
Now adjust data_path in ```franka_kitchen_main_config.yaml``` to <path to cbet_datasets>/relay_kitchen and the same for the ```block_push_main_config.yaml``` to <path to cbet_datasets>/block_push.

### Common mistakes

- there is an important paramter in every main config called ```window_size``` and ```future_seq_length```. These can be adapted for recurrent models with arbitarty length and need to be 1 for all MLP based networks.



### TO Do

- Implement Clipping during inference sampling because of Guidance
- Implement Callbacks for Visualization 


### Install
- activate in install.sh `git clone https://github.com/google-research/relay-policy-learning`
- Start install.sh
- follow readme of relay-policy-learning
- git clone https://github.com/jeffacce/play-to-policy
- follow Franka kitchen section in play-to-policy

### Notes on K-diffusion conditioning

The score of a edm diffusion model is defined as:
$$\nabla_{x} \mathrm{log}p(x; \sigma) = (D(x;\sigma) - x) / \sigma^{2}$$
In our case we always have an observation conditioned model:
$$\nabla_{x} \mathrm{log}p(x; \sigma | o) = (D(x;\sigma | o) - x) / \sigma^{2}$$

---

#### Classifier-Guidance

We can use an additional neural network, which has $x$ as its input to get a gradient for guidance during sampling:
$$\nabla_{x} \mathrm{log}p(x|y, o) = \nabla_{x} \mathrm{log}p(x|o) + \nabla_{x} \mathrm{log}p(y|x, o)$$

Our k-diffusion denoiser predicts $D(x;\sigma| o)$
and not $\nabla_{x} \mathrm{log}p(x| o)$. During sampling we compute the derivative $\nabla_{x} \mathrm{log}p(x| o)$ in the method:

```python
def to_d(action, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (action- denoised) / utils.append_dims(sigma, action.ndim)
```
which is equal to:
$$\nabla_{x} \mathrm{log}p(x; \sigma) = (D(x;\sigma) - x) / \sigma^{2}$$

To balance out the $/\sigma^2$ in this computation we need to multiply our guidance gradient by $\sigma^2 \nabla_{x} \mathrm{log}p(y|x, o)$:

$$= (\mathrm{denoised} - x) / \sigma^{2}$$

$$= (D(x;\sigma) +  \nabla_{x} \mathrm{log}p(y|x) * \sigma^{2} - x) / \sigma^{2}$$

$$= (D(x;\sigma) - x) / \sigma^{2} + \nabla_{x} \mathrm{log}p(y|x)$$

$$= \nabla_{x} \mathrm{log}p(x) + \nabla_{x} \mathrm{log}p(y|x)$$

Stuff from [Clip issue](https://github.com/crowsonkb/k-diffusion/issues/20).

---

### Classifier-Free Guidance

By using dropout with a rate of 10 - 30% on the goal input, we are able to learn an implicit non-goal conditioned model. During inference we can use $\lambda$ a guidance factor to parameterize how the gradient of the non-goal conditioned model and the goal-conditioned one are used. 

The general gradient of our inverse ODE is defined by:
$$\nabla_\mathbf{x} \log p(\mathbf{x} \mid \mathbf{y}, \mathbf{o}) = (1 - \lambda) \nabla_\mathbf{x} \log p(\mathbf{x} \mid \mathbf{o}) + \lambda \nabla_\mathbf{x} \log p(\mathbf{y} \mid \mathbf{x}, \mathbf{o})$$

Depending on $\lambda$ we get different behavior:

- $\lambda =1$: we recover normal conditioning
- $\lambda >1$: we guide more towards the goal-condtioned gradient
- $\lambda < 1$: we guide more towards non-goal conditioning
- $\lambda =0$: we have non-goal conditioning 

### Compositional Goal-conditioning

In the Franka Kitchen task we learn a set of 4 skills for every goal, since every trajectory contains a sequence of 4 skills.

### Composing Skills Idea

Can we use negative guidance to force the model to execute one specific skills?

Given the goal-encodings: $s_{g_1}, s_{g_2}, s_{g_3}$, with the following skills:

- $s_{g_1}$: [S1, S2, S3, S4]
- $s_{g_2}$: [S2, S3, S4, S5]
- $s_{g_3}$: [S3, S4, S5, S6]
- $s_{g_4}$: [S4, S5, S6, S7]

We want our model to execute skill S1 only by:

$$\nabla_\mathbf{x} \log p(\mathbf{x} \mid \hat{\mathbf{S}}_{g_1}, \mathbf{o}_t) = \nabla_\mathbf{x} \log p(\mathbf{x} \mid \mathbf{s}_1, \mathbf{o}_t) - \sum_{i=2}^{4}\nabla_\mathbf{x} \log p(\mathbf{x} \mid \mathbf{s}_{g_i}, \mathbf{o}_t)$$

Is this possible?


## Notes about the Goal-conditioned Kitchen Environment

The state vector of the kitchen contains the following information:

0: timestep 

[1:10]: the robot state information

11,12: bottom burner

15,16: top burner

17,18: light switch

19: slide cabinet

20,21: hinge cabinet

22: microwave

[23:29]: kettle

Expected tasks in one-hot encoding:

[
    0: bottom_burner,
    1: top_burner,
    2: light_switch,
    3: sldie_cabinet,
    4: hinge cabinet,
    5: microwave,
    6: kettle,
]

## Notes about the Goal-conditioned Block-Push Environment

The state vector contains the following information:

0,1,2 - red block x, y, yaw

3,4,5 - green block x, y, yaw

6,7,8,9 - end effector x, y, target x, target y

10,11,12 - green target x, y, yaw

13,14,15 - red target x, y, yaw

### Current bug:

- goal state consists of [ 0, 1, 3, 4] the future block positions,
but they did not account for the two possible target positions: red and green. Thus, the agent tries to complete the desired target position, but doesn't care about pushing it in the right colour. Waiting for the fix from the paper guys. 

---


### Diffusion Variants

There exist two general implementations of diffusion agents: the vanilla Denoising Diffusion Probabilistic Model (DDPM) variant  ```agents/model/diffusion_agent.yaml``` based on [^16] and the general class of score-based diffusion agents ```agents/model/k_diffusion_agent.yaml``` from [^20]. The vanilla DDPM is good as a baseline, since it has very few hyperparmeters and work well on almost all tasks without a lot of tuning. The class of k_diffusion are all based on stochastic ODE variants and uses the noise schedule and all scaling and preprocessing methods from from [^20]. 

- Denoising Diffusion Probabilistic Model (DDPM): ```diffusion_agent.yaml```
- Stochastic ODE based Diffusion Model: ```k_diffusion_agent.yaml```

**Samplers**

There exist several samplers for the EDM implementations. Based on the results of the paper [^20] the best sampler is the edm_sampler with/without additional noise injection and a 2nd order Heun solver. Although first results on the state-action based problems show, that there does not exist such a performance gap and sometimes the simpler variants are better.

**Samplers from the K-diffusion code**

- Heun edm sampler: _sample_heun_
- Euler Ancestral sampler: _sample_euler_ancestral_
- Euler edm sampler: _sample_euler_
- DPM edm sampler: _sample_dpm_2_
- Linear Multistep sampler: _sample_lms_
- DPM ancestral sampler: _sample_dpm_2_ancestral_
- DPM fast solver: _dpm_solver_fast_
- DPM adaptive solver: _sample_dpm_adaptive_
- DPM++ 2nd order ancestral sampling: _sample_dpmpp_2s_ancestral_
- DPM++ solver: _sample_dpmpp_2m_

The *DPM* based Samplers are based on the work of [^26], [^27]. While they showed great performance for image-based diffusion processes, they performed much worse on state-action based data so far. 

---


[^16]: [paper link](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf), Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." Advances in Neural Information Processing Systems 33 (2020): 6840-6851.

[^17]: [paper link](https://arxiv.org/pdf/2208.06193), Wang, Zhendong, Jonathan J. Hunt, and Mingyuan Zhou. "Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning." arXiv preprint arXiv:2208.06193 (2022).

[^18]: [paper link](https://openreview.net/pdf?id=PxTIG12RRHS), Song, Yang, et al. "Score-Based Generative Modeling through Stochastic Differential Equations." International Conference on Learning Representations. 2020.

[^19]: [paper link](https://arxiv.org/pdf/2206.11251), Shafiullah, Nur Muhammad Mahi, et al. "Behavior Transformers: Cloning $ k $ modes with one stone." arXiv preprint arXiv:2206.11251 (2022).

[^20]: [paper link](https://arxiv.org/abs/2206.00364), Karras, Tero, et al. "Elucidating the Design Space of Diffusion-Based Generative Models." arXiv preprint arXiv:2206.00364 (2022).


[^26]: [paper link](https://arxiv.org/abs/2206.00927), Lu, Cheng, et al. "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps." arXiv preprint arXiv:2206.00927 (2022).

[^27]: [paper link](https://arxiv.org/pdf/2211.01095), Lu, Cheng, et al. "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models." Advances in Neural Information Processing Systems 35 (2022).
