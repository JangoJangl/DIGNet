#!/bin/bash

# Declare a string array with type
declare -a representations=('se3_log_map_representation'  'trans_quat_representation')
declare -a grasp_dims=(6 7)
declare -a agents=(diffusion_grasp_agent diffusion_grasp_agent_vnn)
declare -a networks=(networks.diffusion.diffusion_models.FiLMResidualMLPNetworkV2 networks.diffusion.diffusion_models.DiffusionMLPNetwork)
declare -a  noise_pred=(true false)

arraylength=${#representations[@]}
# Read the array values with space
for (( i=0; i<${arraylength}; i++ )); do
  for agent in "${agents[@]}"; do
    for network in "${networks[@]}"; do
      for np in "${noise_pred[@]}"; do
        eval "python diffusion/training.py representation='${representations[$i]}' grasp_dim=${grasp_dims[$i]} agents=${agent} model_target=${network} predict_epsilon=${np}"
        done
      done
  done
done