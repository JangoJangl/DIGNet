#!/bin/bash

# Create a folder for the jobs
declare FOLDER_NAME="architecture_search"
rm -rf ./${FOLDER_NAME}
mkdir ./${FOLDER_NAME}

# Declare options for hyperparameter search
declare -a seeds=(26 42 69)
declare -a representations=('se3_log_map_representation'  'trans_quat_representation')
declare -a grasp_dims=(6 7)
declare -a agents=(diffusion_grasp_agent diffusion_grasp_agent_vnn)
declare -a networks=(networks.diffusion.diffusion_models.FiLMResidualMLPNetworkV2 networks.diffusion.diffusion_models.DiffusionMLPNetwork)
declare -a noise_pred=(true false)

arraylength=${#representations[@]}
# Read the array values with space
for  seed in "${seeds[@]}"; do
  for (( i=0; i<${arraylength}; i++ )); do
    for agent in "${agents[@]}"; do
      for network in "${networks[@]}"; do
        for np in "${noise_pred[@]}"; do
          declare job_name="job_${representations[$i]}_${grasp_dims[$i]}_${agent}_${network}_${np}_${seed}.sh"
          cp job_template.sh $FOLDER_NAME/$job_name
          sed -i "3i#SBATCH -J ${job_name}" $FOLDER_NAME/$job_name
          echo "python diffusion/training.py \
                representation='${representations[$i]}' \
                grasp_dim=${grasp_dims[$i]} \
                agents=${agent} \
                model_target=${network} \
                predict_epsilon=${np} \
                seed=${seed} \
                root_path=$TMP/acronym \
                data_path=$TMP/acronym/mesh_contacts_graspnet \
                h5_path=$TMP/acronym/grasps\
                ">> $FOLDER_NAME/$job_name
          done
        done
    done
  done
done