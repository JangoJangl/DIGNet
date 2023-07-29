#!/bin/bash

# Create a folder for the jobs
declare FOLDER_NAME="architecture_search"
rm -rf ./${FOLDER_NAME}
mkdir ./${FOLDER_NAME}

# Declare options for hyperparameter search
declare -a seeds=(69)
declare -a representations=('se3_log_map_representation'  'se3_pseudo_log_map_representation' 'rotation_6d_representation' 'trans_quat_representation')
declare -a grasp_dims=(6 6 9 7)
declare -a agents=(diffusion_grasp_agent)
declare -a networks=(networks.diffusion.diffusion_models.DiffusionMLPNetwork)
declare -a noise_pred=(true)
declare -a scales=('var_scale' 'no_scale')
declare -a radii=(0.01 0.05 0.1)
declare -a diffusion_steps=(10 100)
declare -a batch_sizes=(5 100)
declare -a embedding_dimensions=(1024)

arraylength=${#representations[@]}
# Read the array values with space
for  radius in "${radii[@]}"; do
  for (( i=0; i<${arraylength}; i++ )); do
    for scale in "${scales[@]}"; do
      for steps in "${diffusion_steps[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
          declare job_name="job_${representations[$i]}_${grasp_dims[$i]}_${scale}_${radius}_${steps}_${batch_size}.sh"
          cp job_template.sh $FOLDER_NAME/$job_name
          sed -i "3i#SBATCH -J ${job_name}" $FOLDER_NAME/$job_name
          echo "python diffusion/training.py \
                group='final_param_search' \
                representation='${representations[$i]}' \
                grasp_dim=${grasp_dims[$i]} \
                scaler='${scale}' \
                n_steps = ${steps} \
                batch_size=${batch_size} \
                pointnet_radius=${radius} \
                root_path=\$TMP/acronym \
                data_path=\$TMP/acronym/mesh_contacts_graspnet \
                h5_path=\$TMP/acronym/grasps\
                ">> $FOLDER_NAME/$job_name
                done
          done
    done
  done
done