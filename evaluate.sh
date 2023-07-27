runs=$(find logs/cluster/grasping_end_to_end_learning/runs/ -mindepth 2 -maxdepth 2)

for run in $runs; do
    echo "Evaluating ${PWD}/${run}"
    echo ${PYTHONPATH}
    eval "python diffusion/evaluation.py pretrained_model_path=${PWD}/${run} \
        root_path=/home/temp_store/jandl/Dataset/acronym \
        data_path=/home/temp_store/jandl/Dataset/acronym/mesh_contacts_graspnet \
        h5_path=/home/temp_store/jandl/Dataset/acronym/grasps "
done