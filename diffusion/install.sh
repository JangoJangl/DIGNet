#!/bin/bash

############ GENERAL ENV SETUP ############
echo New Environment Name:
read envname

echo Creating new conda environment $envname 
conda create -n $envname python=3.8 -y -q

eval "$(conda shell.bash hook)"
conda activate $envname

echo
echo Activating $envname
if [[ "$CONDA_DEFAULT_ENV" != "$envname" ]]
then
    echo Failed to activate conda environment.
    exit 1
fi


############ PYTHON ############
echo install mamba
conda install mamba -c conda-forge -y -q


############ REQUIRED DEPENDENCIES (PYBULLET) ############
echo Installing dependencies...

# Pytorch
# mamba install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
# mamba install pytorch torchvision cudatoolkit=11.6 -c pytorch -c conda-forge -y
mamba install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia -y
mamba install -c fvcore -c iopath -c conda-forge fvcore iopath -y

mamba install -c conda-forge hydra-core -y -q
# Mujoco System Dependencies
mamba install -c conda-forge glew patchelf -y -q
#mamba install pytorch3d -c pytorch3d -y
mamba install pytorch3d -c pytorch3d -y
# Set Conda Env Variables
#conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
#conda env config vars set LD_PRELOAD=$LD_PRELOAD:$CONDA_PREFIX/lib/libGLEW.so
# conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/usr/lib/x86_64-linux-gnu/nvidia-opengl/MUJOCO_GL=egl
# Activate Mujoco Py Env Variables
#conda activate $envname

# pip install mujoco
# Install MujocoPy
#pip install mujoco-py

# Install other PIP Dependencies
# pip install git+https://github.com/rail-berkeley/d4rl@master
# pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
pip install dm-control==0.0.403778684
#pip install gym==0.21.0
pip install termcolor
pip install wandb
pip install tikzplotlib
pip install einops
#pip install torchdiffeq
pip install gin-config
pip install pybullet
pip install -U scikit-learn
#pip install torchsde
pip install open3d
pip install trimesh


# install Simulations Framework and acronym
echo Install acronym tools
cd $(find ~/repos -name "acronym")
pip install -e .
echo install alr_sim
cd $(find ~/repos -name "SimulationFramework")
pip install -e .

pip install rtree

mamba install -c conda-forge suitesparse -y -q

pip install theseus-ai 


# pip install opencv-python==4.6.0.66

# Clone bet
#echo Clone Behavior Transformers
# git clone https://github.com/notmahi/bet

#echo Clone Relay policy learning/\
# git clone https://github.com/google-research/relay-policy-learning

echo Done installing all necessary packages. Please follow the next steps mentioned on the readme

exit 0
