# If encountering "Your shell has not been properly configured to use 'conda activate'.", then "conda deactivate" from (base)
source activate base	
conda deactivate
	
conda init bash
conda create --name conda39-dectransformer python=3.9
conda activate conda39-dectransformer
pip install -r requirements.txt -t ./pipenv
conda deactivate
conda clean --all	# Purge cache and unused apps
condo info

# Tensorflow with Cuda GPU install (Conda)
# See
# https://utho.com/docs/tutorial/how-to-install-anaconda-on-ubuntu-20-04-lts/
# https://www.tensorflow.org/install/pip
conda install cuda -c nvidia
conda install -c nvidia cudnn
	* Set these env variables after CUDNN is installed
	CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib

# Torch not compiled with CUDA enabled ubuntu
# https://www.datasciencelearner.com/assertionerror-torch-not-compiled-with-cuda-enabled-fix/
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113


source dectransformer.sh
conda activate decision-transformer-atari



# Atari
python3 run_dt_atari.py --seed 123 --block_size 90 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Breakout' --batch_size 128 --data_dir_prefix ./dqn_replay


python3 run_dt_atari.py --seed 123 --block_size 90 --epochs 1 --model_type 'reward_conditioned' --num_steps 5000 --num_buffers 50 --game 'Breakout' --batch_size 128 --data_dir_prefix ./dqn_replay



# Gym
~/.mujocp and ~/.d4rl directories to be set up. Then
Please add following line(s) to .bashrc:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/timityjoe/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

python3 experiment.py --env hopper --dataset medium --model_type dt -w










