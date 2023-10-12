#!/bin/bash

# See https://docs.omniverse.nvidia.com/isaacsim/latest/manual_standalone_python.html
echo "Setting up dectransformer Environment..."
source activate base	
conda deactivate
conda activate conda39-dectransformer
echo "$PYTHON_PATH"
