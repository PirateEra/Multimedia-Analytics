# Load required modules
module load 2023
module load Anaconda3/2023.07-2

# Remove any existing environment (optional safety check)
conda remove --name g_retriever --all -y

# Create a new environment with Python 3.9
conda create --name g_retriever python=3.9 -y
source activate g_retriever  # or: conda activate g_retriever

# Install PyTorch 2.1 with CUDA 11.8
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install PyTorch Geometric dependencies compiled for torch 2.1.0 + CUDA 11.8
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Install torch-geometric itself
pip install torch-geometric

# Install remaining dependencies
pip install \
  peft \
  pandas \
  ogb \
  transformers \
  wandb \
  sentencepiece \
  datasets \
  pcst_fast \
  gensim \
  scipy==1.12 \
  protobuf \
  python-dotenv \
  ollama

