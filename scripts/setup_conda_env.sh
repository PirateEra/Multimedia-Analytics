module load 2023
module load Anaconda3/2023.07-2

conda create --name g_retriever python=3.9 -y
source activate g_retriever

# Install PyTorch with CUDA 11.8
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install PyTorch Geometric extensions
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# Install torch-geometric (this pulls in pyg_lib correctly)
pip install torch-geometric

# Install remaining packages
pip install peft pandas ogb transformers wandb sentencepiece datasets pcst_fast gensim scipy==1.12 protobuf
