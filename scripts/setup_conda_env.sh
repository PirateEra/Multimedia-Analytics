module load 2023
module load Anaconda3/2023.07-2

conda create --name g_retriever python=3.9 -y 
conda activate g_retriever
conda install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia 
python -c "import torch; print(torch.__version__)" 
python -c "import torch; print(torch.version.cuda)" 
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html 
pip install peft 
pip install pandas 
pip install ogb 
pip install transformers 
pip install wandb 
pip install sentencepiece 
pip install torch_geometric 
pip install datasets 
pip install pcst_fast 
pip install gensim 
pip install scipy==1.1
pip install protobuf 