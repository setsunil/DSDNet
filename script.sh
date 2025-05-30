echo y | conda create -n pytorch131_DSDp python=3.8
conda activate pytorch131_DSDp
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
echo y | pip install -r requirements.txt
python setup.py develop --no_cuda_ext