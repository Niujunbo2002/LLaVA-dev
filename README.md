
## Install Package

```Shell
conda create -n nativeres python=3.10 -y
conda activate nativeres
pip install --upgrade pip  # enable PEP 660 support
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.50.3
pip install -r requirements.txt
```