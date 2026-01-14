apt-get install gh

git clone https://github.com/Seungwoo-Yoon/sam-3d-objects.git
cd sam-3d-objects

/opt/miniforge3/condabin/conda env create -f environments/default.yml
/opt/miniforge3/condabin/conda activate sam3d-objects

export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
pip install -e '.[dev]'
pip install -e '.[p3d]'

export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'

./patching/hydra
