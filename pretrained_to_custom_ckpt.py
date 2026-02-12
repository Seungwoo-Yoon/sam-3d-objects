from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
import yaml

from sam3d_objects.model.backbone.generator.shortcut.model import ShortCut
from sam3d_objects.model.io import load_model_from_checkpoint

config_path = 'checkpoints/hf/dual_backbone_generator.yaml'

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# instantiate the model using the loaded config after converting to OmegaConf
config = OmegaConf.create(config)
print(config)
model: ShortCut = instantiate(config)

# load pretrained checkpoint for sparse flow
sparse_flow_ckpt_path = 'checkpoints/hf/ss_generator.ckpt'
model = load_model_from_checkpoint(
    model,
    sparse_flow_ckpt_path,
    strict=True,
    device='cpu',
    freeze=False,
    eval=False,
    state_dict_key='state_dict',
)