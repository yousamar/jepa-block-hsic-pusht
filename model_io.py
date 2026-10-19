"""Portable inference-only FP32 state dictionaries, without optimizer or pickle models."""
from pathlib import Path
import hydra
import torch
from omegaconf import OmegaConf

FORMAT='jepa-block-hsic-pusht.state_dict.v1'

def save_inference_model(model, model_config, path, metadata=None):
    config=OmegaConf.to_container(model_config,resolve=True) if OmegaConf.is_config(model_config) else model_config
    payload={'format':FORMAT,'model_config':config,
             'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
             'metadata':metadata or {}}
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    torch.save(payload,path)

def load_inference_model(path,device='cpu'):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(payload,dict) or payload.get('format')!=FORMAT:
        raise ValueError('Expected a portable inference bundle exported by model_io.py')
    model=hydra.utils.instantiate(OmegaConf.create(payload['model_config']))
    model.load_state_dict(payload['state_dict'],strict=True)
    return model.to(device).eval().requires_grad_(False)
