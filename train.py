"""PushT LeWM: prediction + SIGReg + optional differentiable-bandwidth HSIC.

Derived from huguesva/jepa-stable-token-failure (6b866c9) and le-wm.
The optional HSIC term is training-only. No prefix objectives are included.
"""
from __future__ import annotations

import os
import json
import time
import hashlib
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from ldep import block_dependence_loss
from nuisance.data import NuisanceHDF5Dataset, limit_to_episodes
from nuisance.nuisance import NuisanceSpec, NuisanceRenderer
from utils import get_column_normalizer, get_img_preprocessor


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]  # label
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]
    dep_weight = float(cfg.loss.dep.weight)
    if dep_weight != 0.0:
        output['dep_loss'] = block_dependence_loss(emb, int(cfg.loss.dep.blocks))
        output['weighted_dep_loss'] = dep_weight * output['dep_loss']
        output['loss'] = output['loss'] + output['weighted_dep_loss']

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    self._experiment_metrics = {k:v.detach() for k,v in output.items() if torch.is_tensor(v) and v.numel()==1}
    return output


class JsonMetricsCallback(Callback):
    """Persist actual update counts, diagnostics, compute and memory for each run."""
    def __init__(self, run_dir):
        self.path = Path(run_dir) / 'metrics.jsonl'
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if pl_module.device.type == "cuda": torch.cuda.synchronize()
        self.started = time.perf_counter()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if pl_module.device.type == "cuda": torch.cuda.synchronize()
        row = {k:float(v) for k,v in pl_module._experiment_metrics.items()}
        if not all(__import__('math').isfinite(v) for v in row.values()):
            raise FloatingPointError('Non-finite training metric')
        row.update(epoch=int(trainer.current_epoch),step=int(trainer.global_step),batch_idx=int(batch_idx),wall_time_unix=time.time(),
                   batch_size=int(batch['pixels'].shape[0]),seconds=time.perf_counter()-self.started,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated() if pl_module.device.type == "cuda" else 0,peak_reserved_bytes=torch.cuda.max_memory_reserved() if pl_module.device.type == "cuda" else 0)
        with self.path.open('a') as f:
            f.write(json.dumps(row)+'\n')
    def on_train_epoch_end(self, trainer, pl_module):
        (self.path.parent / 'train_progress.json').write_text(json.dumps({'epoch_completed':int(trainer.current_epoch)+1,'global_step':int(trainer.global_step),'train_batches':int(trainer.num_training_batches)}))


class SaveObjectCallback(Callback):
    """Serialise the world model after every ``epoch_interval`` epochs.

    Replaces upstream's ``SaveCkptCallback``, whose ``save_pretrained`` helper
    does not exist in stable-worldmodel 0.0.5.  The output format is the same
    ``*_object.ckpt`` pickle that ``swm.policy.AutoCostModel`` loads.
    """

    def __init__(self, run_dir: Path, run_name: str, epoch_interval: int = 1, keep_last: int = 2):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.run_name = run_name
        self.epoch_interval = epoch_interval
        self.keep_last = keep_last

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        if epoch % self.epoch_interval and epoch != trainer.max_epochs:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(pl_module.model, self.run_dir / f"{self.run_name}_epoch_{epoch}_object.ckpt")
        self._prune(epoch)

    def _prune(self, epoch: int) -> None:
        """Keep only the most recent few epoch pickles (72 MB each)."""
        kept = {epoch}
        for e in range(epoch, 0, -1):
            if len(kept) >= self.keep_last:
                break
            kept.add(e)
        for path in self.run_dir.glob(f"{self.run_name}_epoch_*_object.ckpt"):
            try:
                e = int(path.stem.split("_epoch_")[1].split("_")[0])
            except (IndexError, ValueError):
                continue
            if e not in kept:
                path.unlink(missing_ok=True)


def build_dataset(cfg):
    """Build the (optionally nuisance-wrapped) training dataset."""
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    name = dataset_cfg.pop("name")
    cache_dir = cfg.get("cache_dir") or os.environ.get("LOCAL_DATASET_DIR")

    spec = NuisanceSpec(**OmegaConf.to_container(cfg.nuisance, resolve=True))
    renderer = NuisanceRenderer(spec) if spec.paints_border else None

    dataset = NuisanceHDF5Dataset(
        name=name, cache_dir=cache_dir, renderer=renderer, **dataset_cfg
    )
    limit_to_episodes(dataset, cfg.get("max_episodes"))
    return dataset, spec


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    # Seed before constructing the dataset, model, or SIGReg module.  The
    # previous campaign passed cfg.seed only to the data split/shuffle and
    # instantiated the model before Manager seeded the process.
    pl.seed_everything(cfg.seed, workers=True)

    #########################
    ##       dataset       ##
    #########################

    dataset, spec = build_dataset(cfg)

    # A padded border enlarges the model input; the resize target must follow it
    # or the Resize would undo the padding and shrink the scene after all.
    input_size = spec.output_size(cfg.img_size)
    print(
        f"[nuisance-train] condition={spec.condition} mode={spec.mode} "
        f"border={spec.border} input_size={input_size} "
        f"border_area={spec.border_area_fraction(cfg.img_size):.3f}"
    )

    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=input_size)
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            transforms.append(get_column_normalizer(dataset, col, col))

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    dataset.post_transform = spt.data.transforms.Compose(*transforms)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    def fingerprint(items):
        digest = hashlib.sha256()
        for key,value in items:
            digest.update(str(key).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    initial_hash = fingerprint(world_model.model.state_dict().items())
    split_hashes = {}
    for label,subset in [('train',train_set),('validation',val_set)]:
        if hasattr(subset,'indices'):
            raw = torch.as_tensor(subset.indices,dtype=torch.int64)
            split_hashes[label] = fingerprint([(label,raw)])
    (run_dir/'reproducibility.json').write_text(json.dumps({
        'seed':int(cfg.seed),'initial_model_sha256':initial_hash,'split_sha256':split_hashes,
        'train_samples':len(train_set),'validation_samples':len(val_set),
        'full_train_batches':len(train),'batch_size':int(cfg.loader.batch_size),
        'loader_generator_state_sha256':fingerprint([('rng',rnd_gen.get_state())]),
    },indent=2))

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            JsonMetricsCallback(run_dir),
            SaveObjectCallback(
                run_dir, cfg.output_model_name, epoch_interval=cfg.get("ckpt_every", 1)
            )
        ],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
        default_root_dir=str(run_dir),
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        seed=cfg.seed,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    trainer.save_checkpoint(ckpt_path)
    print(f"[nuisance-train] condition={spec.condition} run_dir={run_dir}")
    return


if __name__ == "__main__":
    run()
