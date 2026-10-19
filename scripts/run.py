"""Portable training/evaluation commands. One process, one selected GPU."""
import argparse,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','eval'])
    p.add_argument('--condition',choices=['stable','grey','unpredictable'],required=True)
    p.add_argument('--data',required=True,help='Directory containing pusht_expert_train.h5')
    p.add_argument('--output',required=True);p.add_argument('--lambda-hsic',type=float,default=200)
    p.add_argument('--seed',type=int);p.add_argument('--checkpoint');p.add_argument('--episodes',type=int,default=50)
    p.add_argument('--workers',type=int,default=16);p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--device',default='cuda');p.add_argument('--max-steps',type=int)
    a=p.parse_args();out=Path(a.output).resolve();data=Path(a.data).resolve()
    condition='neutral' if a.condition=='grey' else 'markov';rho=0 if a.condition=='unpredictable' else 1
    nuisance=[f'nuisance.condition={condition}','nuisance.mode=pad','nuisance.geometry=patch','nuisance.border=14',
              'nuisance.patch_size=14','nuisance.n_patches=1','nuisance.amplitude=1',f'nuisance.rho={rho}',
              'nuisance.frame_unit=model','nuisance.base_seed=20260922']
    env=os.environ.copy();env.setdefault('MUJOCO_GL','egl');env.setdefault('WANDB_MODE','disabled')
    env.setdefault('TOKENIZERS_PARALLELISM','false');env.setdefault('MPLBACKEND','Agg')
    if a.mode=='train':
        tag=f'{a.condition}_lambda{a.lambda_hsic:g}_seed{a.seed or 4711}'
        cmd=[sys.executable,str(ROOT/'train.py'),f'seed={a.seed if a.seed is not None else 4711}',
             f'cache_dir={data}',f'run_dir={out}',f'output_model_name={tag}',f'loss.dep.weight={a.lambda_hsic}',
             f'trainer.max_epochs={a.epochs}','trainer.devices=1',f'num_workers={a.workers}',*nuisance]
        if a.device=='cpu':cmd+=['trainer.accelerator=cpu','trainer.precision=32-true']
        if a.workers==0:cmd+=['loader.persistent_workers=false','loader.prefetch_factor=null']
        if a.max_steps is not None:cmd+=[f'+trainer.max_steps={a.max_steps}']
    else:
        if not a.checkpoint:p.error('--checkpoint is required for eval')
        path=Path(a.checkpoint).resolve();policy=str(path)
        if policy.endswith('_object.ckpt'):policy=policy[:-len('_object.ckpt')]
        tag=f'{a.condition}_eval{a.seed if a.seed is not None else 42}'
        cmd=[sys.executable,str(ROOT/'eval.py'),f'policy={policy}',f'cache_dir={data}',
             f'seed={a.seed if a.seed is not None else 42}',f'solver.device={a.device}',f'eval.num_eval={a.episodes}',
             'nuisance.stream=1','eval.goal_nuisance=matched',f'output.dir={out}',f'output.tag={tag}',f'output.filename={tag}.txt',*nuisance]
    subprocess.run(cmd,cwd=ROOT,env=env,check=True)

if __name__=='__main__':main()
