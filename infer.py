"""Encode/predict preprocessed inputs, or check the interface without a dataset."""
import argparse
from pathlib import Path
import numpy as np
import torch
from model_io import load_inference_model

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--input')
    p.add_argument('--output',default='prediction.npz');p.add_argument('--device',default='cpu');p.add_argument('--self-test',action='store_true')
    args=p.parse_args();torch.set_num_threads(2);model=load_inference_model(args.checkpoint,args.device)
    if args.self_test:
        gen=torch.Generator().manual_seed(42)
        arrays={'pixels':torch.randn(1,3,3,252,252,generator=gen),'action':torch.randn(1,3,10,generator=gen)}
    elif args.input:
        with np.load(args.input,allow_pickle=False) as z:
            arrays={k:torch.from_numpy(z[k]) for k in ['pixels','action']}
    else:p.error('Supply --input or --self-test')
    info={k:v.to(args.device) for k,v in arrays.items()}
    with torch.inference_mode():
        encoded=model.encode(info);prediction=model.predict(encoded['emb'],encoded['act_emb'])
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,embedding=encoded['emb'].cpu().numpy(),prediction=prediction.cpu().numpy())
    print({'embedding_shape':list(encoded['emb'].shape),'prediction_shape':list(prediction.shape),'output':str(out)})
    if args.self_test:print('Synthetic interface check only; not a planning or accuracy evaluation.')

if __name__=='__main__':main()
