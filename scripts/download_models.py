"""Download a private Release using the authenticated GitHub CLI; verify SHA-256."""
import argparse,hashlib,json,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='yousamar/jepa-block-hsic-pusht')
    p.add_argument('--tag',default='models-lambda200-step32866-v1');p.add_argument('--output',default='models')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    subprocess.run(['gh','release','download',a.tag,'--repo',a.repo,'--dir',str(out),'--pattern','*.pt','--pattern','models.json','--pattern','SHA256SUMS'],check=True)
    manifest=json.loads((out/'models.json').read_text())
    for model in manifest['models']:
        f=out/model['file'];h=hashlib.sha256()
        with f.open('rb') as stream:
            for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
        if h.hexdigest()!=model['sha256']:raise RuntimeError(f'Checksum mismatch: {f}')
        print('VERIFIED',f)

if __name__=='__main__':main()
