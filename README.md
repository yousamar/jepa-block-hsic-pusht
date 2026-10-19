# JEPA 分块 HSIC：PushT 训练与推理

这个项目探索：在 LeWorldModel 中增加块间 HSIC，能否减少重复信息，为其他因素保留表达空间。动机来自 Hugues Van Assel 的 [JEPA 分析与稳定色块实验](https://huguesva.github.io/blog/2026/world-models/)。

**当前实现是共享投影后直接分块的 HSIC。四个独立头、融合层和 SIGReg 前后分配是尚未验证的新方案，只在方法文档中介绍。** 已有实验并未取得稳定的规划改善，HSIC 也可能压掉有用信息。

- [思路、方法与新方案（中文）](docs/idea-and-method.zh-CN.md)
- [模型卡与训练来源](docs/model-card.md)
- [测试与导出核验](verification/README.md)
- [上游来源与许可证](THIRD_PARTY.md)

## 安装

建议使用 Linux、Python 3.10 和支持 CUDA 的 PyTorch。CPU 可用于测试、加载模型和小规模接口检查。

```bash
git clone https://github.com/yousamar/jepa-block-hsic-pusht.git
cd jepa-block-hsic-pusht
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

框架版本固定在 `requirements.txt`；导出核验时的核心版本见 `requirements-tested.txt`。已在原实验依赖环境中测试独立代码目录，尚未重新安装一套全新环境。运行环境可能需要系统编译工具和 EGL/OpenGL 支持。

## 下载数据

```bash
export STABLEWM_HOME=/path/to/lewm-data
bash scripts/download_data.sh
```

脚本下载[官方 PushT 数据](https://huggingface.co/datasets/quentinll/lewm-pusht)，解压需要安装 `zstd`。仓库不包含数据集。

## 训练

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run.py train \
  --condition stable --lambda-hsic 200 --seed 4711 \
  --data "$STABLEWM_HOME" --output outputs/stable_lambda200
```

- `--lambda-hsic 0`：原预测＋SIGReg 基线；`200`：增加 HSIC。
- `--condition`：`stable` 稳定色块、`grey` 灰背景、`unpredictable` 逐帧随机色块。
- 默认：2轮、batch128、BF16、192维分为四个48维块、SIGReg 权重0.09、AdamW 学习率5e-5。
- 数据加载默认16 workers，可用 `--workers` 按主机内存调整；`--workers 0` 可关闭多进程加载。

训练保存 `*_epoch_2_object.ckpt` 供推理使用，`*_weights.ckpt` 用于本地恢复训练。`--max-steps 20` 只适合检查流程，不能评价方法效果，其学习率日程也不等同完整训练。

## 下载模型与检查推理接口

**模型附件正在同步，尚未上线。** 三个文件均已导出并通过等价性核验，计划放在私有 Release `models-lambda200-step32866-v1`。上线后，拥有仓库访问权限并登录 [GitHub CLI](https://cli.github.com/) 即可下载：

```bash
python scripts/download_models.py
python infer.py --checkpoint models/stable_lambda200_step32866.pt \
  --self-test --output outputs/interface_check.npz
```

每个模型约72 MB、18,034,478个参数，FP32，不含优化器。三个文件分别对应三个背景条件；步数均为32,866，详细来源和限制见模型卡。

`--self-test` 使用合成输入，只检查接口。自己的 `.npz` 输入应包含 `pixels`（形状 `[B,T,3,252,252]`，完成训练时的 ImageNet 预处理）和 `action`（形状 `[B,T,10]`，完成数据集标准化与动作分组）。使用 `--input` 读取，输出包含编码与预测。原始环境图像的处理交给下面的规划评估入口。

## 闭环规划评估

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run.py eval \
  --condition stable --checkpoint models/stable_lambda200_step32866.pt \
  --seed 42 --episodes 50 --data "$STABLEWM_HOME" --output outputs/eval_stable_42
```

换用 seed4242 再评估50集；更换模型时同时选择匹配的背景条件。脚本保持原 CEM 配置、单色块几何及观测/目标颜色配对，保存每集成败、起点、目标和规划配置。也支持原来的可信 `*_object.ckpt` 文件。

## 已有结果

以下为同预算27,866步的 λ0/200 比较，每模型100集：

| 背景 | λ0 | λ200 |
|---|---:|---:|
| 稳定色块 | 4% | 9% |
| 灰背景 | 82% | 19% |
| 逐帧随机色块 | 67% | 10% |

完整视觉实验只有一个训练种子，不能据此宣称稳健提升。Release 中的32,866步模型是另一终点，结果与来源另见模型卡；不要把两个终点混为同预算比较。

## 来源

模型和 SIGReg 来自 [LeWorldModel](https://github.com/lucas-maes/le-wm)，PushT 干预与评估代码基于 [jepa-stable-token-failure](https://github.com/huguesva/jepa-stable-token-failure)。保留 MIT 许可证，版本与来源记录见 `SOURCE_PROVENANCE.json`。博客作者的十种子实验不属于本项目的训练重复。
