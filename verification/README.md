# Verification scope

- 11 CPU tests passed: lambda0 exact forward/gradient/RNG equivalence, HSIC gradient and scale checks, nuisance controls, portable save/load and format rejection.
- All three FP32 inference exports exactly match the original state tensors, image/action embeddings, prediction, autoregressive rollout, planning cost and a small seeded CEM solve.
- Published Python/YAML source hashes are recorded with normalized LF newlines. The final training-config header correction was checked to preserve the parsed configuration.
- Training and evaluation Hydra configs both load. Dependencies are the original working environment; a fresh pip installation was not performed. The original uv-managed runtime has no pip module and was not modified.
- No new training, full planning evaluation or four-head experiment was started for this publication.

All three model files were downloaded and verified against their exported SHA-256 hashes. Release uploads also check GitHub-provided SHA-256 digests when available; see the model Release and its checksum manifest.
