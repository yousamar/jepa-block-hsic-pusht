# λ200 PushT models: 32,866 updates

## Contents

Three inference-only FP32 state dictionaries: stable, grey, unpredictable. Each includes a resolved model architecture and minimal provenance metadata. There is no optimizer, scheduler, dataset or training RNG state. The bundles are exported and verified on the experiment server; transfer to GitHub Release is pending. Exact sizes, SHA-256 hashes and verification results are already recorded in `verification/export-validation.json`; the eventual Release will also include `models.json` and `SHA256SUMS`.

The architecture is the existing ViT-Tiny LeWorldModel with a shared projection MLP, 192 dimensions, four fixed48-dimensional HSIC blocks, action encoder and autoregressive predictor. **It is not the proposed four-head/fusion architecture.** Total parameters:18,034,478. HSIC has no parameters and is absent during inference.

## Training

- Initial training: seed4711,27,866 updates over2 epochs, batch128, BF16, AdamW, peak lr5e-5, weight decay0.001, SIGReg0.09, differentiable-bandwidth HSIC λ200.
- Extension: precisely5000 extra updates; model and AdamW moments retained; learning rate restarted with50 warmup updates and cosine annealing to zero. New fixed stage RNG seed32577; no exact original RNG continuation is claimed.
- Observation geometry:224×224 task image padded to252×252, one14×14 corner token, base nuisance seed20260922. Stable color has rho1, unpredictable color rho0, grey is neutral128.

The local continuation files are retained unchanged. The inference bundles are new exports, not replacements for the originals. Their metadata records original object-checkpoint hashes, not machine-specific paths.

## Intended use and limitations

Use for reproducing and inspecting these exploratory results, not as a high-performing PushT policy. At100 freshly paired planning episodes/model, successes after extension were8/100 stable,19/100 grey,11/100 unpredictable. These models do not establish robust improvement. The two evaluation seeds are not independent world-model training seeds.

Use a matching condition during evaluation. The observation/goal color realization must be matched, actions normalized with the dataset statistics, and images preprocessed as in training. A model file alone does not include the action scaler: the evaluation pipeline fits it from the official training dataset, matching the original protocol.

Export verification compares all state tensors, image encoding, action-conditioned prediction, autoregressive rollout, planning costs and a small seeded CEM solve against the original object. This checks serialization equivalence, not renewed statistical efficacy. Full600-episode historical results are summarized separately; export verification does not claim to rerun them.

## Sources

[Blog motivation](https://huguesva.github.io/blog/2026/world-models/), [upstream stress test](https://github.com/huguesva/jepa-stable-token-failure), [LeWorldModel](https://github.com/lucas-maes/le-wm). See THIRD_PARTY.md and the retained MIT license.
