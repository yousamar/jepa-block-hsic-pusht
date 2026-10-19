# Sources and attribution

- Motivation: Hugues Van Assel, [Feature Selection in World Models: A Spectral View](https://huguesva.github.io/blog/2026/world-models/), especially the JEPA analysis and stable-token PushT stress test. The author's ten-seed results are background, not experiments performed by this project.
- PushT nuisance renderer, dataset wrapping and evaluation plumbing: [huguesva/jepa-stable-token-failure](https://github.com/huguesva/jepa-stable-token-failure), commit `6b866c984da79638db963d1f9a36564fa2c2a1f9`, with local API-port and reproducibility changes. Architecture and SIGReg are inherited from [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm). The original MIT copyright notice is preserved in LICENSE.
- Framework and environments: [galilai-group/stable-worldmodel](https://github.com/galilai-group/stable-worldmodel), pinned to `44c45bd1f774995aac0cc2b41444ebea24c1d5ad`; installed as a dependency, not vendored here.
- Project additions: block-versus-rest differentiable-bandwidth HSIC integration, experiment configuration, portable inference bundles, diagnostic tests and the research note. The four-head/fusion proposals are described in the note and are not implemented in this release.

See SOURCE_PROVENANCE.json for source revisions and source hashes. No claim of ownership over the upstream model, stress test, or SIGReg is made.
