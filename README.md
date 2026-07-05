# SSFT

This repository collects code for supervised fine-tuning (SFT), reinforcement
learning (RL), and data generation. It supports a cyclic training workflow:

1. Fine-tune a base model on an SFT dataset.
2. Continue training the SFT model with RL.
3. Use the RL model to generate an improved SFT dataset.
4. Repeat the cycle with the improved data.

## Submodules

The repository uses three Git submodules:

- [`model_launch/`](model_launch/): [`swiss-ai/model-launch`](https://github.com/swiss-ai/model-launch) on its default branch. It launches and serves models on clariden. The repository scripts use its `sml` tooling and model-serving environment definitions.
- [`verl_rl/`](verl_rl/): [`swiss-ai/verl`](https://github.com/swiss-ai/verl) on the `1p5-async-rl` branch. This verl fork is used for reinforcement learning training.
- [`verl_sft/`](verl_sft/): [`matteosantelmo/verl`](https://github.com/matteosantelmo/verl) on the `apertus-sft` branch. This verl fork is used for supervised fine-tuning.

Clone the repository and initialize all submodules with:

```bash
git clone --recurse-submodules <SSFT_REPOSITORY_URL>
cd SSFT
```

For an existing clone, initialize them with:

```bash
git submodule update --init --recursive
```
