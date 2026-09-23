# OpenVul SFT on Qwen3.5-4B

Use OpenVul rejection-SFT data and OpenVul SFT hyperparameters to fine-tune `Qwen/Qwen3.5-4B`.

This is not the original OpenVul backbone (`Qwen3-4B`). Paths are not machine-specific.

## Quick start

```bash
git clone https://github.com/MuseMamba/openvul-qwen3-5-4b-sft.git
cd openvul-qwen3-5-4b-sft
chmod +x run_openvul_sft_qwen35.sh

export OPENVUL_WORKSPACE=/large-disk/openvul
export HF_TOKEN=...
mkdir -p "$OPENVUL_WORKSPACE"

bash run_openvul_sft_qwen35.sh disk
bash run_openvul_sft_qwen35.sh all
```

## Commands

| command | what |
|---|---|
| `disk` | show free space and current workspace disk |
| `deps` | install git/curl/python3/uv if missing |
| `setup` | deps + clone OpenVul + Python 3.11.13 venv |
| `download` | rejection SFT dataset + Qwen3.5-4B |
| `clean` | convert prompt/completion to Qwen3.5 chat `messages` |
| `train` | SFT with OpenVul `sft.sh` hparams |
| `all` | setup, download, clean, train |

Disk gates: setup ~20G, download ~40G, train/all ~150G. Override: `FORCE_DISK=1`.

## Layout under `$OPENVUL_WORKSPACE`

```
code/OpenVul
data/sft_reject
data/sft_qwen35
models/Qwen3.5-4B
.cache/huggingface
.cache/uv
code/OpenVul/outputs/Qwen3.5-4B-OpenVul-SFT
```

Optional: `OPEN_VUL_CODE`, `OPEN_VUL_DATA`, `OPEN_VUL_MODELS`.

## Hyperparameters

lr `1e-5`, linear, warmup `0.1`, weight decay `0.01`, 5 epochs, per-device batch 1, grad acc 8, max length 32768.

OOM: `export MAX_LEN=16384`

## Sources

- OpenVul code: https://github.com/youpengl/OpenVul
- SFT data: `Leopo1d/OpenVul_Rejection_Sampling_based_Vulnerability_Reasoning_Dataset_for_SFT`
- Paper: arXiv:2602.14012
