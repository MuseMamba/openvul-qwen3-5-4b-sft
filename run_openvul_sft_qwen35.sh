#!/usr/bin/env bash
# OpenVul to Qwen3.5-4B SFT. Workspace: OPENVUL_WORKSPACE or cwd.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:-all}"

if [[ -n "${OPENVUL_WORKSPACE:-}" ]]; then
  ROOT="${OPENVUL_WORKSPACE}"
else
  ROOT="$(pwd)"
fi
if [[ -d "$ROOT" ]]; then
  ROOT="$(cd "$ROOT" && pwd)"
fi

CODE="${OPEN_VUL_CODE:-$ROOT/code/OpenVul}"
DATA="${OPEN_VUL_DATA:-$ROOT/data}"
MODELS="${OPEN_VUL_MODELS:-$ROOT/models}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.cache/uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/.cache}"
CLEAN="${DATA}/sft_qwen35"
HF_SFT_DIR="${DATA}/sft_reject"
PREPARE_PY="${PREPARE_PY:-$SCRIPT_DIR/prepare_sft_qwen35.py}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-4B}"
SFT_REPO="${SFT_REPO:-Leopo1d/OpenVul_Rejection_Sampling_based_Vulnerability_Reasoning_Dataset_for_SFT}"
OUT_NAME="${OUT_NAME:-Qwen3.5-4B-OpenVul-SFT}"
MAX_LEN="${MAX_LEN:-32768}"
NUM_GPUS="${NUM_GPUS:-4}"
LR="${LR:-1e-5}"
WARMUP="${WARMUP:-0.1}"
WD="${WD:-0.01}"
EPOCHS="${EPOCHS:-5}"
GA="${GA:-8}"
PER_DEV_BS="${PER_DEV_BS:-1}"

NEED_SETUP_GB="${NEED_SETUP_GB:-20}"
NEED_DOWNLOAD_GB="${NEED_DOWNLOAD_GB:-40}"
NEED_TRAIN_GB="${NEED_TRAIN_GB:-150}"

log() { printf '\n==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

existing_parent() {
  local p="$1"
  while [[ ! -e "$p" && "$p" != / ]]; do
    p="$(dirname "$p")"
  done
  echo "$p"
}

disk_avail_gb() {
  local p
  p="$(existing_parent "$1")"
  df -Pk "$p" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1024/1024}'
}

disk_mount() {
  local p
  p="$(existing_parent "$1")"
  df -Pk "$p" 2>/dev/null | awk 'NR==2 {print $6}'
}

print_disk_map() {
  echo
  echo "Available disks:"
  df -hT -x tmpfs -x devtmpfs -x squashfs 2>/dev/null || df -h
  echo
  echo "Put workspace on a disk with >= 150G free:"
  echo "  export OPENVUL_WORKSPACE=/large-disk/openvul"
  echo "  mkdir -p \"\$OPENVUL_WORKSPACE\""
  echo "  bash $0 disk"
}

print_paths() {
  local avail
  avail="$(disk_avail_gb "$ROOT")"
  cat <<ENDPATHS
Paths
  WORKSPACE     $ROOT
                export OPENVUL_WORKSPACE=/abs/path
  mount         $(disk_mount "$ROOT")
  free          ${avail:-?} GB
  CODE          $CODE
  DATA          $DATA
  MODELS        $MODELS
  HF_HOME       $HF_HOME
  UV_CACHE_DIR  $UV_CACHE_DIR
  PREPARE_PY    $PREPARE_PY
Estimate        setup ~${NEED_SETUP_GB}G ; after download ~${NEED_DOWNLOAD_GB}G ; after train ~${NEED_TRAIN_GB}G
ENDPATHS
}

require_disk() {
  local need="$1"
  local phase="$2"
  local avail
  avail="$(disk_avail_gb "$ROOT")"
  if [[ -z "$avail" ]]; then
    echo "WARN: cannot read free disk"
    print_disk_map
    return 0
  fi
  log "disk [$phase] need ~${need}GB, workspace disk has ${avail}GB free"
  if (( avail < need )); then
    print_disk_map
    echo
    echo "Not enough space: WORKSPACE=$ROOT free ${avail}GB < ${need}GB"
    echo "export OPENVUL_WORKSPACE=/large-disk/openvul && mkdir -p \"\$OPENVUL_WORKSPACE\" && bash $0 $ACTION"
    echo "Force: FORCE_DISK=1 bash $0 $ACTION"
    if [[ "${FORCE_DISK:-0}" == "1" ]]; then
      echo "WARN: FORCE_DISK=1, continue"
      return 0
    fi
    exit 1
  fi
}

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

have_sudo() {
  command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null
}

apt_install() {
  if command -v apt-get >/dev/null 2>&1; then
    if have_sudo; then
      sudo apt-get update -y
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$@"
    elif [[ "$(id -u)" -eq 0 ]]; then
      apt-get update -y
      DEBIAN_FRONTEND=noninteractive apt-get install -y "$@"
    else
      echo "WARN: no sudo, cannot apt install: $*"
      return 1
    fi
  else
    return 1
  fi
}

ensure_uv() {
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  if command -v uv >/dev/null 2>&1; then
    echo "uv: $(command -v uv) $(uv --version 2>/dev/null || true)"
    return 0
  fi
  log "installing uv"
  if ! command -v curl >/dev/null 2>&1; then
    apt_install curl ca-certificates || die "curl missing, cannot install uv"
  fi
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  hash -r || true
  command -v uv >/dev/null 2>&1 || die "uv not on PATH. export PATH=\"\$HOME/.local/bin:\$PATH\""
  echo "uv: $(uv --version)"
}

ensure_host_deps() {
  log "host dependencies"
  local missing=()
  command -v git >/dev/null 2>&1 || missing+=(git)
  command -v curl >/dev/null 2>&1 || missing+=(curl ca-certificates)
  command -v python3 >/dev/null 2>&1 || missing+=(python3)
  python3 -c "import venv" 2>/dev/null || missing+=(python3-venv)
  if ((${#missing[@]})); then
    echo "missing: ${missing[*]}"
    apt_install "${missing[@]}" || die "install manually: ${missing[*]}"
  fi
  echo "git=$(command -v git) python3=$(command -v python3) curl=$(command -v curl)"
  ensure_uv
}

activate_env() {
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  if [[ -f "${CODE}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${CODE}/.venv/bin/activate"
  elif [[ -f "${ROOT}/env/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT}/env/bin/activate"
  else
    die "no venv. run: bash $0 setup"
  fi
}

cmd_setup() {
  print_paths
  require_disk "$NEED_SETUP_GB" "setup"
  ensure_host_deps
  mkdir -p "$CODE" "$DATA" "$MODELS" "$CLEAN"
  if [[ ! -d "${CODE}/.git" ]]; then
    log "clone OpenVul -> $CODE"
    git clone https://github.com/youpengl/OpenVul.git "$CODE"
  else
    log "OpenVul exists: $CODE"
  fi
  cd "$CODE"
  log "Python 3.11.13 venv"
  uv python install 3.11.13 || true
  if [[ ! -f "${CODE}/.venv/bin/activate" ]]; then
    uv venv --python 3.11.13 || python3 -m venv "${CODE}/.venv"
  fi
  # shellcheck disable=SC1091
  source "${CODE}/.venv/bin/activate"
  uv pip install -r requirements.txt
  uv pip install -U datasets huggingface_hub transformers accelerate
  uv pip install flash-attn==2.8.1 --no-build-isolation || echo "WARN: flash-attn failed, train will use sdpa"
  echo "setup ok"
}

cmd_download() {
  print_paths
  require_disk "$NEED_DOWNLOAD_GB" "download"
  activate_env
  mkdir -p "$HF_SFT_DIR" "$MODELS"
  log "download SFT data $SFT_REPO -> $HF_SFT_DIR"
  python3 - <<PY
from datasets import load_dataset
from pathlib import Path
dest = Path(r"${HF_SFT_DIR}")
dest.mkdir(parents=True, exist_ok=True)
if (dest / "dataset_dict.json").exists() or (dest / "state.json").exists():
    print("skip existing", dest)
else:
    ds = load_dataset("${SFT_REPO}")
    ds.save_to_disk(str(dest))
    print(ds)
PY
  local dest="${MODELS}/Qwen3.5-4B"
  log "download ${MODEL_ID} -> ${dest}"
  if [[ -f "${dest}/config.json" ]]; then
    echo "skip $dest"
  else
    if command -v hf >/dev/null 2>&1; then
      hf download "${MODEL_ID}" --local-dir "$dest"
    else
      huggingface-cli download "${MODEL_ID}" --local-dir "$dest"
    fi
  fi
}

cmd_clean() {
  print_paths
  activate_env
  [[ -d "$HF_SFT_DIR" ]] || die "missing SFT data: $HF_SFT_DIR"
  [[ -f "$PREPARE_PY" ]] || die "missing $PREPARE_PY"
  local tok="${MODELS}/Qwen3.5-4B"
  [[ -f "${tok}/config.json" ]] || tok="${MODEL_ID}"
  log "clean -> $CLEAN"
  python3 "$PREPARE_PY" --src "$HF_SFT_DIR" --dst "$CLEAN" --tokenizer "$tok" --max-length "$MAX_LEN"
}

cmd_train() {
  print_paths
  require_disk "$NEED_TRAIN_GB" "train"
  activate_env
  [[ -d "${CLEAN}/hf" ]] || die "missing cleaned data: ${CLEAN}/hf"
  local model="${MODELS}/Qwen3.5-4B"
  [[ -f "${model}/config.json" ]] || model="${MODEL_ID}"
  cd "$CODE"
  mkdir -p "outputs/${OUT_NAME}"
  local n_gpu
  n_gpu="$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)"
  echo "GPUs=${n_gpu} NUM_GPUS=${NUM_GPUS}"
  [[ "${n_gpu}" -ge 1 ]] || die "no GPU"
  local attn="flash_attention_2"
  python3 -c "import flash_attn" 2>/dev/null || attn="sdpa"
  local acc_cfg="examples/accelerate_configs/sft_zero3.yaml"
  [[ -f "$acc_cfg" ]] || acc_cfg=""
  log "SFT model=$model epochs=$EPOCHS lr=$LR max_len=$MAX_LEN attn=$attn"
  local cmd=(
    trl/scripts/sft.py
    --model_name_or_path "${model}"
    --run_name "${OUT_NAME}"
    --output_dir "outputs/${OUT_NAME}"
    --dataset_name "${CLEAN}/hf"
    --learning_rate "${LR}"
    --lr_scheduler_type linear
    --warmup_ratio "${WARMUP}"
    --weight_decay "${WD}"
    --num_train_epochs "${EPOCHS}"
    --bf16 true
    --torch_dtype bfloat16
    --gradient_checkpointing
    --attn_implementation "${attn}"
    --max_length "${MAX_LEN}"
    --per_device_train_batch_size "${PER_DEV_BS}"
    --per_device_eval_batch_size "${PER_DEV_BS}"
    --gradient_accumulation_steps "${GA}"
    --eval_strategy no
    --save_strategy epoch
    --logging_steps 1
    --log_level info
    --dataset_train_split train
    --report_to "${REPORT_TO:-none}"
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
    --trust_remote_code true
  )
  if [[ -n "$acc_cfg" ]]; then
    accelerate launch --config_file "$acc_cfg" "${cmd[@]}"
  else
    accelerate launch --num_processes "${NUM_GPUS}" "${cmd[@]}"
  fi
}

cmd_all() {
  print_paths
  require_disk "$NEED_TRAIN_GB" "all"
  cmd_setup
  cmd_download
  cmd_clean
  cmd_train
}

cmd_disk() {
  print_paths
  print_disk_map
}

cmd_help() {
  cat <<ENDHELP
Usage:
  export OPENVUL_WORKSPACE=/large-disk/openvul
  bash run_openvul_sft_qwen35.sh disk
  bash run_openvul_sft_qwen35.sh all

Commands: disk | paths | deps | setup | download | clean | train | all
Need ~20G setup, ~40G download, ~150G train. Force: FORCE_DISK=1
ENDHELP
}

case "$ACTION" in
  help|-h|--help) cmd_help ;;
  paths) print_paths ;;
  disk) cmd_disk ;;
  deps) ensure_host_deps ;;
  setup) cmd_setup ;;
  download) cmd_download ;;
  clean) cmd_clean ;;
  train) cmd_train ;;
  all) cmd_all ;;
  *) die "unknown command $ACTION" ;;
esac
