#!/usr/bin/env bash
set -euo pipefail

export DATASET=DNS
export EXP_NAME=bert_base_12l_MLM_train_run

EXPDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Project root: $EXPDIR"

export EXPDIR
export XDG_CACHE_HOME="${EXPDIR}/.cache"
export CONDA_PKGS_DIRS="${EXPDIR}/.conda_pkgs"
export CONDA_ENVS_PATH="${EXPDIR}/.conda_envs"

if [ ! -f "${EXPDIR}/environment.yml" ]; then
  echo "ERROR: environment.yml not found in ${EXPDIR}" >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

export MKL_INTERFACE_LAYER=GNU

ENV_PREFIX="${EXPDIR}/.conda_env"
if [ -d "${ENV_PREFIX}" ]; then
  echo "Updating env at ${ENV_PREFIX}..."
  conda env update -p "${ENV_PREFIX}" \
    -f "${EXPDIR}/environment.yml" --prune
else
  echo "Creating env at ${ENV_PREFIX}..."
  conda env create -p "${ENV_PREFIX}" \
    -f "${EXPDIR}/environment.yml"
fi

conda activate "${ENV_PREFIX}"
pip install -e "${EXPDIR}"

exec python "${EXPDIR}/src/training/bert/bert_bpe_8k_regular_dataset_train.py"

echo "Training completed successfully. Artifacts are saved in /out/${EXP_NAME}/.."