#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
run_id="${RUN_ID:-stage1_u_tokenizer_task_agnostic_v1_200}"
output_dir="${asset_root}/scenario_factory/stage1_u_tokenizer_pretraining/${run_id}"
log_dir="${asset_root}/scenario_factory/stage1_u_tokenizer_pretraining/logs"
protocol="${project_root}/configs/scenario_factory/stage1_u_tokenizer_pretraining_v1.json"
source_audit="${asset_root}/scenario_factory/stage1_u_tokenizer_preflight_v1/source_audit.json"
python_bin="${asset_root}/envs/orion-cl-centos7/bin/python"

if [[ -e "${output_dir}" ]]; then
  echo "[FAIL] refusing to reuse output directory: ${output_dir}" >&2
  exit 1
fi
mkdir -p "${log_dir}"

cd "${project_root}"
printf '%s  %s\n' \
  "f3150e424a900ba48a7ef83aea94dfb25ae78b9f90391714e5ee611c247543af" "${source_audit}" \
  "874fe266f41aaf1dca0b9b8d0c2478705b695ff2104960c89a5d621be10578fb" "scripts/audit_stage1_u_tokenizer_pretraining_sources.py" \
  "23f69314b6881937a5da90b370991d39b09282ff1f5ccbdeb15a303442b1d4de" "tests/test_audit_stage1_u_tokenizer_pretraining_sources.py" \
  "be8a3733e6500d06e248920a3bcbe04520e33b34c778c21a796ed337caa657e6" "uq_estimator/stage1_u_tokenizer_pretraining.py" \
  "4aaf43ec33b6e95cf564607adb2bc26c77086014a89a44ab7c0526a590fc519e" "tests/test_stage1_u_tokenizer_pretraining.py" \
  "4d77b7f8291afd755ef337853dd7cb4d9abcfd2e02948357f30b1b34788ed2d2" "scripts/train_stage1_u_tokenizer_pretraining.py" \
  "91957667706335662b0f27b2cb9eab92914ae2cc324eb327f3bd0fd15560d217" "tests/test_train_stage1_u_tokenizer_pretraining.py" \
  | sha256sum --check --strict

protocol_sha256="$(sha256sum "${protocol}" | awk '{print $1}')"
job_id="$(sbatch --parsable \
  --partition=Nvidia_A800 \
  --gres=gpu:1 \
  --cpus-per-task=2 \
  --mem=16G \
  --time=00:20:00 \
  --job-name=stage1_u_tok_v1 \
  --output="${log_dir}/stage1_u_tok_v1-%j.out" \
  --export=ALL,PROJECT_ROOT="${project_root}",ASSET_ROOT="${asset_root}" \
  --wrap="cd '${project_root}' && PYTHONPATH='${project_root}' '${python_bin}' scripts/train_stage1_u_tokenizer_pretraining.py --protocol '${protocol}' --source-audit '${source_audit}' --output-dir '${output_dir}' --steps 200 --batch-size 4 --seed 20260831")"

attestation="${log_dir}/stage1_u_tok_v1-${job_id}.submission.json"
"${python_bin}" - "${attestation}" "${job_id}" "${protocol}" "${protocol_sha256}" "${source_audit}" "${output_dir}" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
path, job_id, protocol, protocol_sha, audit, output = sys.argv[1:]
payload = {
    "schema": "orion.stage1_u_tokenizer_submission.v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "job_id": job_id,
    "scope": "one bounded task-agnostic U-tokenizer representation pretraining",
    "protocol": {"path": protocol, "sha256": protocol_sha},
    "source_audit": audit,
    "output_dir": output,
    "resources": {"partition": "Nvidia_A800", "gpu": 1, "cpu": 2, "memory": "16G", "time": "00:20:00"},
    "orion_weights_loaded": False,
    "stage2l_training": False,
    "formal_training": False,
    "automatic_retry_allowed": False,
}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
echo "${job_id}"
