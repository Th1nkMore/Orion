#!/usr/bin/env bash
set -euo pipefail

submit=0
if [[ "${1:-}" == "--submit" ]]; then
  submit=1
elif [[ -n "${1:-}" && "${1:-}" != "--dry-run" ]]; then
  echo "usage: $0 [--dry-run|--submit]" >&2
  exit 2
fi

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
python_bin="${PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
carla_root="${CARLA_ROOT:-${asset_root}/carla/CARLA_0.9.15}"
runtime_pythonpath="${carla_root}/PythonAPI:${carla_root}/PythonAPI/carla:${project_root}:${project_root}/Bench2Drive:${project_root}/Bench2Drive/leaderboard:${project_root}/Bench2Drive/scenario_runner:${project_root}/Bench2DriveZoo:${PYTHONPATH:-}"
config="${ORION_TRAIN_CONFIG:-${project_root}/adzoo/orion/configs/orion_stage3_train.py}"
checkpoint="${ORION_CHECKPOINT:-${asset_root}/checkpoints/Orion.pth}"
visual_cache="${VISUAL_CACHE:-${asset_root}/scenario_factory/stage2l_smokes/route196_v1/orion_visual_context.pt}"
records="${QA_RECORDS:-${asset_root}/scenario_factory/qa_factory_smokes/route196_dev_smoke_v3/qa_dataset/records.jsonl}"
protocol="${STAGE2L_TRAINING_PROTOCOL:-${project_root}/configs/scenario_factory/stage2l_training_v5_structured_semantic_bottleneck.json}"
amendment="${LAUNCH_AMENDMENT:-${project_root}/configs/scenario_factory/amendments/20260829_stage2l_route196_structured_semantic_launch_v2.json}"
output_root="${OUTPUT_ROOT:-${asset_root}/scenario_factory/stage2l_smokes/route196_structured_semantic_v1_240/training}"
log_root="${LOG_ROOT:-${asset_root}/logs/stage2l_route196_structured_semantic_v1_240}"

for path in "${python_bin}" "${config}" "${checkpoint}" "${visual_cache}" "${records}" "${protocol}" "${amendment}"; do
  if [[ ! -e "${path}" ]]; then
    echo "missing required input: ${path}" >&2
    exit 1
  fi
done
if [[ -e "${output_root}" ]] && find "${output_root}" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing to overwrite structured semantic smoke output: ${output_root}" >&2
  exit 1
fi

"${python_bin}" - "${project_root}" "${protocol}" "${amendment}" "${output_root}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root, protocol_path, amendment_path, output_root = map(Path, sys.argv[1:])
protocol = json.loads(protocol_path.read_text())
amendment = json.loads(amendment_path.read_text())
key = protocol.get("launch_authorization_key")
locks = amendment.get("launch_locks", {})
authorized = amendment.get("authorized_run", {})
validated = amendment.get("validated_inputs", {})

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

expected = {
    "training_protocol_sha256": sha(protocol_path),
    "trainer_sha256": sha(root / "scripts/train_stage2l_route196_semantic_smoke.py"),
    "bridge_trainer_helpers_sha256": sha(root / "scripts/train_stage2l_route196_bridge_smoke.py"),
    "uq_relevance_bridge_sha256": sha(root / "uq_estimator/uq_relevance_tokenizer.py"),
    "two_pass_runtime_sha256": sha(root / "uq_estimator/stage2l_bridge_runtime.py"),
    "semantic_bottleneck_sha256": sha(root / "uq_estimator/stage2l_semantic_bottleneck.py"),
    "semantic_runtime_sha256": sha(root / "uq_estimator/stage2l_semantic_runtime.py"),
    "submit_wrapper_sha256": sha(root / "scripts/submit_stage2l_route196_semantic_smoke.sh"),
}
if amendment.get("schema") != "orion.scenario_factory.amendment.v1":
    raise SystemExit("invalid structured semantic launch amendment schema")
if locks.get(key) is not True or locks.get("stage2l_pilot_training_allowed") is not False:
    raise SystemExit("structured semantic launch is inactive while the pilot remains locked")
if locks.get("stage2p_allowed") is not False:
    raise SystemExit("structured semantic smoke may not authorize Stage2-P")
if int(authorized.get("maximum_submissions", 0)) != 1:
    raise SystemExit("structured semantic amendment is not single-run")
if authorized.get("fresh_initialization_from_original_orion_checkpoint") is not True:
    raise SystemExit("structured semantic smoke must start from original ORION")
if Path(authorized.get("output_root", "")).resolve() != output_root.resolve():
    raise SystemExit("structured semantic output root differs from amendment")
if any(validated.get(name) != value for name, value in expected.items()):
    raise SystemExit("structured semantic source/protocol hash differs from amendment")
print("STRUCTURED_SEMANTIC_SINGLE_RUN_AUTHORIZED=1")
PY

train_parts=(
  env "PYTHONPATH=${runtime_pythonpath}" "IS_BENCH2DRIVE=True"
  "${python_bin}" "${project_root}/scripts/train_stage2l_route196_semantic_smoke.py"
  --config "${config}"
  --checkpoint "${checkpoint}"
  --visual-cache "${visual_cache}"
  --records "${records}"
  --training-protocol "${protocol}"
  --launch-amendment "${amendment}"
  --output-dir "${output_root}"
  --max-steps "${MAX_STEPS:-240}"
)
printf -v train_command '%q ' "${train_parts[@]}"
sbatch_args=(
  sbatch --parsable
  "--partition=${SLURM_PARTITION:-Nvidia_A800}"
  --gres=gpu:1 "--cpus-per-task=${STAGE2L_CPUS:-2}" --mem=192G --time=03:00:00
  "--exclude=${SLURM_EXCLUDE:-gpu5}"
  --job-name=s2l_r196_semantic
  "--output=${log_root}/train-%j.out"
  --export=ALL --wrap "${train_command}"
)

if [[ "${submit}" != "1" ]]; then
  echo "DRY_RUN_ONLY=1"
  echo "STAGE2L_PILOT_LOCKED=1"
  echo "STAGE2P_LOCKED=1"
  printf 'SBATCH_COMMAND='
  printf '%q ' "${sbatch_args[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${log_root}"
job_id="$("${sbatch_args[@]}")"
echo "STAGE2L_ROUTE196_STRUCTURED_SEMANTIC_JOB_ID=${job_id}"
echo "OUTPUT_ROOT=${output_root}"
