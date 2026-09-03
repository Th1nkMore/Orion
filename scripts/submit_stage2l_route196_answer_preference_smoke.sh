#!/usr/bin/env bash
set -euo pipefail

# Prepared objective-repair smoke.  It remains fail-closed until its launch
# amendment is activated after code tests and dry-run validation.
project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"

export PROJECT_ROOT="${project_root}"
export ASSET_ROOT="${asset_root}"
export BALANCE_DRIVING_STANCES=1
export TRAINING_PROTOCOL="${project_root}/configs/scenario_factory/stage2l_training_v3_answer_preference.json"
export LAUNCH_AMENDMENT="${project_root}/configs/scenario_factory/amendments/20260829_stage2l_route196_answer_preference_launch_v2.json"
export LAMBDA_ANSWER_PREFERENCE=2.0
export ANSWER_PREFERENCE_MARGIN=0.2
export MAX_STEPS=240
export EXISTING_VISUAL_CACHE="${asset_root}/scenario_factory/stage2l_smokes/route196_v1/orion_visual_context.pt"
export RUN_ROOT="${asset_root}/scenario_factory/stage2l_smokes/route196_answer_preference_v1_240"
export LOG_ROOT="${asset_root}/logs/stage2l_route196_answer_preference_v1_240"

exec "${project_root}/scripts/submit_stage2l_route196_overfit_smoke.sh" "$@"
