#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/public/home/lidachuan/project/Orion}"
asset_root="${ASSET_ROOT:-/public/share/lidachuan/orion_assets}"
python_bin="${PYTHON_BIN:-${asset_root}/envs/orion-cl-centos7/bin/python}"
scenario_root="${asset_root}/scenario_factory"
output="${OUTPUT:-${scenario_root}/stage2l_formal_inventory17_audit_20260831/formal_inventory_audit.json}"

reports=(
  "${scenario_root}/stage2l_formal_v5_qa_smokes/route147_step223_v1/multiframe_event_factory_report.json"
  "${scenario_root}/stage2l_formal_v5_qa_wave1_20260831/route151_step218/multiframe_event_factory_report.json"
  "${scenario_root}/stage2l_formal_v5_qa_wave1_20260831/route152_step125/multiframe_event_factory_report.json"
  "${scenario_root}/stage2l_formal_v5_qa_wave2_20260831/route157_step193/multiframe_event_factory_report.json"
  "${scenario_root}/stage2l_formal_v5_qa_wave2_20260831/route158_step154/multiframe_event_factory_report.json"
  "${scenario_root}/stage2l_formal_v5_qa_wave2_20260831/route160_step518/multiframe_event_factory_report.json"
  "${scenario_root}/stage2l_formal_v5_qa_wave2_20260831/route161_step308/multiframe_event_factory_report.json"
  "${scenario_root}/stage2l_formal_v5_qa_wave1_20260831/route162_step277/multiframe_event_factory_report.json"
  "${scenario_root}/stage2l_formal_v5_qa_wave2_20260831/route164_step522/multiframe_event_factory_report.json"
  "${scenario_root}/stage2l_formal_v5_qa_wave2_20260831/route165_step421/multiframe_event_factory_report.json"
  "${scenario_root}/stage2l_formal_v5_qa_wave3_20260831/route168_step482/multiframe_event_factory_report.json"
  "${scenario_root}/stage2l_formal_v5_qa_wave3_20260831/route180_step696/multiframe_event_factory_report.json"
  "${scenario_root}/stage2l_formal_v5_qa_wave3_20260831/route185_step304/multiframe_event_factory_report.json"
  "${scenario_root}/stage2l_formal_v5_qa_wave3_20260831/route188_step139/multiframe_event_factory_report.json"
  "${scenario_root}/stage2l_multiframe_qa_v5/route194_step684/multiframe_event_factory_report.json"
  "${scenario_root}/stage2l_formal_v5_qa_wave1_20260831/route195_step230/multiframe_event_factory_report.json"
  "${scenario_root}/stage2l_formal_v5_qa_wave3_20260831/route207_step924/multiframe_event_factory_report.json"
)

caches=(
  "${scenario_root}/stage2l_visual_cache_reattested_v5_20260831_wave1_v2/route147_step223/visual_cache_manifest.json"
  "${scenario_root}/stage2l_visual_cache_reattested_v5_20260831_wave1_v2/route151_step218/visual_cache_manifest.json"
  "${scenario_root}/stage2l_visual_cache_reattested_v5_20260831_v2/route152_step125/visual_cache_manifest.json"
  "${scenario_root}/stage2l_visual_cache_reattested_v5_20260831_wave1_v2/route157_step193/visual_cache_manifest.json"
  "${scenario_root}/stage2l_visual_cache_reattested_v5_20260831_wave1_v2/route158_step154/visual_cache_manifest.json"
  "${scenario_root}/stage2l_visual_cache_reattested_v5_20260831_wave1_v2/route160_step518/visual_cache_manifest.json"
  "${scenario_root}/stage2l_visual_cache_reattested_v5_20260831_wave1_v2/route161_step308/visual_cache_manifest.json"
  "${scenario_root}/stage2l_visual_cache_fresh_v5_20260831/route162_step277/orion_visual_contexts.json"
  "${scenario_root}/stage2l_visual_cache_reattested_v5_20260831_wave1_v2/route164_step522/visual_cache_manifest.json"
  "${scenario_root}/stage2l_visual_cache_reattested_v5_20260831_wave1_v2/route165_step421/visual_cache_manifest.json"
  "${scenario_root}/stage2l_visual_cache_reattested_v5_20260831_wave1_v2/route168_step482/visual_cache_manifest.json"
  "${scenario_root}/stage2l_visual_cache_reattested_v5_20260831_wave1_v2/route180_step696/visual_cache_manifest.json"
  "${scenario_root}/stage2l_visual_cache_reattested_v5_20260831_wave1_v2/route185_step304/visual_cache_manifest.json"
  "${scenario_root}/stage2l_visual_cache_fresh_v5_20260831/route188_step139/orion_visual_contexts.json"
  "${scenario_root}/stage2l_visual_cache_reattested_v5_20260831_wave1_v2/route194_step684/visual_cache_manifest.json"
  "${scenario_root}/stage2l_visual_cache_reattested_v5_20260831_wave1_v2/route195_step230/visual_cache_manifest.json"
  "${scenario_root}/stage2l_visual_cache_fresh_v5_20260831/route207_step924/orion_visual_contexts.json"
)

if [[ "${#reports[@]}" -ne 17 || "${#caches[@]}" -ne 17 ]]; then
  echo "formal inventory must bind exactly 17 currently usable events" >&2
  exit 1
fi
if [[ -e "${output}" ]]; then
  echo "refusing to overwrite ${output}" >&2
  exit 1
fi

args=(
  "${python_bin}" "${project_root}/scripts/audit_stage2l_formal_inventory.py"
  --partial-bank "${project_root}/results/scenario_factory/event_banks/stage2l_formal24_partial18_reviewed_v1.json"
  --formal-plan "${project_root}/results/scenario_factory/formal_route_plans/stage2l_formal24_16_4_4_20260829_v1/formal_route_plan.json"
  --formal-data-protocol "${project_root}/configs/scenario_factory/stage2l_formal24_data_and_corruption_protocol_v1.json"
  --qa-factory-config "${project_root}/configs/scenario_factory/qa_factory_v5_vlm_task_fields.json"
  --output "${output}"
)
for index in "${!reports[@]}"; do
  args+=(--factory-report "${reports[index]}")
  args+=(--visual-cache-manifest "${caches[index]}")
done

export PYTHONPATH="${project_root}:${PYTHONPATH:-}"
"${args[@]}"
