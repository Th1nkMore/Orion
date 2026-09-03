# ------------------------------------------------------------------------
# Modified from Bench2Drive(https://github.com/Thinklab-SJTU/Bench2Drive)
# Copyright (c) Xiaomi, Inc. All rights reserved.
# ------------------------------------------------------------------------

import os
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque
import math
from collections import OrderedDict
import torch
import carla
import numpy as np
from PIL import Image
from torchvision import transforms as T
try:
    # The leaderboard prepends the agent directory to sys.path.  Prefer the
    # planner shipped with this agent because Bench2DriveZoo's generic planner
    # has a different run_step return contract.
    from pid_controller import PIDController
    from planner import RoutePlanner
except ModuleNotFoundError:
    # Support regular package imports used by training and smoke tests.
    from team_code.pid_controller import PIDController
    from team_code.planner import RoutePlanner
from leaderboard.autoagents import autonomous_agent
from mmcv import Config
from mmcv.models import build_model 
from mmcv.utils import (get_dist_info, init_dist, load_checkpoint,wrap_fp16_model)
from mmcv.datasets.pipelines import Compose
from mmcv.parallel.collate import collate as  mm_collate_to_batch_form
from mmcv.core.bbox import get_box_type
from pyquaternion import Quaternion
from scipy.optimize import fsolve
from uq_estimator.corruptions import (
    corrupt_multiview_images,
    corrupt_multiview_images_with_metadata,
    normalized_front_tensor_to_bgr,
)
from uq_estimator.corruption_schedule import (
    RouteTriggeredTimedWindow,
    project_route_progress,
)
from uq_estimator.temporal_corruptions import (
    StaleFrameBuffer,
    stale_delay_ms_for_severity,
)
from uq_estimator.corruption_visual_approval import verify_visual_approval
from uq_estimator.lens_waterdrop_paired_template import (
    apply_paired_waterdrop_template,
    extract_paired_waterdrop_template,
)
from uq_estimator.risk_governor import UQRiskGovernor, load_score_trace
from uq_estimator.closedloop_safety_metrics import (
    SCHEMA_VERSION as CLOSEDLOOP_SAFETY_SCHEMA_VERSION,
    summarize_dynamic_actor_safety,
    vertical_separating_gap,
)
from uq_estimator.online_observation_uq import (
    RobustPreEventCalibrator,
    aggregate_observation_evidence,
    load_frozen_pairwise_adapter,
    pool_observation_evidence_grid,
    pool_observation_evidence_grids,
    summarize_spatial_observation_evidence,
)
from uq_estimator.privileged_yield_labels import (
    DynamicYieldLabeler,
    TrajectoryConflictConfig,
    build_safe_yield_trajectory,
    evaluate_trajectory_conflicts,
    select_actor_categories,
    trajectory_residual,
)
from uq_estimator.dynamic_yield_expert import (
    BrakingAwareYieldStateMachine,
    DynamicYieldExpertConfig,
    build_dynamics_aware_yield_trajectory,
    compute_junction_yield_geometry,
    first_path_junction_entry,
    resolve_junction_scoped_conflict,
)
from uq_estimator.bounded_crossing_expert import (
    BoundedCrossingExpertConfig,
    build_braking_aware_crossing_trajectory,
)
from uq_estimator.stage2_artifact_capture import Stage2ArtifactWriter
try:
    from orion_native_glare import (
        install_orion_native_glare_sensor_patch,
        normalize_native_glare_profile,
        readback_render_condition,
        record_render_condition_readback,
        requested_render_condition,
    )
except ModuleNotFoundError:
    from team_code.orion_native_glare import (
        install_orion_native_glare_sensor_patch,
        normalize_native_glare_profile,
        readback_render_condition,
        record_render_condition_readback,
        requested_render_condition,
    )
try:
    from orion_native_motion_blur import (
        install_orion_native_motion_blur_sensor_patch,
        normalize_native_motion_blur_profile,
        readback_native_motion_blur_condition,
        requested_native_motion_blur_condition,
    )
except ModuleNotFoundError:
    from team_code.orion_native_motion_blur import (
        install_orion_native_motion_blur_sensor_patch,
        normalize_native_motion_blur_profile,
        readback_native_motion_blur_condition,
        requested_native_motion_blur_condition,
    )
from leaderboard.autoagents.agent_wrapper import AgentWrapper
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
SAVE_PATH = os.environ.get('SAVE_PATH', None)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)

CAMERA_ORDER = (
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
)


def parse_corruption_view_indices(spec):
    """Resolve a comma-separated camera selection into tensor view indices."""
    aliases = {
        'front': ('CAM_FRONT',),
        'front_group': CAMERA_ORDER[:3],
        'rear': ('CAM_BACK',),
        'rear_group': CAMERA_ORDER[3:],
        'all': CAMERA_ORDER,
    }
    normalized = (spec or 'front').strip()
    cameras = aliases.get(normalized, tuple(
        item.strip() for item in normalized.split(',') if item.strip()
    ))
    unknown = [camera for camera in cameras if camera not in CAMERA_ORDER]
    if unknown:
        raise ValueError(
            f'Unknown corruption cameras {unknown}; expected names from '
            f'{CAMERA_ORDER} or aliases {sorted(aliases)}'
        )
    if not cameras:
        raise ValueError('At least one corruption camera must be selected')
    return [CAMERA_ORDER.index(camera) for camera in cameras]


def parse_corruption_region(spec):
    """Parse a normalized ``top,left,bottom,right`` corruption region."""
    normalized = (spec or '').strip()
    if not normalized:
        return None
    parts = [item.strip() for item in normalized.split(',')]
    if len(parts) != 4:
        raise ValueError(
            'ORION_CLOSEDLOOP_CORRUPTION_REGION must contain '
            'top,left,bottom,right'
        )
    region = tuple(float(item) for item in parts)
    top, left, bottom, right = region
    if not all(math.isfinite(value) for value in region):
        raise ValueError('Corruption region coordinates must be finite')
    if not (0.0 <= top < bottom <= 1.0):
        raise ValueError('Corruption region must satisfy 0 <= top < bottom <= 1')
    if not (0.0 <= left < right <= 1.0):
        raise ValueError('Corruption region must satisfy 0 <= left < right <= 1')
    return region


def parse_boolean_flag(value, name):
    normalized = str(value).strip().lower()
    if normalized in {'1', 'true', 'yes', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'off'}:
        return False
    raise ValueError(f'{name} must be a boolean flag, got {value!r}')


def build_stage2_external_k_map(
    *, camera_index, region, active, strength, grid_size=40,
    device='cpu', dtype=torch.float32,
):
    """Build the bounded engineering-smoke K map in canonical camera order."""
    if not 0 <= int(camera_index) < len(CAMERA_ORDER):
        raise ValueError('Stage2-P external K camera index is out of range')
    if int(grid_size) <= 0:
        raise ValueError('Stage2-P external K grid size must be positive')
    if not math.isfinite(float(strength)) or not 0.0 <= float(strength) <= 1.0:
        raise ValueError('Stage2-P external K strength must lie in [0,1]')
    if region is None:
        raise ValueError('Stage2-P external K requires a normalized region')
    top, left, bottom, right = region
    output = torch.zeros(
        (len(CAMERA_ORDER), int(grid_size), int(grid_size)),
        device=device,
        dtype=dtype,
    )
    if not active:
        return output
    row_start = max(0, min(grid_size - 1, int(math.floor(top * grid_size))))
    row_end = max(row_start + 1, min(grid_size, int(math.ceil(bottom * grid_size))))
    column_start = max(
        0, min(grid_size - 1, int(math.floor(left * grid_size)))
    )
    column_end = max(
        column_start + 1, min(grid_size, int(math.ceil(right * grid_size)))
    )
    output[
        int(camera_index), row_start:row_end, column_start:column_end
    ] = float(strength)
    return output


def render_front_corruption_preview(image, corruption, severity, region):
    """Render the model's front-view intervention for saved RGB diagnostics."""
    preview = image.copy()
    if corruption == 'camera_dropout':
        return np.zeros_like(preview)
    if not corruption.startswith('local_') or region is None:
        return preview
    height, width = preview.shape[:2]
    top = max(0, min(height, int(math.floor(region[0] * height))))
    left = max(0, min(width, int(math.floor(region[1] * width))))
    bottom = max(0, min(height, int(math.ceil(region[2] * height))))
    right = max(0, min(width, int(math.ceil(region[3] * width))))
    patch = preview[top:bottom, left:right]
    if corruption == 'local_blur':
        kernel = (3, 5, 9)[severity - 1]
        preview[top:bottom, left:right] = cv2.blur(
            patch, (kernel, kernel)
        )
    elif corruption == 'local_dark':
        factor = (0.75, 0.5, 0.3)[severity - 1]
        preview[top:bottom, left:right] = np.clip(
            patch.astype(np.float32) * factor, 0, 255
        ).astype(preview.dtype)
    elif corruption == 'local_glare':
        alpha = (0.35, 0.60, 0.85)[severity - 1]
        preview[top:bottom, left:right] = np.clip(
            (1.0 - alpha) * patch.astype(np.float32) + alpha * 255.0,
            0,
            255,
        ).astype(preview.dtype)
    elif corruption == 'local_occlusion':
        preview[top:bottom, left:right] = 0
    return preview


def load_film_checkpoint(model, film_ckpt_path):
    """Load optional FiLM weights on top of the base ORION checkpoint."""
    if not film_ckpt_path:
        return

    if not os.path.exists(film_ckpt_path):
        raise FileNotFoundError(f'FiLM checkpoint not found: {film_ckpt_path}')

    film_ckpt = torch.load(film_ckpt_path, map_location='cpu')
    transformer = model.pts_bbox_head.transformer

    if hasattr(transformer, 'film_gamma') and 'film_gamma_weight' in film_ckpt:
        transformer.film_gamma.weight.data.copy_(film_ckpt['film_gamma_weight'])
        transformer.film_gamma.bias.data.copy_(film_ckpt['film_gamma_bias'])
        transformer.film_beta.weight.data.copy_(film_ckpt['film_beta_weight'])
        transformer.film_beta.bias.data.copy_(film_ckpt['film_beta_bias'])
        print(f'[FiLM] Loaded L1 weights from {film_ckpt_path}')

    if hasattr(model, 'film_gamma_l2') and 'film_gamma_l2_weight' in film_ckpt:
        model.film_gamma_l2.weight.data.copy_(film_ckpt['film_gamma_l2_weight'])
        model.film_gamma_l2.bias.data.copy_(film_ckpt['film_gamma_l2_bias'])
        model.film_beta_l2.weight.data.copy_(film_ckpt['film_beta_l2_weight'])
        model.film_beta_l2.bias.data.copy_(film_ckpt['film_beta_l2_bias'])
        print(f'[FiLM] Loaded L2 weights from {film_ckpt_path}')


def parse_team_config(path_to_conf_file):
    """Parse TEAM_CONFIG payload.

    Supported formats:
    - config.py+base_ckpt.pth
    - config.py+base_ckpt.pth+film_ckpt.pt
    - config.py+base_ckpt.pth+save_name
    - config.py+base_ckpt.pth+film_ckpt.pt+save_name
    """
    parts = path_to_conf_file.split('+')
    if len(parts) < 2:
        raise ValueError(f'Invalid TEAM_CONFIG payload: {path_to_conf_file}')

    config_path = parts[0]
    ckpt_path = parts[1]
    film_ckpt_path = None
    save_name = None

    extras = parts[2:]
    if extras and extras[0].endswith(('.pt', '.pth')):
        film_ckpt_path = extras[0]
        extras = extras[1:]
    if extras:
        save_name = extras[0]

    return config_path, ckpt_path, film_ckpt_path, save_name


def resolve_local_model_paths(config, project_root=None):
    """Resolve repository-local model assets independently of the cwd.

    Bench2Drive changes into its own repository before importing the agent,
    while the ORION configs historically express the Q-Former assets relative
    to the ORION repository.  Keep Hugging Face model identifiers untouched
    and only rewrite relative paths that actually exist below the project root.
    """
    root = pathlib.Path(project_root or pathlib.Path(__file__).resolve().parents[1])
    local_path_keys = {'tokenizer', 'lm_head'}

    def visit(node):
        if isinstance(node, dict):
            for key, value in list(node.items()):
                if key in local_path_keys and isinstance(value, str) and not os.path.isabs(value):
                    candidate = (root / value).resolve()
                    if candidate.exists():
                        node[key] = str(candidate)
                        print(f'[Config] Resolved {key}: {value} -> {candidate}')
                        continue
                visit(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                visit(value)

    visit(getattr(config, '_cfg_dict', config))
    return config

def get_entry_point():
    return 'OrionAgent'

class OrionAgent(autonomous_agent.AutonomousAgent):

    def setup_model(self, model, pipeline):
        self.model = model
        self.inference_only_pipeline = pipeline

    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.native_glare_profile = normalize_native_glare_profile(
            os.environ.get('ORION_NATIVE_GLARE_PROFILE', 'none')
        )
        self.native_motion_blur_profile = normalize_native_motion_blur_profile(
            os.environ.get('ORION_NATIVE_MOTION_BLUR_PROFILE', 'none')
        )
        if (
            self.native_glare_profile != 'none'
            and self.native_motion_blur_profile != 'none'
        ):
            raise RuntimeError(
                'native glare and native motion blur are mutually exclusive '
                'paired render conditions'
            )
        install_orion_native_glare_sensor_patch(
            AgentWrapper, self.native_glare_profile
        )
        install_orion_native_motion_blur_sensor_patch(
            AgentWrapper, self.native_motion_blur_profile
        )
        self.native_glare_render_condition = requested_render_condition(
            self.native_glare_profile
        )
        self.native_render_condition = (
            requested_native_motion_blur_condition(
                self.native_motion_blur_profile
            )
            if self.native_motion_blur_profile != 'none'
            else self.native_glare_render_condition
        )
        self.native_glare_readback_recorded = False
        self.steer_step = 0
        self.last_moving_status = 0
        self.last_moving_step = -1
        self.last_steers = 0
        self.pidcontroller = PIDController() 
        now = datetime.datetime.now()
        self.config_path, self.ckpt_path, self.film_ckpt_path, save_name = parse_team_config(path_to_conf_file)
        self.uq_mode = os.environ.get('ORION_CLOSEDLOOP_UQ_MODE', 'none')
        if self.uq_mode not in {'none', 'zero', 'correct'}:
            raise ValueError(
                'ORION_CLOSEDLOOP_UQ_MODE must be none, zero, or correct; '
                f'got {self.uq_mode!r}'
            )
        if self.uq_mode != 'none':
            raise RuntimeError(
                'Legacy global UQ inference modes are retired from the current '
                'closed-loop pipeline; use the frozen spatial observation-UQ '
                'checkpoint and the planning-layer Stage-2 path'
            )
        self.uq_conditioning = os.environ.get(
            'ORION_CLOSEDLOOP_CONDITIONING', 'none'
        )
        if self.uq_conditioning not in {'none', 'token', 'vision_adapter'}:
            raise ValueError(
                'ORION_CLOSEDLOOP_CONDITIONING must be none, token, or '
                'vision_adapter; '
                f'got {self.uq_conditioning!r}'
            )
        if self.uq_conditioning != 'none':
            raise RuntimeError(
                'Legacy Density token/vision-adapter conditioning is retired; '
                'ORION_CLOSEDLOOP_CONDITIONING must be none'
            )
        self.closedloop_corruption = os.environ.get(
            'ORION_CLOSEDLOOP_CORRUPTION', ''
        )
        if self.closedloop_corruption == 'lens_waterdrop':
            raise RuntimeError(
                'lens_waterdrop is the retired failed v1 visual prototype; '
                'use the explicitly approved lens_waterdrop_paired_template '
                'condition instead'
            )
        supported_corruptions = {
            '', 'blur', 'dark', 'camera_dropout', 'local_blur', 'local_dark',
            'local_glare', 'local_occlusion', 'front_stale',
            'lens_waterdrop_paired_template',
        }
        if self.closedloop_corruption not in supported_corruptions:
            raise ValueError(
                'Unsupported ORION_CLOSEDLOOP_CORRUPTION '
                f'{self.closedloop_corruption!r}; expected one of '
                f'{sorted(supported_corruptions)}'
            )
        self.corruption_severity = int(os.environ.get(
            'ORION_CLOSEDLOOP_CORRUPTION_SEVERITY', '1'
        ))
        if self.corruption_severity not in {1, 2, 3}:
            raise ValueError('ORION_CLOSEDLOOP_CORRUPTION_SEVERITY must be 1, 2, or 3')
        self.corruption_view_spec = os.environ.get(
            'ORION_CLOSEDLOOP_CORRUPTION_VIEWS', 'front'
        )
        self.corruption_view_indices = parse_corruption_view_indices(
            self.corruption_view_spec
        )
        if (
            self.closedloop_corruption in {
                'front_stale', 'lens_waterdrop_paired_template'
            }
            and self.corruption_view_indices != [0]
        ):
            raise ValueError(
                '%s is intentionally limited to CAM_FRONT; set '
                'ORION_CLOSEDLOOP_CORRUPTION_VIEWS=front'
                % self.closedloop_corruption
            )
        self.stale_frame_buffer = (
            StaleFrameBuffer(stale_delay_ms_for_severity(self.corruption_severity))
            if self.closedloop_corruption == 'front_stale'
            else None
        )
        self.corruption_seed = int(os.environ.get(
            'ORION_CLOSEDLOOP_CORRUPTION_SEED', '0'
        ))
        self.corruption_region = parse_corruption_region(os.environ.get(
            'ORION_CLOSEDLOOP_CORRUPTION_REGION', ''
        ))
        if (
            self.closedloop_corruption.startswith('local_')
            and self.corruption_region is None
        ):
            raise ValueError(
                'Localized closed-loop corruption requires a frozen explicit '
                'ORION_CLOSEDLOOP_CORRUPTION_REGION'
            )
        self.corruption_start_seconds = float(os.environ.get(
            'ORION_CLOSEDLOOP_CORRUPTION_START_SECONDS', '0'
        ))
        corruption_end = os.environ.get(
            'ORION_CLOSEDLOOP_CORRUPTION_END_SECONDS', ''
        )
        self.corruption_end_seconds = (
            float(corruption_end) if corruption_end else float('inf')
        )
        if self.corruption_start_seconds < 0:
            raise ValueError('Corruption start time must be non-negative')
        if self.corruption_end_seconds <= self.corruption_start_seconds:
            raise ValueError('Corruption end time must be after its start time')
        corruption_start_progress = os.environ.get(
            'ORION_CLOSEDLOOP_CORRUPTION_START_PROGRESS', ''
        )
        corruption_end_progress = os.environ.get(
            'ORION_CLOSEDLOOP_CORRUPTION_END_PROGRESS', ''
        )
        corruption_duration_seconds = os.environ.get(
            'ORION_CLOSEDLOOP_CORRUPTION_DURATION_SECONDS', ''
        )
        has_start_progress = bool(corruption_start_progress)
        has_end_progress = bool(corruption_end_progress)
        has_duration = bool(corruption_duration_seconds)
        if has_duration and (not has_start_progress or has_end_progress):
            raise ValueError(
                'Route-triggered timed corruption requires start progress and '
                'duration, without end progress'
            )
        if not has_duration and has_start_progress != has_end_progress:
            raise ValueError(
                'Route-progress corruption windows require both start and end'
            )
        self.corruption_start_progress = (
            float(corruption_start_progress)
            if corruption_start_progress else None
        )
        self.corruption_end_progress = (
            float(corruption_end_progress) if corruption_end_progress else None
        )
        self.corruption_duration_seconds = (
            float(corruption_duration_seconds)
            if corruption_duration_seconds else None
        )
        self.corruption_timed_window = None
        if self.corruption_duration_seconds is not None:
            self.corruption_timed_window = RouteTriggeredTimedWindow(
                self.corruption_start_progress,
                self.corruption_duration_seconds,
            )
            self.corruption_schedule_mode = 'route_triggered_timed'
        elif self.corruption_start_progress is not None:
            if not (
                0.0 <= self.corruption_start_progress
                < self.corruption_end_progress <= 1.0
            ):
                raise ValueError(
                    'Require 0 <= corruption start progress < end progress <= 1'
                )
            self.corruption_schedule_mode = 'route_progress'
        else:
            self.corruption_schedule_mode = 'simulation_time'

        self.stage2p_engineering_smoke = parse_boolean_flag(
            os.environ.get('ORION_STAGE2_ENGINEERING_SMOKE', '0'),
            'ORION_STAGE2_ENGINEERING_SMOKE',
        )
        stage2_k_start = os.environ.get(
            'ORION_STAGE2_EXTERNAL_K_START_PROGRESS', ''
        ).strip()
        stage2_k_duration = os.environ.get(
            'ORION_STAGE2_EXTERNAL_K_DURATION_SECONDS', ''
        ).strip()
        if bool(stage2_k_start) != bool(stage2_k_duration):
            raise ValueError(
                'Stage2-P external K requires start progress and duration'
            )
        self.stage2_external_k_window = None
        if stage2_k_start:
            self.stage2_external_k_window = RouteTriggeredTimedWindow(
                float(stage2_k_start), float(stage2_k_duration)
            )
        stage2_k_camera = os.environ.get(
            'ORION_STAGE2_EXTERNAL_K_CAMERA', 'CAM_FRONT'
        ).strip()
        if stage2_k_camera not in CAMERA_ORDER:
            raise ValueError('ORION_STAGE2_EXTERNAL_K_CAMERA is invalid')
        self.stage2_external_k_camera_index = CAMERA_ORDER.index(
            stage2_k_camera
        )
        self.stage2_external_k_region = parse_corruption_region(
            os.environ.get('ORION_STAGE2_EXTERNAL_K_REGION', '')
        )
        self.stage2_external_k_strength = float(os.environ.get(
            'ORION_STAGE2_EXTERNAL_K_STRENGTH', '1.0'
        ))
        self.stage2_external_k_grid_size = int(os.environ.get(
            'ORION_STAGE2_EXTERNAL_K_GRID_SIZE', '40'
        ))
        if self.stage2p_engineering_smoke:
            if (
                self.stage2_external_k_window is None
                or self.stage2_external_k_region is None
                or self.closedloop_corruption
            ):
                raise RuntimeError(
                    'Stage2-P engineering smoke requires a bounded external '
                    'K window/region and a clean visual input'
                )
            build_stage2_external_k_map(
                camera_index=self.stage2_external_k_camera_index,
                region=self.stage2_external_k_region,
                active=False,
                strength=self.stage2_external_k_strength,
                grid_size=self.stage2_external_k_grid_size,
            )
        elif any((
            stage2_k_start,
            stage2_k_duration,
            os.environ.get('ORION_STAGE2_EXTERNAL_K_REGION', '').strip(),
        )):
            raise RuntimeError(
                'External K settings require ORION_STAGE2_ENGINEERING_SMOKE=1'
            )

        self.paired_waterdrop_profile = None
        self.paired_waterdrop_template = None
        if self.closedloop_corruption == 'lens_waterdrop_paired_template':
            self.paired_waterdrop_profile = os.environ.get(
                'ORION_PAIRED_WATERDROP_PROFILE', ''
            )
            if self.paired_waterdrop_profile not in {'light', 'medium', 'heavy'}:
                raise ValueError(
                    'ORION_PAIRED_WATERDROP_PROFILE must explicitly select '
                    'light, medium, or heavy'
                )

        approval_requests = []
        if self.closedloop_corruption == 'front_stale':
            approval_requests.append((
                'front_stale',
                'delay_ms:%d' % stale_delay_ms_for_severity(
                    self.corruption_severity
                ),
            ))
        elif self.closedloop_corruption == 'lens_waterdrop_paired_template':
            approval_requests.append((
                'lens_waterdrop_paired_template',
                'profile:%s' % self.paired_waterdrop_profile,
            ))
        if self.native_motion_blur_profile != 'none':
            approval_requests.append((
                'native_motion_blur',
                'profile:%s' % self.native_motion_blur_profile,
            ))
        if len(approval_requests) > 1:
            raise RuntimeError(
                'Hard-case ORION screens must use exactly one corruption '
                'family per run; stacked corruptions are not approved'
            )
        self.corruption_visual_approval = None
        if approval_requests:
            approval_gate = os.environ.get(
                'ORION_CORRUPTION_VISUAL_APPROVAL_GATE', ''
            )
            if not approval_gate:
                raise RuntimeError(
                    'ORION_CORRUPTION_VISUAL_APPROVAL_GATE is required for '
                    'every corruption-conditioned ORION run'
                )
            project_root = pathlib.Path(__file__).resolve().parents[1]
            approval_gate_path = pathlib.Path(approval_gate)
            if not approval_gate_path.is_absolute():
                approval_gate_path = project_root / approval_gate_path
            family, condition = approval_requests[0]
            self.corruption_visual_approval = verify_visual_approval(
                gate_path=approval_gate_path,
                repository_root=project_root,
                family=family,
                condition=condition,
                require_approved=True,
            ).to_dict()

        if self.closedloop_corruption == 'lens_waterdrop_paired_template':
            project_root = pathlib.Path(__file__).resolve().parents[1]
            bank = pathlib.Path(os.environ.get(
                'ORION_PAIRED_WATERDROP_BANK',
                str(
                    project_root
                    / 'assets/waterdrop_patterns/icra2023_paired_template_v1'
                ),
            )).resolve()
            self.paired_waterdrop_template = extract_paired_waterdrop_template(
                clean_path=bank / 'test__syn__clean_vid__0003__000075.png',
                rainy_path=bank / 'test__syn__rainy_vid__0003__000075.png',
                metadata_path=bank / 'metadata.json',
            )

        self.risk_mode = os.environ.get(
            'ORION_CLOSEDLOOP_RISK_MODE', 'off'
        )
        self.planning_response_mode = os.environ.get(
            'ORION_PLANNING_RESPONSE_MODE', 'off'
        )
        if self.planning_response_mode not in {
            'off',
            'privileged_bounded_crossing',
            'privileged_braking_aware_crossing',
            'privileged_dynamic_yield',
            'privileged_dynamics_aware_yield',
        }:
            raise ValueError(
                'ORION_PLANNING_RESPONSE_MODE must be off, '
                'privileged_bounded_crossing, '
                'privileged_braking_aware_crossing, '
                'privileged_dynamic_yield, or '
                'privileged_dynamics_aware_yield'
            )
        self.planning_actor_categories = tuple(
            item.strip().lower()
            for item in os.environ.get(
                'ORION_PLANNING_ACTOR_CATEGORIES', 'walker'
            ).split(',')
            if item.strip()
        )
        if (
            self.planning_response_mode == 'privileged_bounded_crossing'
            or self.planning_response_mode
            == 'privileged_braking_aware_crossing'
        ) and not self.planning_actor_categories:
            raise ValueError(
                'privileged bounded crossing requires frozen actor categories'
            )
        self.planning_conflict_config = TrajectoryConflictConfig(
            interpolation_step_seconds=float(os.environ.get(
                'ORION_PLANNING_INTERPOLATION_STEP_SECONDS', '0.1'
            )),
            safety_margin_m=float(os.environ.get(
                'ORION_PLANNING_SAFETY_MARGIN_M', '0.75'
            )),
            imminent_horizon_seconds=float(os.environ.get(
                'ORION_PLANNING_IMMINENT_HORIZON_SECONDS', '1.5'
            )),
            clearance_seconds=float(os.environ.get(
                'ORION_PLANNING_CLEARANCE_SECONDS', '1.0'
            )),
            release_seconds=float(os.environ.get(
                'ORION_PLANNING_RELEASE_SECONDS', '0.5'
            )),
            stop_buffer_m=float(os.environ.get(
                'ORION_PLANNING_STOP_BUFFER_M', '2.0'
            )),
            release_creep_distance_m=float(os.environ.get(
                'ORION_PLANNING_RELEASE_CREEP_DISTANCE_M', '1.0'
            )),
        )
        self.dynamic_yield_labeler = (
            DynamicYieldLabeler(self.planning_conflict_config)
            if self.planning_response_mode in {
                'privileged_bounded_crossing',
                'privileged_braking_aware_crossing',
                'privileged_dynamic_yield',
            }
            else None
        )
        self.dynamic_yield_expert_config = DynamicYieldExpertConfig(
            certified_deceleration_mps2=float(os.environ.get(
                'ORION_PLANNING_CERTIFIED_DECELERATION_MPS2', '3.0'
            )),
            reaction_seconds=float(os.environ.get(
                'ORION_PLANNING_REACTION_SECONDS', '0.1'
            )),
            junction_front_clearance_m=float(os.environ.get(
                'ORION_PLANNING_JUNCTION_FRONT_CLEARANCE_M', '0.5'
            )),
            clearance_seconds=float(os.environ.get(
                'ORION_PLANNING_CLEARANCE_SECONDS', '1.0'
            )),
            release_seconds=float(os.environ.get(
                'ORION_PLANNING_RELEASE_SECONDS', '0.5'
            )),
            prepare_creep_speed_mps=float(os.environ.get(
                'ORION_PLANNING_PREPARE_CREEP_SPEED_MPS', '1.0'
            )),
            release_creep_speed_mps=float(os.environ.get(
                'ORION_PLANNING_RELEASE_CREEP_SPEED_MPS', '0.5'
            )),
            release_creep_distance_m=float(os.environ.get(
                'ORION_PLANNING_RELEASE_CREEP_DISTANCE_M', '1.0'
            )),
        )
        self.bounded_crossing_expert_config = BoundedCrossingExpertConfig(
            certified_deceleration_mps2=float(os.environ.get(
                'ORION_PLANNING_CERTIFIED_DECELERATION_MPS2', '3.0'
            )),
            prepare_creep_speed_mps=float(os.environ.get(
                'ORION_PLANNING_PREPARE_CREEP_SPEED_MPS', '1.0'
            )),
            release_creep_speed_mps=float(os.environ.get(
                'ORION_PLANNING_RELEASE_CREEP_SPEED_MPS', '0.5'
            )),
            release_creep_distance_m=float(os.environ.get(
                'ORION_PLANNING_RELEASE_CREEP_DISTANCE_M', '1.0'
            )),
        )
        self.dynamics_aware_yield_expert = (
            BrakingAwareYieldStateMachine(self.dynamic_yield_expert_config)
            if self.planning_response_mode
            == 'privileged_dynamics_aware_yield'
            else None
        )
        self.dynamic_yield_map_resolution_m = float(os.environ.get(
            'ORION_PLANNING_MAP_RESOLUTION_M', '0.1'
        ))
        if self.dynamic_yield_map_resolution_m <= 0.0:
            raise ValueError('ORION_PLANNING_MAP_RESOLUTION_M must be positive')
        self._dynamic_yield_map = None
        requested_legacy_density_uq = parse_boolean_flag(
            os.environ.get('ORION_ENABLE_LEGACY_DENSITY_UQ', '0'),
            'ORION_ENABLE_LEGACY_DENSITY_UQ',
        )
        if requested_legacy_density_uq:
            raise RuntimeError(
                'Legacy Density UQ is retired from the current closed-loop '
                'agent; use the frozen spatial observation-UQ adapter instead'
            )
        self.legacy_density_uq_enabled = False
        oracle_start_progress = os.environ.get(
            'ORION_CLOSEDLOOP_RISK_ORACLE_START_PROGRESS', ''
        )
        oracle_duration_seconds = os.environ.get(
            'ORION_CLOSEDLOOP_RISK_ORACLE_DURATION_SECONDS', ''
        )
        if bool(oracle_start_progress) != bool(oracle_duration_seconds):
            raise ValueError(
                'Native-event risk oracle requires both start progress and duration'
            )
        self.risk_oracle_timed_window = None
        if oracle_start_progress:
            if self.risk_mode != 'oracle':
                raise ValueError(
                    'Native-event risk oracle window requires risk mode oracle'
                )
            self.risk_oracle_timed_window = RouteTriggeredTimedWindow(
                float(oracle_start_progress), float(oracle_duration_seconds)
            )
        self.oracle_corruption_relevant = parse_boolean_flag(
            os.environ.get('ORION_CLOSEDLOOP_ORACLE_CORRUPTION_RELEVANT', '1'),
            'ORION_CLOSEDLOOP_ORACLE_CORRUPTION_RELEVANT',
        )
        risk_trace_path = os.environ.get(
            'ORION_CLOSEDLOOP_RISK_TRACE', ''
        )
        risk_trace = load_score_trace(risk_trace_path) if risk_trace_path else None
        self.risk_governor = UQRiskGovernor(
            mode=self.risk_mode,
            threshold=float(os.environ.get(
                'ORION_CLOSEDLOOP_RISK_THRESHOLD', '0.4'
            )),
            saturation=float(os.environ.get(
                'ORION_CLOSEDLOOP_RISK_SATURATION', '0.8'
            )),
            min_speed=float(os.environ.get(
                'ORION_CLOSEDLOOP_RISK_MIN_SPEED', '1.5'
            )),
            max_speed=float(os.environ.get(
                'ORION_CLOSEDLOOP_RISK_MAX_SPEED', '5.0'
            )),
            slowdown_margin=float(os.environ.get(
                'ORION_CLOSEDLOOP_RISK_SLOWDOWN_MARGIN', '1.0'
            )),
            brake_gain=float(os.environ.get(
                'ORION_CLOSEDLOOP_RISK_BRAKE_GAIN', '0.5'
            )),
            max_brake=float(os.environ.get(
                'ORION_CLOSEDLOOP_RISK_MAX_BRAKE', '0.5'
            )),
            constant_score=float(os.environ.get(
                'ORION_CLOSEDLOOP_RISK_CONSTANT', '0.6209'
            )),
            trace_scores=risk_trace,
        )
        self.closedloop_safety_telemetry = parse_boolean_flag(
            os.environ.get('ORION_CLOSEDLOOP_SAFETY_TELEMETRY', '1'),
            'ORION_CLOSEDLOOP_SAFETY_TELEMETRY',
        )
        self.closedloop_safety_horizon_seconds = float(os.environ.get(
            'ORION_CLOSEDLOOP_SAFETY_HORIZON_SECONDS', '10.0'
        ))
        self.closedloop_safety_range_m = float(os.environ.get(
            'ORION_CLOSEDLOOP_SAFETY_RANGE_M', '60.0'
        ))
        self.closedloop_safety_actor_refresh_steps = int(os.environ.get(
            'ORION_CLOSEDLOOP_SAFETY_ACTOR_REFRESH_STEPS', '5'
        ))
        self.closedloop_safety_vertical_margin_m = float(os.environ.get(
            'ORION_CLOSEDLOOP_SAFETY_VERTICAL_MARGIN_M', '1.0'
        ))
        self.closedloop_safety_max_actor_records = int(os.environ.get(
            'ORION_CLOSEDLOOP_SAFETY_MAX_ACTOR_RECORDS', '32'
        ))
        if self.closedloop_safety_horizon_seconds <= 0.0:
            raise ValueError(
                'ORION_CLOSEDLOOP_SAFETY_HORIZON_SECONDS must be positive'
            )
        if self.closedloop_safety_range_m <= 0.0:
            raise ValueError('ORION_CLOSEDLOOP_SAFETY_RANGE_M must be positive')
        if self.closedloop_safety_actor_refresh_steps <= 0:
            raise ValueError(
                'ORION_CLOSEDLOOP_SAFETY_ACTOR_REFRESH_STEPS must be positive'
            )
        if self.closedloop_safety_vertical_margin_m < 0.0:
            raise ValueError(
                'ORION_CLOSEDLOOP_SAFETY_VERTICAL_MARGIN_M must be non-negative'
            )
        if self.closedloop_safety_max_actor_records <= 0:
            raise ValueError(
                'ORION_CLOSEDLOOP_SAFETY_MAX_ACTOR_RECORDS must be positive'
            )
        self._safety_actor_cache = {}
        self._safety_actor_cache_step = None
        self._safety_telemetry_error_reported = False
        self.observation_uq_checkpoint_path = os.environ.get(
            'ORION_OBSERVATION_UQ_CHECKPOINT', ''
        )
        self.observation_uq_checkpoint_sha256 = os.environ.get(
            'ORION_OBSERVATION_UQ_CHECKPOINT_SHA256', ''
        )
        if self.planning_response_mode in {
            'privileged_bounded_crossing',
            'privileged_braking_aware_crossing',
            'privileged_dynamic_yield',
            'privileged_dynamics_aware_yield',
        }:
            if self.risk_mode != 'off':
                raise ValueError(
                    'planning-level oracle requires scalar risk governor off'
                )
            if self.legacy_density_uq_enabled:
                raise ValueError(
                    'planning-level oracle requires legacy Density UQ disabled'
                )
            if self.observation_uq_checkpoint_path:
                raise ValueError(
                    'planning-level oracle must not load a learned UQ adapter'
                )
            if not self.closedloop_safety_telemetry:
                raise ValueError(
                    'planning-level oracle requires closed-loop safety telemetry'
                )
        self.observation_uq_front_view = int(os.environ.get(
            'ORION_OBSERVATION_UQ_FRONT_VIEW', '0'
        ))
        if not 0 <= self.observation_uq_front_view < len(CAMERA_ORDER):
            raise ValueError('ORION_OBSERVATION_UQ_FRONT_VIEW is out of range')
        self.observation_uq_adapter = None
        self.observation_uq_metadata = None
        self.observation_uq_previous_features = None
        self.observation_uq_previous_valid = False
        self.observation_uq_calibrator = RobustPreEventCalibrator(
            baseline_start_seconds=float(os.environ.get(
                'ORION_OBSERVATION_UQ_BASELINE_START_SECONDS', '1.0'
            )),
            baseline_end_seconds=float(os.environ.get(
                'ORION_OBSERVATION_UQ_BASELINE_END_SECONDS', '4.0'
            )),
            minimum_baseline_frames=int(os.environ.get(
                'ORION_OBSERVATION_UQ_MIN_BASELINE_FRAMES', '40'
            )),
            relative_scale_floor=float(os.environ.get(
                'ORION_OBSERVATION_UQ_RELATIVE_SCALE_FLOOR', '0.05'
            )),
            absolute_scale_floor=float(os.environ.get(
                'ORION_OBSERVATION_UQ_ABSOLUTE_SCALE_FLOOR', '0.001'
            )),
            z_center=float(os.environ.get(
                'ORION_OBSERVATION_UQ_Z_CENTER', '4.0'
            )),
            attack_alpha=float(os.environ.get(
                'ORION_OBSERVATION_UQ_ATTACK_ALPHA', '0.8'
            )),
            release_alpha=float(os.environ.get(
                'ORION_OBSERVATION_UQ_RELEASE_ALPHA', '0.2'
            )),
        )
        if IS_BENCH2DRIVE:
            self.save_name = save_name or pathlib.Path(self.ckpt_path).stem
        else:
            self.save_name = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        if not (hasattr(self, 'model') and self.model is not None and 
                hasattr(self, 'inference_only_pipeline') and self.inference_only_pipeline is not None):
            cfg = Config.fromfile(self.config_path)
            resolve_local_model_paths(cfg)
            if hasattr(cfg, 'plugin'):
                if cfg.plugin:
                    import importlib
                    if hasattr(cfg, 'plugin_dir'):
                        plugin_dir = cfg.plugin_dir
                        plugin_dir = os.path.join("Bench2DriveZoo", plugin_dir)
                        _module_dir = os.path.dirname(plugin_dir)
                        _module_dir = _module_dir.split('/')
                        _module_path = _module_dir[0]
                        for m in _module_dir[1:]:
                            _module_path = _module_path + '.' + m
                        print(_module_path)
                        plg_lib = importlib.import_module(_module_path)  
    
            self.model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
            checkpoint = load_checkpoint(self.model, self.ckpt_path, map_location='cpu')
            if self.film_ckpt_path and (
                cfg.model.get('use_uq_token', False)
                or cfg.model.get('use_uq_vision_adapter', False)
            ):
                from uq_estimator.training import load_uq_token_weights
                loaded = load_uq_token_weights(self.model, self.film_ckpt_path)
                print(
                    f'[UQ] Loaded {loaded} adaptation tensors from '
                    f'{self.film_ckpt_path}'
                )
            else:
                load_film_checkpoint(self.model, self.film_ckpt_path)
            self.model.cuda()
            self.model.eval()
            self.inference_only_pipeline = []
            for inference_only_pipeline in cfg.inference_only_pipeline:
                if inference_only_pipeline["type"] not in ['LoadMultiViewImageFromFilesInCeph']:
                    self.inference_only_pipeline.append(inference_only_pipeline)
            self.inference_only_pipeline = Compose(self.inference_only_pipeline)

        density_head_enabled = bool(getattr(
            getattr(self.model, 'pts_bbox_head', None),
            'use_uncertainty',
            False,
        ))
        if density_head_enabled != self.legacy_density_uq_enabled:
            raise RuntimeError(
                'Legacy Density-UQ configuration mismatch: environment='
                f'{self.legacy_density_uq_enabled}, model_head='
                f'{density_head_enabled}'
            )
        if self.stage2p_engineering_smoke:
            stage2_head = getattr(self.model, 'pts_bbox_head', None)
            if (
                getattr(self.model, 'stage2_spatial_uq_source', None)
                != 'external_oracle'
                or getattr(stage2_head, 'stage2_input_semantics', None)
                != 'task_risk_k'
                or getattr(stage2_head, 'stage2p_engineering_smoke', None)
                is not True
            ):
                raise RuntimeError(
                    'Stage2-P engineering smoke model/checkpoint contract differs'
                )
            if self.risk_mode != 'off' or self.planning_response_mode != 'off':
                raise RuntimeError(
                    'Stage2-P engineering smoke forbids governor and privileged response'
                )

        if self.observation_uq_checkpoint_path:
            self.observation_uq_adapter, self.observation_uq_metadata = (
                load_frozen_pairwise_adapter(
                    self.observation_uq_checkpoint_path,
                    expected_sha256=(
                        self.observation_uq_checkpoint_sha256 or None
                    ),
                    device='cuda',
                )
            )
            self.model.capture_observation_uq_features = True
        elif self.risk_mode == 'aligned_learned':
            raise RuntimeError(
                'aligned_learned requires the frozen spatial observation-UQ '
                'adapter; legacy Density UQ fallback is retired'
            )

        self.model.uq_inference_mode = self.uq_mode
        self.model.uq_inference_conditioning = self.uq_conditioning
        self.stage2_artifact_writer = None
        self.stage2_artifact_stride_steps = int(os.environ.get(
            'ORION_STAGE2_ARTIFACT_STRIDE_STEPS', '10'
        ))
        if self.stage2_artifact_stride_steps <= 0:
            raise RuntimeError(
                'ORION_STAGE2_ARTIFACT_STRIDE_STEPS must be positive'
            )
        stage2_artifact_root = os.environ.get(
            'ORION_STAGE2_ARTIFACT_ROOT', ''
        ).strip()
        if stage2_artifact_root:
            if not bool(getattr(self.model, 'use_stage2_spatial_uq', False)):
                raise RuntimeError(
                    'Stage-2 artifact capture requires the new spatial VLM path'
                )
            if getattr(self.model, 'stage2_spatial_uq_source', None) != 'learned_adapter':
                raise RuntimeError(
                    'closed-loop artifact capture currently requires learned_adapter source'
                )
            runtime = getattr(self.model, 'stage1_spatial_uq_runtime', None)
            if runtime is None:
                raise RuntimeError('Stage-2 artifact capture lacks Stage-1 runtime')
            route_group = os.environ.get(
                'ORION_STAGE2_ARTIFACT_ROUTE_GROUP', ''
            ).strip()
            self.stage2_artifact_writer = Stage2ArtifactWriter(
                stage2_artifact_root,
                route_group=route_group,
                uq_source='learned_stage1_spatial_uq',
                camera_order=CAMERA_ORDER,
                stage1_checkpoint_sha256=runtime.metadata['sha256'],
            )
        print(
            '[ClosedLoop] '
            f'uq_mode={self.uq_mode}, conditioning={self.uq_conditioning}, '
            f'corruption={self.closedloop_corruption or "clean"}, '
            f'severity={self.corruption_severity}, '
            f'views={self.corruption_view_spec}, risk_mode={self.risk_mode}, '
            f'native_risk_oracle={bool(self.risk_oracle_timed_window)}, '
            f'planning_response={self.planning_response_mode}, '
            f'legacy_density_uq={self.legacy_density_uq_enabled}, '
            f'risk_threshold={self.risk_governor.threshold}, '
            f'risk_saturation={self.risk_governor.saturation}, '
            f'risk_min_speed={self.risk_governor.min_speed}, '
            f'observation_uq={bool(self.observation_uq_adapter)}, '
            f'safety_telemetry={self.closedloop_safety_telemetry}, '
            f'safety_schema={CLOSEDLOOP_SAFETY_SCHEMA_VERSION}'
        )
        stage2_source = getattr(
            self.model, 'stage2_spatial_uq_source', 'disabled'
        )
        stage1_runtime = getattr(
            self.model, 'stage1_spatial_uq_runtime', None
        )
        print(
            '[SpatialStage2] '
            f'enabled={bool(getattr(self.model, "use_stage2_spatial_uq", False))}, '
            f'source={stage2_source}, '
            f'stage1_checkpoint_sha256='
            f'{stage1_runtime.metadata["sha256"] if stage1_runtime is not None else None}, '
            'legacy_density_uq_used=False, '
            'legacy_scalar_governor_used=False',
            flush=True,
        )
        if self.observation_uq_metadata is not None:
            print(
                '[ObservationUQ] '
                f'checkpoint_sha256={self.observation_uq_metadata["sha256"]}, '
                f'front_view={self.observation_uq_front_view}, '
                f'baseline_seconds=['
                f'{self.observation_uq_calibrator.baseline_start_seconds},'
                f'{self.observation_uq_calibrator.baseline_end_seconds})',
                flush=True,
            )

        self.takeover = False
        self.stop_time = 0
        self.takeover_time = 0
        self.save_path = None
        self.control_trace_path = None
        self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        self.lat_ref, self.lon_ref = 42.0, 2.0
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0	
        self.prev_control = control
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            # string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string = self.save_name
            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            (self.save_path / 'rgb_front').mkdir()
            (self.save_path / 'rgb_front_model_input').mkdir()
            (self.save_path / 'rgb_front_model_tensor').mkdir()
            (self.save_path / 'rgb_front_right').mkdir()
            (self.save_path / 'rgb_front_left').mkdir()
            (self.save_path / 'rgb_back').mkdir()
            (self.save_path / 'rgb_back_right').mkdir()
            (self.save_path / 'rgb_back_left').mkdir()
            (self.save_path / 'meta').mkdir()
            (self.save_path / 'bev').mkdir()
            self.control_trace_path = self.save_path / 'control_trace.jsonl'
   
        # write extrinsics directly
        self.lidar2img = {
        'CAM_FRONT':np.array([[ 1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -9.52000000e+02],
                                  [ 0.00000000e+00,  4.50000000e+02, -1.14251841e+03, -8.09704417e+02],
                                  [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -1.19000000e+00],
                                 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
          'CAM_FRONT_LEFT':np.array([[ 6.03961325e-14,  1.39475744e+03,  0.00000000e+00, -9.20539908e+02],
                                   [-3.68618420e+02,  2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                   [-8.19152044e-01,  5.73576436e-01,  0.00000000e+00, -8.29094072e-01],
                                   [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
          'CAM_FRONT_RIGHT':np.array([[ 1.31064327e+03, -4.77035138e+02,  0.00000000e+00,-4.06010608e+02],
                                       [ 3.68618420e+02,  2.58109396e+02, -1.14251841e+03,-6.47296750e+02],
                                    [ 8.19152044e-01,  5.73576436e-01,  0.00000000e+00,-8.29094072e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]),
         'CAM_BACK':np.array([[-5.60166031e+02, -8.00000000e+02,  0.00000000e+00, -1.28800000e+03],
                     [ 5.51091060e-14, -4.50000000e+02, -5.60166031e+02, -8.58939847e+02],
                     [ 1.22464680e-16, -1.00000000e+00,  0.00000000e+00, -1.61000000e+00],
                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_BACK_LEFT':np.array([[-1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -6.84385123e+02],
                                  [-4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                  [-9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
                                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
  
        'CAM_BACK_RIGHT': np.array([[ 3.60989788e+02, -1.34723223e+03,  0.00000000e+00, -1.04238127e+02],
                                    [ 4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                    [ 9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        }
        self.lidar2cam = {
        'CAM_FRONT':np.array([[ 1.  ,  0.  ,  0.  ,  0.  ],
                                 [ 0.  ,  0.  , -1.  , -0.24],
                                 [ 0.  ,  1.  ,  0.  , -1.19],
                              [ 0.  ,  0.  ,  0.  ,  1.  ]]),
        'CAM_FRONT_LEFT':np.array([[ 0.57357644,  0.81915204,  0.  , -0.22517331],
                                      [ 0.        ,  0.        , -1.  , -0.24      ],
                                   [-0.81915204,  0.57357644,  0.  , -0.82909407],
                                   [ 0.        ,  0.        ,  0.  ,  1.        ]]),
          'CAM_FRONT_RIGHT':np.array([[ 0.57357644, -0.81915204, 0.  ,  0.22517331],
                                   [ 0.        ,  0.        , -1.  , -0.24      ],
                                   [ 0.81915204,  0.57357644,  0.  , -0.82909407],
                                   [ 0.        ,  0.        ,  0.  ,  1.        ]]),
        'CAM_BACK':np.array([[-1. ,  0.,  0.,  0.  ],
                             [ 0. ,  0., -1., -0.24],
                             [ 0. , -1.,  0., -1.61],
                             [ 0. ,  0.,  0.,  1.  ]]),
     
        'CAM_BACK_LEFT':np.array([[-0.34202014,  0.93969262,  0.  , -0.25388956],
                                  [ 0.        ,  0.        , -1.  , -0.24      ],
                                  [-0.93969262, -0.34202014,  0.  , -0.49288953],
                                  [ 0.        ,  0.        ,  0.  ,  1.        ]]),
  
        'CAM_BACK_RIGHT':np.array([[-0.34202014, -0.93969262,  0.  ,  0.25388956],
                                  [ 0.        ,  0.         , -1.  , -0.24      ],
                                  [ 0.93969262, -0.34202014 ,  0.  , -0.49288953],
                                  [ 0.        ,  0.         ,  0.  ,  1.        ]])
        }
        self.lidar2ego = np.array([[ 0. ,  1. ,  0. , -0.39],
                                   [-1. ,  0. ,  0. ,  0.  ],
                                   [ 0. ,  0. ,  1. ,  1.84],
                                   [ 0. ,  0. ,  0. ,  1.  ]])
        
        topdown_extrinsics =  np.array([[0.0, -0.0, -1.0, 50.0], [0.0, 1.0, -0.0, 0.0], [1.0, -0.0, 0.0, -0.0], [0.0, 0.0, 0.0, 1.0]])
        unreal2cam = np.array([[0,1,0,0], [0,0,-1,0], [1,0,0,0], [0,0,0,1]])
        self.coor2topdown = unreal2cam @ topdown_extrinsics
        topdown_intrinsics = np.array([[548.993771650447, 0.0, 256.0, 0], [0.0, 548.993771650447, 256.0, 0], [0.0, 0.0, 1.0, 0], [0, 0, 0, 1.0]])
        self.coor2topdown = topdown_intrinsics @ self.coor2topdown

    def _init(self):
        try:
            locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
            lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
            EARTH_RADIUS_EQUA = 6378137.0
            def equations(vars):
                x, y = vars
                eq1 = lon * math.cos(x * math.pi / 180) - (locx * x * 180) / (math.pi * EARTH_RADIUS_EQUA) - math.cos(x * math.pi / 180) * y
                eq2 = math.log(math.tan((lat + 90) * math.pi / 360)) * EARTH_RADIUS_EQUA * math.cos(x * math.pi / 180) + locy - math.cos(x * math.pi / 180) * EARTH_RADIUS_EQUA * math.log(math.tan((90 + x) * math.pi / 360))
                return [eq1, eq2]
            initial_guess = [0, 0]
            solution = fsolve(equations, initial_guess)
            self.lat_ref, self.lon_ref = solution[0], solution[1]
        except Exception as e:
            print(e, flush=True)
            self.lat_ref, self.lon_ref = 0, 0        
        self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)
        self._corruption_route_points = np.asarray([
            [transform.location.x, transform.location.y]
            for transform, _ in self._global_plan_world_coord
        ], dtype=np.float64)
        print(
            f'[RoutePlanner] global_plan={len(self._global_plan)}, '
            f'world_plan={len(self._global_plan_world_coord)}, '
            f'planner_route={len(self._route_planner.route)}, '
            f'progress_start={self._corruption_route_points[0].tolist()}, '
            f'progress_end={self._corruption_route_points[-1].tolist()}',
            flush=True,
        )
        self.initialized = True
        self.metric_info = {}
  
  

    def sensors(self):
        sensors =[
                # camera rgb
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.80, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_RIGHT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -2.0, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': 1600, 'height': 900, 'fov': 110,
                    'id': 'CAM_BACK'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_RIGHT'
                },
                # imu
                {
                    'type': 'sensor.other.imu',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'IMU'
                },
                # gps
                {
                    'type': 'sensor.other.gnss',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'GPS'
                },
                # speed
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'SPEED'
                },       
            ]
        if IS_BENCH2DRIVE:
            sensors += [
                    {	
                        'type': 'sensor.camera.rgb',
                        'x': 0.0, 'y': 0.0, 'z': 50.0,
                        'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                        'width': 512, 'height': 512, 'fov': 5 * 10.0,
                        'id': 'bev'
                    }]
        return sensors

    def tick(self, input_data):
        self.step += 1
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        imgs = {}
        for cam in CAMERA_ORDER:
            # img = cv2.cvtColor(input_data[cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
            img = input_data[cam][1][:, :, :3]
            _, img = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            imgs[cam] = img
        
        #NOTE@Jianfeng: we directly use BGR image and let pipeline do the convert
        # breakpoint()
        # cv2.imwrite('./work_dirs/tick_input_img.jpg', img)
        # bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)

        bev = input_data['bev'][1][:, :, :3]
        gps = input_data['GPS'][1][:2]
        speed = input_data['SPEED'][1]['speed']
        compass = input_data['IMU'][1][-1]
        acceleration = input_data['IMU'][1][:3]
        angular_velocity = input_data['IMU'][1][3:6]

        pos = self.gps_to_location(gps)
        (_, curr_command), (near_node, near_command) = self._route_planner.run_step(pos)

        if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
            compass = 0.0
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)

        result = {
                'imgs': imgs,
                'gps': gps,
                'pos':pos,
                'speed': speed,
                'compass': compass,
                'bev': bev,
                'acceleration':acceleration,
                'angular_velocity':angular_velocity,
                'command_curr':curr_command,
                'command_near':near_command,
                'command_near_xy':near_node
                }
        
        return result
    
    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        if not self.native_glare_readback_recorded:
            render_readback = (
                readback_native_motion_blur_condition(
                    self.sensor_interface,
                    self.native_motion_blur_profile,
                )
                if self.native_motion_blur_profile != 'none'
                else readback_render_condition(
                    self.sensor_interface,
                    CarlaDataProvider.get_world(),
                    self.native_glare_profile,
                )
            )
            output_root = os.environ.get('OUTPUT_ROOT')
            if output_root:
                record_render_condition_readback(
                    output_root,
                    self.native_render_condition,
                    render_readback,
                )
            elif (
                self.native_glare_profile != 'none'
                or self.native_motion_blur_profile != 'none'
            ):
                raise RuntimeError(
                    'OUTPUT_ROOT is required to persist native-render readback'
                )
            self.native_glare_readback_recorded = True
        tick_data = self.tick(input_data)
        sim_time_seconds = self.step / 20.0
        ego_location = self.hero_actor.get_transform().location
        route_progress = project_route_progress(
            (ego_location.x, ego_location.y), self._corruption_route_points
        )
        if self.corruption_timed_window is not None:
            within_corruption_window = self.corruption_timed_window.is_active(
                route_progress, sim_time_seconds
            )
        elif self.corruption_start_progress is not None:
            within_corruption_window = bool(
                self.corruption_start_progress <= route_progress
                < self.corruption_end_progress
            )
        else:
            within_corruption_window = bool(
                self.corruption_start_seconds <= sim_time_seconds
                < self.corruption_end_seconds
            )
        corruption_active = bool(
            self.closedloop_corruption and within_corruption_window
        )
        if self.risk_oracle_timed_window is not None:
            oracle_event_active = self.risk_oracle_timed_window.is_active(
                route_progress, sim_time_seconds
            )
            oracle_event_schedule = 'native_event_route_triggered_timed'
        else:
            oracle_event_active = bool(
                corruption_active and self.oracle_corruption_relevant
            )
            oracle_event_schedule = 'corruption_state'

        paired_waterdrop_front_bgr = None
        corruption_metadata = None
        if (
            corruption_active
            and self.closedloop_corruption
            == 'lens_waterdrop_paired_template'
        ):
            if self.paired_waterdrop_template is None:
                raise RuntimeError('paired waterdrop template was not loaded')
            clean_front_bgr = tick_data['imgs']['CAM_FRONT']
            paired_result = apply_paired_waterdrop_template(
                clean_front_bgr[..., ::-1].copy(),
                template=self.paired_waterdrop_template,
                profile=self.paired_waterdrop_profile,
                require_resolution=True,
            )
            paired_waterdrop_front_bgr = paired_result.image[..., ::-1].copy()
            corruption_metadata = dict(paired_result.metadata)
            corruption_metadata.update({
                'corruption': 'lens_waterdrop_paired_template',
                'applied': True,
                'view_indices': [0],
                'application_stage': 'pre_pipeline_1600x900_front_rgb',
            })
        results = {}
        results['lidar2img'] = []
        results['lidar2cam'] = []
        results['cam_intrinsic'] = []
        results['img'] = []
        results['folder'] = ' '
        results['scene_token'] = ' '  
        results['frame_idx'] = self.step
        results['timestamp'] = self.step / 20
        results['box_type_3d'], _ = get_box_type('LiDAR')
  
        for cam in CAMERA_ORDER:
            results['lidar2img'].append(self.lidar2img[cam])
            results['lidar2cam'].append(self.lidar2cam[cam])
            results['cam_intrinsic'].append(np.matmul(self.lidar2img[cam], np.linalg.inv(self.lidar2cam[cam])))
            results['img'].append(
                paired_waterdrop_front_bgr
                if cam == 'CAM_FRONT'
                and paired_waterdrop_front_bgr is not None
                else tick_data['imgs'][cam]
            )
        results['lidar2img'] = np.stack(results['lidar2img'],axis=0)
        results['lidar2cam'] = np.stack(results['lidar2cam'],axis=0)
        raw_theta = tick_data['compass']   if not np.isnan(tick_data['compass']) else 0
        ego_theta = -raw_theta + np.pi/2
        rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))
        can_bus = np.zeros(18)
        can_bus[0] = tick_data['pos'][0]
        can_bus[1] = -tick_data['pos'][1]
        can_bus[3:7] = rotation
        can_bus[7] = tick_data['speed']
        can_bus[10:13] = tick_data['acceleration']
        can_bus[11] *= -1
        can_bus[13:16] = -tick_data['angular_velocity']
        can_bus[16] = ego_theta
        can_bus[17] = ego_theta / np.pi * 180 
        results['can_bus'] = can_bus
        command = tick_data['command_curr']
        results['command'] = command2nohot(tick_data['command_curr'])
        results['ego_fut_cmd'] = command2hot(tick_data['command_curr'])
  
        theta_to_lidar = raw_theta
        command_near_xy = np.array([tick_data['command_near_xy'][0]-can_bus[0],-tick_data['command_near_xy'][1]-can_bus[1]])
        rotation_matrix = np.array([[np.cos(theta_to_lidar),-np.sin(theta_to_lidar)],[np.sin(theta_to_lidar),np.cos(theta_to_lidar)]])
        local_command_xy = rotation_matrix @ command_near_xy
  
        ego2world = np.eye(4)
        ego2world[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
        ego2world[0:2,3] = can_bus[0:2]
        ego_pose = ego2world
        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        results['ego_pose'] = ego_pose
        results['ego_pose_inv'] = ego_pose_inv
        lidar2global = ego2world @ self.lidar2ego
        ego_pose = lidar2global
        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        results['ego_pose'] = ego_pose
        results['ego_pose_inv'] = ego_pose_inv
        results['lidar2ego'] = self.lidar2ego
        results['l2g_r_mat'] = lidar2global[0:3,0:3]
        results['l2g_t'] = lidar2global[0:3,3]
        stacked_imgs = np.stack(results['img'],axis=-1)
        results['img_shape'] = stacked_imgs.shape
        results['ori_shape'] = stacked_imgs.shape
        results['pad_shape'] = stacked_imgs.shape
        results = self.inference_only_pipeline(results)
        self.device="cuda"
        input_data_batch = mm_collate_to_batch_form([results], samples_per_gpu=1)
        for key, data in input_data_batch.items():
            if key != 'img_metas':
                if torch.is_tensor(data[0]):
                    data[0] = data[0].to(self.device)
            if key == 'input_ids':
                for i in range(len(data[0])):
                    for k in range(len(data[0][i])):
                        # print(data[0][i][k])
                        data[0][i][k] = data[0][i][k].to(self.device)

        tick_data['model_input_front'] = (
            paired_waterdrop_front_bgr
            if paired_waterdrop_front_bgr is not None
            else tick_data['imgs']['CAM_FRONT']
        )
        if self.stale_frame_buffer is not None:
            temporal_result = self.stale_frame_buffer.apply(
                input_data_batch['img'][0],
                timestamp_seconds=sim_time_seconds,
                active=corruption_active,
                view_indices=self.corruption_view_indices,
            )
            input_data_batch['img'][0] = temporal_result.images
            if corruption_active:
                corruption_metadata = temporal_result.metadata
        elif (
            corruption_active
            and self.closedloop_corruption
            != 'lens_waterdrop_paired_template'
        ):
            if self.closedloop_corruption in {'blur', 'dark'}:
                input_data_batch['img'][0] = corrupt_multiview_images(
                    input_data_batch['img'][0],
                    corruption=self.closedloop_corruption,
                    severity=self.corruption_severity,
                    view_indices=self.corruption_view_indices,
                )
            else:
                corruption_result = corrupt_multiview_images_with_metadata(
                    input_data_batch['img'][0],
                    corruption=self.closedloop_corruption,
                    severity=self.corruption_severity,
                    view_indices=self.corruption_view_indices,
                    seed=self.corruption_seed,
                    region=self.corruption_region,
                )
                input_data_batch['img'][0] = corruption_result.images
                corruption_metadata = corruption_result.metadata.to_dict()
            if 0 in self.corruption_view_indices:
                preview_region = (
                    corruption_metadata['normalized_region']
                    if corruption_metadata is not None
                    else self.corruption_region
                )
                tick_data['model_input_front'] = render_front_corruption_preview(
                    tick_data['imgs']['CAM_FRONT'],
                    self.closedloop_corruption,
                    self.corruption_severity,
                    preview_region,
                )
        if (
            corruption_metadata is not None
            and self.corruption_visual_approval is not None
        ):
            corruption_metadata = dict(corruption_metadata)
            corruption_metadata['visual_approval'] = (
                self.corruption_visual_approval
            )
        # Save the tensor that is actually passed to ORION, rather than a
        # reconstruction from the raw Q20 frame.  Restrict the GPU-to-CPU copy
        # to the existing diagnostic save cadence.
        if SAVE_PATH is not None and self.step % 10 == 0:
            model_input_front_tensor = (
                normalized_front_tensor_to_bgr(input_data_batch['img'][0])
                .detach()
                .cpu()
                .numpy()
            )
            if (
                model_input_front_tensor.shape != (640, 640, 3)
                or model_input_front_tensor.dtype != np.uint8
            ):
                raise RuntimeError(
                    'Exact ORION CAM_FRONT tensor capture must be uint8 '
                    '640x640x3; got shape=%r dtype=%s'
                    % (model_input_front_tensor.shape, model_input_front_tensor.dtype)
                )
            tick_data['model_input_front_tensor'] = model_input_front_tensor
            if (
                corruption_metadata is not None
                and (
                    (
                        corruption_metadata.get('corruption') == 'front_stale'
                        and corruption_metadata.get('applied') is True
                    )
                    or corruption_metadata.get('corruption')
                    == 'lens_waterdrop_paired_template'
                )
            ):
                # Preserve a visually honest preview at the save cadence.  The
                # exact tensor directory remains the authoritative artifact.
                tick_data['model_input_front'] = model_input_front_tensor.copy()
                    
        custom_wrap_fp16_model(self.model)
        stage2_external_k_active = False
        if self.stage2p_engineering_smoke:
            stage2_external_k_active = self.stage2_external_k_window.is_active(
                route_progress, sim_time_seconds
            )
            task_risk_k = build_stage2_external_k_map(
                camera_index=self.stage2_external_k_camera_index,
                region=self.stage2_external_k_region,
                active=stage2_external_k_active,
                strength=self.stage2_external_k_strength,
                grid_size=self.stage2_external_k_grid_size,
                device=input_data_batch['img'][0].device,
                dtype=input_data_batch['img'][0].dtype,
            )
            # Match the nested one-sample contract consumed by ORION.forward_test.
            input_data_batch['stage2_spatial_uq'] = [[task_risk_k]]
        output_data_batch = self.model(input_data_batch, return_loss=False)
        if (
            self.stage2_artifact_writer is not None
            and self.step % self.stage2_artifact_stride_steps == 0
        ):
            stage2_head = getattr(self.model, 'pts_bbox_head', None)
            planning_context = getattr(
                stage2_head, 'stage2_planning_context', None
            )
            observation_uq = getattr(
                stage2_head, 'stage2_observation_uq', None
            )
            task_context = getattr(
                stage2_head, 'stage2_task_context', None
            )
            runtime_output = getattr(
                self.model, 'stage1_spatial_uq_runtime_output', None
            )
            if (
                planning_context is None
                or observation_uq is None
                or task_context is None
                or runtime_output is None
            ):
                raise RuntimeError(
                    'ORION did not expose complete Stage-2 capture tensors'
                )
            self.stage2_artifact_writer.write(
                step=self.step,
                planning_context=planning_context,
                task_context=task_context,
                observation_uq=observation_uq,
                raw_observation_uq=runtime_output.raw_score,
                metadata={
                    'sim_time_seconds': sim_time_seconds,
                    'route_progress': route_progress,
                    'baseline_ready': runtime_output.baseline_ready,
                    'baseline_count': runtime_output.baseline_count,
                    'previous_valid': runtime_output.previous_valid,
                    'capture_stride_steps': self.stage2_artifact_stride_steps,
                },
            )
        base_out_truck = (
            output_data_batch[0]['pts_bbox']['ego_fut_preds'].cpu().numpy()
        )
        out_truck = base_out_truck.copy()
        closedloop_safety = self._collect_closedloop_safety_geometry()
        planning_response_record = None
        if self.stage2p_engineering_smoke:
            stage2_output = getattr(
                getattr(self.model, 'pts_bbox_head', None),
                'stage2_task_output',
                None,
            )
            if stage2_output is None:
                raise RuntimeError('Stage2-P smoke produced no trajectory output')
            stage2_residual = stage2_output.trajectory_residual
            if (
                tuple(stage2_residual.shape) != (1, 6, 2)
                or not bool(torch.isfinite(stage2_residual).all())
            ):
                raise RuntimeError('Stage2-P trajectory residual is invalid')
            if (
                not stage2_external_k_active
                and int(torch.count_nonzero(stage2_residual)) != 0
            ):
                raise RuntimeError('Zero external K changed native trajectory')
            residual_numpy = stage2_residual[0].detach().float().cpu().numpy()
            out_truck = out_truck + residual_numpy
            planning_response_record = {
                'mode': 'stage2p_controlled_k_engineering_smoke',
                'external_k_active': bool(stage2_external_k_active),
                'external_k_camera': CAMERA_ORDER[
                    self.stage2_external_k_camera_index
                ],
                'external_k_region': list(self.stage2_external_k_region),
                'external_k_strength': self.stage2_external_k_strength,
                'global_gate': float(stage2_output.global_gate[0].detach().cpu()),
                'trajectory_residual_m': residual_numpy.tolist(),
                'formal_stage2p_ready': False,
                'closed_loop_safety_claim': False,
            }
        elif self.planning_response_mode in {
            'privileged_bounded_crossing',
            'privileged_braking_aware_crossing',
            'privileged_dynamic_yield',
        }:
            if not closedloop_safety or not closedloop_safety.get('available'):
                raise RuntimeError(
                    'privileged planning response requires available actor telemetry'
                )
            planning_safety = closedloop_safety
            if self.planning_response_mode in {
                'privileged_bounded_crossing',
                'privileged_braking_aware_crossing',
            }:
                planning_safety = select_actor_categories(
                    closedloop_safety,
                    self.planning_actor_categories,
                )
            conflict = evaluate_trajectory_conflicts(
                base_out_truck.tolist(),
                planning_safety,
                config=self.planning_conflict_config,
            )
            yield_label = self.dynamic_yield_labeler.update(
                conflict,
                sim_time_seconds,
            )
            braking_profile = None
            if (
                self.planning_response_mode
                == 'privileged_braking_aware_crossing'
            ):
                target_plan, braking_profile = (
                    build_braking_aware_crossing_trajectory(
                        base_out_truck.tolist(),
                        yield_label,
                        tick_data['speed'],
                        config=self.bounded_crossing_expert_config,
                    )
                )
            else:
                target_plan = build_safe_yield_trajectory(
                    base_out_truck.tolist(),
                    yield_label,
                )
            out_truck = np.asarray(target_plan, dtype=base_out_truck.dtype)
            planning_response_record = {
                'mode': self.planning_response_mode,
                'actor_categories': (
                    list(self.planning_actor_categories)
                    if self.planning_response_mode
                    in {
                        'privileged_bounded_crossing',
                        'privileged_braking_aware_crossing',
                    }
                    else None
                ),
                'braking_profile': (
                    braking_profile.to_dict()
                    if braking_profile is not None else None
                ),
                'base_plan_cumulative_m': base_out_truck.tolist(),
                'target_plan_cumulative_m': out_truck.tolist(),
                'trajectory_residual_m': trajectory_residual(
                    base_out_truck.tolist(), out_truck.tolist()
                ),
                'conflict': conflict.to_dict(),
                'yield_label': yield_label.to_dict(),
            }
        elif self.planning_response_mode == 'privileged_dynamics_aware_yield':
            if not closedloop_safety or not closedloop_safety.get('available'):
                raise RuntimeError(
                    'dynamics-aware planning response requires actor telemetry'
                )
            raw_conflict = evaluate_trajectory_conflicts(
                base_out_truck.tolist(),
                closedloop_safety,
                config=self.planning_conflict_config,
            )
            expert_state_before = self.dynamics_aware_yield_expert.state
            should_query_map = bool(
                raw_conflict.has_conflict or expert_state_before != 'go'
            )
            junction_entry = (
                self._query_path_junction_entry(
                    raw_conflict,
                    closedloop_safety,
                )
                if should_query_map
                else None
            )
            conflict_resolution = resolve_junction_scoped_conflict(
                raw_conflict,
                junction_entry,
                expert_state=expert_state_before,
            )
            junction_scoped_conflict = (
                conflict_resolution.junction_scoped_conflict
            )
            effective_conflict = conflict_resolution.effective_conflict
            junction_entry_distance = (
                conflict_resolution.geometry_entry_distance_m
            )
            ego_forward_extent = float(
                closedloop_safety['ego']['extent_xy_m'][0]
            )
            yield_geometry = compute_junction_yield_geometry(
                junction_entry_distance,
                ego_forward_extent,
                tick_data['speed'],
                config=self.dynamic_yield_expert_config,
            )
            yield_label = self.dynamics_aware_yield_expert.update(
                effective_conflict,
                yield_geometry,
                sim_time_seconds,
            )
            target_plan = build_dynamics_aware_yield_trajectory(
                base_out_truck.tolist(),
                yield_label,
                tick_data['speed'],
                config=self.dynamic_yield_expert_config,
            )
            out_truck = np.asarray(target_plan, dtype=base_out_truck.dtype)
            planning_response_record = {
                'mode': self.planning_response_mode,
                'base_plan_cumulative_m': base_out_truck.tolist(),
                'target_plan_cumulative_m': out_truck.tolist(),
                'trajectory_residual_m': trajectory_residual(
                    base_out_truck.tolist(), out_truck.tolist()
                ),
                'raw_conflict': raw_conflict.to_dict(),
                'effective_conflict': effective_conflict.to_dict(),
                'junction_scope': {
                    'queried': should_query_map,
                    'junction_scoped_conflict': junction_scoped_conflict,
                    'entry': (
                        junction_entry.to_dict()
                        if junction_entry is not None
                        else None
                    ),
                },
                'yield_geometry': yield_geometry.to_dict(),
                'yield_label': yield_label.to_dict(),
            }
        steer_traj, throttle_traj, brake_traj, metadata_traj = self.pidcontroller.control_pid(out_truck, tick_data['speed'], local_command_xy)
        if brake_traj < 0.05: brake_traj = 0.0
        if throttle_traj > brake_traj: brake_traj = 0.0
        if tick_data['speed']>5:
            throttle_traj = 0
        uq_output = getattr(
            getattr(self.model, 'pts_bbox_head', None), 'uq_output', None
        )
        density_uq_score = None
        if self.legacy_density_uq_enabled and uq_output is not None:
            density_uq_score = float(
                uq_output.score.detach().float().mean().cpu()
            )
        raw_uq_score = density_uq_score
        observation_uq_record = None
        if self.observation_uq_adapter is not None:
            captured_features = getattr(
                self.model, 'observation_uq_features', None
            )
            if captured_features is None:
                raise RuntimeError(
                    'ORION did not expose the requested observation-UQ features'
                )
            if captured_features.ndim != 5:
                raise RuntimeError(
                    'ORION observation-UQ features must have [B,V,C,H,W] shape'
                )
            current_features = captured_features.detach().permute(
                0, 1, 3, 4, 2
            ).contiguous()
            previous_valid = torch.tensor(
                [self.observation_uq_previous_valid],
                dtype=torch.bool,
                device=current_features.device,
            )
            observation_score_map = self.observation_uq_adapter(
                current_features,
                self.observation_uq_previous_features,
                previous_valid,
            )
            aggregate = aggregate_observation_evidence(
                observation_score_map,
                front_view_index=self.observation_uq_front_view,
            )
            spatial_summary = None
            if self.corruption_region is not None:
                spatial_summary = summarize_spatial_observation_evidence(
                    observation_score_map,
                    self.corruption_region,
                    front_view_index=self.observation_uq_front_view,
                )
            calibrated = self.observation_uq_calibrator.update(
                aggregate.front_raw_score,
                sim_time_seconds,
            )
            raw_uq_score = calibrated.filtered_score
            observation_uq_record = {
                'checkpoint_sha256': self.observation_uq_metadata['sha256'],
                'front_view_index': self.observation_uq_front_view,
                'previous_valid': self.observation_uq_previous_valid,
                'feature_shape': list(current_features.shape),
                'aggregate': aggregate.to_dict(),
                'calibration': calibrated.to_dict(),
                'camera_order': CAMERA_ORDER,
                'front_pooled_grid': pool_observation_evidence_grid(
                    observation_score_map,
                    view_index=self.observation_uq_front_view,
                ),
                'pooled_grids': pool_observation_evidence_grids(
                    observation_score_map,
                ),
                'spatial_summary': (
                    spatial_summary.to_dict()
                    if spatial_summary is not None else None
                ),
            }
            self.observation_uq_previous_features = current_features.detach()
            self.observation_uq_previous_valid = True
        governed_throttle, governed_brake, risk_decision = (
            self.risk_governor.apply(
                throttle=float(throttle_traj),
                brake=float(brake_traj),
                speed=float(tick_data['speed']),
                raw_score=raw_uq_score,
                step=self.step,
                oracle_active=oracle_event_active,
            )
        )
        control = carla.VehicleControl()
        self.pid_metadata = metadata_traj
        self.pid_metadata['agent'] = 'only_traj'
        control.steer = np.clip(float(steer_traj), -1, 1)
        control.throttle = np.clip(float(governed_throttle), 0, 0.75)
        control.brake = np.clip(float(governed_brake), 0, 1)
        self.pid_metadata['steer'] = control.steer
        self.pid_metadata['throttle'] = control.throttle
        self.pid_metadata['brake'] = control.brake
        self.pid_metadata['steer_traj'] = float(steer_traj)
        self.pid_metadata['throttle_traj'] = float(throttle_traj)
        self.pid_metadata['brake_traj'] = float(brake_traj)
        self.pid_metadata['base_plan'] = base_out_truck.tolist()
        self.pid_metadata['plan'] = out_truck.tolist()
        self.pid_metadata['planning_response'] = planning_response_record
        self.pid_metadata['command'] = command
        self.pid_metadata['command_near_xy'] = command_near_xy.tolist()
        self.pid_metadata['local_command_xy '] = local_command_xy.tolist()
        if density_uq_score is not None:
            self.pid_metadata['density_uq_score'] = density_uq_score
        if observation_uq_record is not None:
            self.pid_metadata['observation_uq'] = observation_uq_record
        self.pid_metadata['uq_mode'] = self.uq_mode
        self.pid_metadata['uq_conditioning'] = self.uq_conditioning
        self.pid_metadata['corruption'] = self.closedloop_corruption or 'clean'
        self.pid_metadata['corruption_severity'] = self.corruption_severity
        self.pid_metadata['corruption_views'] = self.corruption_view_spec
        self.pid_metadata['corruption_seed'] = self.corruption_seed
        self.pid_metadata['corruption_region'] = self.corruption_region
        self.pid_metadata['corruption_metadata'] = corruption_metadata
        self.pid_metadata['corruption_visual_approval'] = (
            self.corruption_visual_approval
        )
        self.pid_metadata['oracle_corruption_relevant'] = (
            self.oracle_corruption_relevant
        )
        self.pid_metadata['oracle_event_active'] = oracle_event_active
        self.pid_metadata['oracle_event_schedule'] = oracle_event_schedule
        self.pid_metadata['oracle_event_trigger_time_seconds'] = (
            self.risk_oracle_timed_window.trigger_time_seconds
            if self.risk_oracle_timed_window is not None else None
        )
        self.pid_metadata['corruption_active'] = corruption_active
        self.pid_metadata['corruption_schedule_mode'] = self.corruption_schedule_mode
        self.pid_metadata['corruption_trigger_time_seconds'] = (
            self.corruption_timed_window.trigger_time_seconds
            if self.corruption_timed_window is not None else None
        )
        self.pid_metadata['route_progress'] = route_progress
        self.pid_metadata['risk_governor'] = risk_decision.to_dict()
        adapter_delta = getattr(self.model, 'uq_vision_adapter_delta', None)
        if adapter_delta is not None:
            self.pid_metadata['uq_vision_adapter_delta'] = float(
                adapter_delta.detach().float().cpu()
            )
        self.pid_metadata['closedloop_safety'] = closedloop_safety
        metric_info = self.get_metric_info()
        self.metric_info[self.step] = metric_info     
        self._append_control_trace(
            sim_time_seconds=sim_time_seconds,
            route_progress=route_progress,
            speed=float(tick_data['speed']),
            corruption_active=corruption_active,
            corruption_schedule_mode=self.corruption_schedule_mode,
            corruption_trigger_time_seconds=(
                self.corruption_timed_window.trigger_time_seconds
                if self.corruption_timed_window is not None else None
            ),
            corruption_elapsed_seconds=(
                self.corruption_timed_window.elapsed_seconds(sim_time_seconds)
                if self.corruption_timed_window is not None else None
            ),
            corruption_metadata=corruption_metadata,
            oracle_corruption_relevant=self.oracle_corruption_relevant,
            oracle_event_active=oracle_event_active,
            oracle_event_schedule=oracle_event_schedule,
            oracle_event_trigger_time_seconds=(
                self.risk_oracle_timed_window.trigger_time_seconds
                if self.risk_oracle_timed_window is not None else None
            ),
            oracle_event_elapsed_seconds=(
                self.risk_oracle_timed_window.elapsed_seconds(sim_time_seconds)
                if self.risk_oracle_timed_window is not None else None
            ),
            raw_uq_score=raw_uq_score,
            density_uq_score=density_uq_score,
            observation_uq=observation_uq_record,
            risk_decision=risk_decision,
            steer=float(control.steer),
            closedloop_safety=closedloop_safety,
            planning_response=planning_response_record,
        )
        if SAVE_PATH is not None and self.step % 10 == 0:
            self.save(tick_data)
        self.prev_control = control
        return control

    @staticmethod
    def _actor_planar_safety_state(actor, actor_snapshot, category):
        transform = actor_snapshot.get_transform()
        velocity = actor_snapshot.get_velocity()
        bounding_box = actor.bounding_box
        extent = bounding_box.extent
        box_center = carla.Location(
            x=bounding_box.location.x,
            y=bounding_box.location.y,
            z=bounding_box.location.z,
        )
        transform.transform(box_center)
        return {
            'actor_id': int(actor.id),
            'type_id': str(actor.type_id),
            'category': category,
            'position_xy': [
                float(box_center.x),
                float(box_center.y),
            ],
            'position_z': float(box_center.z),
            'actor_origin_xyz': [
                float(transform.location.x),
                float(transform.location.y),
                float(transform.location.z),
            ],
            'velocity_xy': [float(velocity.x), float(velocity.y)],
            'yaw_degrees': float(
                transform.rotation.yaw + bounding_box.rotation.yaw
            ),
            'extent_xy_m': [float(extent.x), float(extent.y)],
            'extent_z_m': float(extent.z),
            'radius_m': float(math.hypot(extent.x, extent.y)),
        }

    def _refresh_safety_actor_cache(self, world):
        actors = world.get_actors()
        self._safety_actor_cache = {
            int(actor.id): actor
            for actor in actors
            if actor.id != self.hero_actor.id
            and actor.is_alive
            and actor.is_active
            and not actor.is_dormant
            and (
                actor.type_id.startswith('vehicle.')
                or actor.type_id.startswith('walker.pedestrian.')
            )
        }
        self._safety_actor_cache_step = self.step

    def _query_path_junction_entry(self, conflict, closedloop_safety):
        """Query the first CARLA-map junction entry on the ORION base path."""

        if self._dynamic_yield_map is None:
            self._dynamic_yield_map = self.hero_actor.get_world().get_map()
        ego = closedloop_safety['ego']
        z = float(ego.get('position_z', 0.0))
        world_points = (
            tuple(ego['position_xy']),
        ) + tuple(conflict.base_plan_world_xy)

        def is_junction_at_xy(xy):
            waypoint = self._dynamic_yield_map.get_waypoint(
                carla.Location(x=float(xy[0]), y=float(xy[1]), z=z),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            return bool(waypoint is not None and waypoint.is_junction)

        return first_path_junction_entry(
            world_points,
            is_junction_at_xy,
            resolution_m=self.dynamic_yield_map_resolution_m,
        )

    def _collect_closedloop_safety_geometry(self):
        if not self.closedloop_safety_telemetry:
            return None
        try:
            world = self.hero_actor.get_world()
            if (
                self._safety_actor_cache_step is None
                or self.step - self._safety_actor_cache_step
                >= self.closedloop_safety_actor_refresh_steps
            ):
                self._refresh_safety_actor_cache(world)

            world_snapshot = world.get_snapshot()
            ego_snapshot = world_snapshot.find(self.hero_actor.id)
            if ego_snapshot is None:
                raise RuntimeError('ego actor missing from CARLA world snapshot')
            ego_state = self._actor_planar_safety_state(
                self.hero_actor, ego_snapshot, 'ego'
            )
            actor_states = []
            ego_x, ego_y = ego_state['position_xy']
            ego_z = ego_state['position_z']
            for actor_id, actor in list(self._safety_actor_cache.items()):
                actor_snapshot = world_snapshot.find(actor_id)
                if actor_snapshot is None:
                    continue
                category = (
                    'walker'
                    if actor.type_id.startswith('walker.pedestrian.')
                    else 'vehicle'
                )
                actor_state = self._actor_planar_safety_state(
                    actor, actor_snapshot, category
                )
                actor_x, actor_y = actor_state['position_xy']
                vertical_gap = vertical_separating_gap(
                    ego_z,
                    ego_state['extent_z_m'],
                    actor_state['position_z'],
                    actor_state['extent_z_m'],
                )
                if (
                    vertical_gap <= self.closedloop_safety_vertical_margin_m
                    and math.hypot(actor_x - ego_x, actor_y - ego_y)
                    <= self.closedloop_safety_range_m
                ):
                    actor_state['vertical_separating_gap_m'] = vertical_gap
                    actor_states.append(actor_state)

            summary = summarize_dynamic_actor_safety(
                ego_state,
                actor_states,
                horizon_seconds=self.closedloop_safety_horizon_seconds,
                max_actor_records=self.closedloop_safety_max_actor_records,
            )
            summary['world_frame'] = int(world_snapshot.frame)
            summary['world_elapsed_seconds'] = float(
                world_snapshot.timestamp.elapsed_seconds
            )
            summary['actor_cache_size'] = len(self._safety_actor_cache)
            summary['range_m'] = self.closedloop_safety_range_m
            summary['vertical_filter_margin_m'] = (
                self.closedloop_safety_vertical_margin_m
            )
            return summary
        except Exception as error:
            if not self._safety_telemetry_error_reported:
                print(
                    '[ClosedLoopSafety] telemetry unavailable: '
                    f'{type(error).__name__}: {error}',
                    flush=True,
                )
                self._safety_telemetry_error_reported = True
            return {
                'schema': CLOSEDLOOP_SAFETY_SCHEMA_VERSION,
                'available': False,
                'error_type': type(error).__name__,
            }

    def _append_control_trace(
        self,
        *,
        sim_time_seconds,
        route_progress,
        speed,
        corruption_active,
        corruption_schedule_mode,
        corruption_trigger_time_seconds,
        corruption_elapsed_seconds,
        corruption_metadata,
        oracle_corruption_relevant,
        oracle_event_active,
        oracle_event_schedule,
        oracle_event_trigger_time_seconds,
        oracle_event_elapsed_seconds,
        raw_uq_score,
        density_uq_score,
        observation_uq,
        risk_decision,
        steer,
        closedloop_safety,
        planning_response,
    ):
        if self.control_trace_path is None:
            return
        record = {
            'step': self.step,
            'sim_time_seconds': sim_time_seconds,
            'route_progress': route_progress,
            'speed': speed,
            'steer': steer,
            'corruption_active': corruption_active,
            'corruption_schedule_mode': corruption_schedule_mode,
            'corruption_trigger_time_seconds': corruption_trigger_time_seconds,
            'corruption_elapsed_seconds': corruption_elapsed_seconds,
            'corruption_metadata': corruption_metadata,
            'oracle_corruption_relevant': oracle_corruption_relevant,
            'oracle_event_active': oracle_event_active,
            'oracle_event_schedule': oracle_event_schedule,
            'oracle_event_trigger_time_seconds': oracle_event_trigger_time_seconds,
            'oracle_event_elapsed_seconds': oracle_event_elapsed_seconds,
            'raw_uq_score': raw_uq_score,
            'density_uq_score': density_uq_score,
            'observation_uq': observation_uq,
            'risk': risk_decision.to_dict(),
            'closedloop_safety': closedloop_safety,
            'planning_response': planning_response,
        }
        with self.control_trace_path.open('a') as outfile:
            outfile.write(json.dumps(record, sort_keys=True) + '\n')

    def save(self, tick_data):
        frame = self.step // 10
        cvt_c = lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # PIL save uses RGB image 
        Image.fromarray(cvt_c(tick_data['imgs']['CAM_FRONT'])).save(self.save_path / 'rgb_front' / ('%04d.png' % frame))
        Image.fromarray(cvt_c(tick_data['model_input_front'])).save(self.save_path / 'rgb_front_model_input' / ('%04d.png' % frame))
        Image.fromarray(cvt_c(tick_data['model_input_front_tensor'])).save(self.save_path / 'rgb_front_model_tensor' / ('%04d.png' % frame))
        Image.fromarray(cvt_c(tick_data['imgs']['CAM_FRONT_LEFT'])).save(self.save_path / 'rgb_front_left' / ('%04d.png' % frame))
        Image.fromarray(cvt_c(tick_data['imgs']['CAM_FRONT_RIGHT'])).save(self.save_path / 'rgb_front_right' / ('%04d.png' % frame))
        Image.fromarray(cvt_c(tick_data['imgs']['CAM_BACK'])).save(self.save_path / 'rgb_back' / ('%04d.png' % frame))
        Image.fromarray(cvt_c(tick_data['imgs']['CAM_BACK_LEFT'])).save(self.save_path / 'rgb_back_left' / ('%04d.png' % frame))
        Image.fromarray(cvt_c(tick_data['imgs']['CAM_BACK_RIGHT'])).save(self.save_path / 'rgb_back_right' / ('%04d.png' % frame))
        Image.fromarray(cvt_c(tick_data['bev'])).save(self.save_path / 'bev' / ('%04d.png' % frame))
        outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
        json.dump(self.pid_metadata, outfile, indent=4)
        outfile.close()

        # metric info
        outfile = open(self.save_path / 'metric_info.json', 'w')
        json.dump(self.metric_info, outfile, indent=4)
        outfile.close()

    def destroy(self):
        if getattr(self, 'stage2_artifact_writer', None) is not None:
            if self.stage2_artifact_writer.records:
                index_path = self.stage2_artifact_writer.finalize()
                print(
                    '[Stage2Capture] artifact_index=%s records=%d'
                    % (index_path, len(self.stage2_artifact_writer.records)),
                    flush=True,
                )
        if getattr(self, 'observation_uq_adapter', None) is not None:
            del self.observation_uq_adapter
        self.observation_uq_previous_features = None
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()

    def gps_to_location(self, gps):
        EARTH_RADIUS_EQUA = 6378137.0
        # gps content: numpy array: [lat, lon, alt]
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y])


def command2hot(command,max_dim=6):
    if command < 0:
        command = 4
    command -= 1
    cmd_one_hot = np.zeros(max_dim)
    cmd_one_hot[command] = 1
    return cmd_one_hot

def command2nohot(command,max_dim=6):
    if command < 0:
        command = 4
    command -= 1
    return command

def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix


custom_fp16 = dict(
                    map_head=False,
                    pts_bbox_head=False)
def custom_wrap_fp16_model(model):
    for m in model.modules():
        if hasattr(m, 'fp16_enabled'):
            m.fp16_enabled = True
    for module_name, v in custom_fp16.items():
        model._modules[module_name].fp16_enabled = v
