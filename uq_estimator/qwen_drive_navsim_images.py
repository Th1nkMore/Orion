"""Deterministic image resolver for paired Qwen-Drive NAVSIM runs.

The released Qwen-Drive planning runner accepts an ``image_resolver`` factory.
This module keeps the official scene file and preprocessing path unchanged and
changes only explicitly selected camera pixels.  The environment-only factory
also makes it usable from the unmodified upstream command-line runner.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Optional, Union

from PIL import Image


SUPPORTED_CORRUPTIONS = {"none", "camera_dropout"}
DEFAULT_CAMERA = "CAM_F0"


def _split_cameras(value: str) -> frozenset[str]:
    cameras = frozenset(item.strip().upper() for item in value.split(",") if item.strip())
    if not cameras:
        raise ValueError("ORION_NAVSIM_CORRUPTION_CAMERAS must not be empty")
    return cameras


def _load_path_manifest(path: Optional[Path]) -> Optional[frozenset[str]]:
    if path is None:
        return None
    entries = frozenset(
        line.strip().replace("\\", "/")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not entries:
        raise ValueError("NAVSIM corruption path manifest is empty")
    return entries


@dataclass(frozen=True)
class NavsimImageResolver:
    """Resolve clean images lazily and apply a deterministic camera dropout."""

    image_root: Path
    corruption: str = "none"
    cameras: frozenset[str] = frozenset({DEFAULT_CAMERA})
    selected_paths: Optional[frozenset[str]] = None

    def __post_init__(self) -> None:
        root = self.image_root.expanduser().resolve()
        object.__setattr__(self, "image_root", root)
        if not root.is_dir():
            raise FileNotFoundError(f"NAVSIM image root does not exist: {root}")
        if self.corruption not in SUPPORTED_CORRUPTIONS:
            raise ValueError(
                f"unsupported NAVSIM corruption {self.corruption!r}; "
                f"expected one of {sorted(SUPPORTED_CORRUPTIONS)}"
            )
        if not self.cameras:
            raise ValueError("at least one NAVSIM corruption camera is required")

    @classmethod
    def from_environment(cls) -> "NavsimImageResolver":
        root = os.environ.get("ORION_NAVSIM_IMAGE_ROOT", "").strip()
        if not root:
            raise ValueError("ORION_NAVSIM_IMAGE_ROOT is required")
        corruption = os.environ.get("ORION_NAVSIM_CORRUPTION", "none").strip().lower()
        cameras = _split_cameras(
            os.environ.get("ORION_NAVSIM_CORRUPTION_CAMERAS", DEFAULT_CAMERA)
        )
        manifest_value = os.environ.get("ORION_NAVSIM_CORRUPTION_PATHS", "").strip()
        manifest = Path(manifest_value).expanduser().resolve() if manifest_value else None
        return cls(
            image_root=Path(root),
            corruption=corruption,
            cameras=cameras,
            selected_paths=_load_path_manifest(manifest),
        )

    @staticmethod
    def _normalize_relative_path(path: str) -> str:
        normalized = str(PurePosixPath(path.replace("\\", "/")))
        if normalized == "." or normalized.startswith("../") or normalized.startswith("/"):
            raise ValueError(f"NAVSIM frame path must be relative: {path!r}")
        return normalized

    def _full_path(self, relative_path: str) -> Path:
        normalized = self._normalize_relative_path(relative_path)
        full_path = (self.image_root / normalized).resolve()
        if not full_path.is_relative_to(self.image_root):
            raise ValueError(f"NAVSIM frame escapes image root: {relative_path!r}")
        if not full_path.is_file():
            raise FileNotFoundError(f"NAVSIM frame is missing: {full_path}")
        return full_path

    def _is_selected(self, relative_path: str) -> bool:
        if self.corruption == "none":
            return False
        normalized = self._normalize_relative_path(relative_path)
        if self.selected_paths is not None and normalized not in self.selected_paths:
            return False
        path_cameras = {part.upper() for part in PurePosixPath(normalized).parts}
        return bool(path_cameras & self.cameras)

    def reference(self, relative_path: str) -> Union[Path, Callable[[], Image.Image]]:
        """Return a path for clean reads or a lazy callable for changed pixels."""

        full_path = self._full_path(relative_path)
        if not self._is_selected(relative_path):
            return full_path
        return lambda: self(relative_path)

    def __call__(self, relative_path: str) -> Image.Image:
        full_path = self._full_path(relative_path)
        with Image.open(full_path) as source:
            clean = source.convert("RGB")
            clean.load()
        if not self._is_selected(relative_path):
            return clean
        if self.corruption == "camera_dropout":
            return Image.new("RGB", clean.size, color=(0, 0, 0))
        raise AssertionError(f"unreachable corruption: {self.corruption}")


def make_reader() -> NavsimImageResolver:
    """Factory consumed by Qwen's ``--image-resolver module:factory`` flag."""

    return NavsimImageResolver.from_environment()
