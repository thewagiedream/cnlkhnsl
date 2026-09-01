from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from models import AppConfig, PresetsBundle


class StorageService:
    def __init__(self):
        self.app_dir = self._detect_app_dir()
        self.app_dir.mkdir(parents=True, exist_ok=True)

        self.internal_output_dir = self.app_dir / "Output"
        self.internal_output_dir.mkdir(parents=True, exist_ok=True)

        self.config_file = self.app_dir / "config.json"
        self.presets_file = self.app_dir / "presets.json"
        self.chunks_file = self.app_dir / "chunks.json"
        self.draft_file = self.app_dir / "draft.json"
        self.workflow_file = self.app_dir / "workflow.json"
        self.lora_triggers_file = self.app_dir / "lora_triggers.json"

    @staticmethod
    def is_android() -> bool:
        return (
            "ANDROID_ARGUMENT" in os.environ
            or "ANDROID_BOOTLOGO" in os.environ
            or "ANDROID_ROOT" in os.environ
            or sys.platform.lower().startswith("android")
        )

    def _detect_app_dir(self) -> Path:
        """
        Android behavior:

        First try public user-accessible folder:

            /storage/emulated/0/Download/ComfyCompanion
            /sdcard/Download/ComfyCompanion

        This lets you copy config/workflow files from PC normally. If
        Android blocks public storage, fall back to app-private storage.
        """

        candidates: list[Path] = []

        if self.is_android():
            candidates.extend(
                [
                    Path("/storage/emulated/0/Download/ComfyCompanion"),
                    Path("/sdcard/Download/ComfyCompanion"),
                ]
            )

            try:
                current_file = Path(__file__).resolve()
                for parent in current_file.parents:
                    if parent.name == "files":
                        candidates.append(parent / "ComfyCompanion_Flet")
                        break
            except Exception:
                pass

            for env_name in ["FLET_APP_STORAGE_DATA", "ANDROID_PRIVATE", "HOME"]:
                value = os.environ.get(env_name)
                if not value:
                    continue
                try:
                    p = Path(value).resolve()
                except Exception:
                    continue
                if str(p) == "/data":
                    continue
                candidates.append(p / "ComfyCompanion_Flet")

            candidates.append(Path(tempfile.gettempdir()) / "ComfyCompanion_Flet")

        else:
            if getattr(sys, "frozen", False):
                candidates.append(Path(sys.executable).resolve().parent)
            else:
                candidates.append(Path(__file__).resolve().parent)

            candidates.append(Path.home() / "ComfyCompanion_Flet")
            candidates.append(Path(tempfile.gettempdir()) / "ComfyCompanion_Flet")

        for candidate in candidates:
            if self._is_writable_dir(candidate):
                return candidate

        raise RuntimeError(
            "Could not find a writable application data directory.\n\n"
            f"Tried:\n" + "\n".join(str(c) for c in candidates)
        )

    def _is_writable_dir(self, path: Path) -> bool:
        try:
            path.mkdir(parents=True, exist_ok=True)
            test_file = path / ".write_test"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink(missing_ok=True)
            return True
        except Exception:
            return False

    def _looks_like_windows_path(self, value: str) -> bool:
        if not value:
            return False
        value = str(value)
        return "\\" in value or ":" in value

    def get_default_output_dir(self) -> Path:
        out = self.app_dir / "Output"
        try:
            out.mkdir(parents=True, exist_ok=True)
        except Exception:
            return self.internal_output_dir
        return out

    def load_json(self, path: Path, default: Any) -> Any:
        """
        Main expected files: config.json, presets.json, chunks.json,
        workflow.json. Convenience fallback: same names with a .txt
        extension (still must contain valid JSON).
        """
        candidates = [path]

        if path.suffix.lower() == ".json":
            candidates.append(path.with_suffix(".txt"))

        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                with candidate.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                continue

        return default

    def save_json(self, path: Path, data: Any):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def load_config(self) -> AppConfig:
        data = self.load_json(self.config_file, {})
        config = AppConfig.from_dict(data)

        if self.is_android():
            if (
                not config.out_dir
                or self._looks_like_windows_path(config.out_dir)
                or not self._is_writable_dir(Path(config.out_dir))
            ):
                config.out_dir = str(self.get_default_output_dir())
        else:
            if not config.out_dir:
                config.out_dir = str(self.get_default_output_dir())

        return config

    def save_config(self, config: AppConfig):
        self.save_json(self.config_file, config.to_dict())

    def load_presets_bundle(self) -> PresetsBundle:
        data = self.load_json(self.presets_file, {})
        return PresetsBundle.from_dict(data)

    def save_presets_bundle(self, bundle: PresetsBundle):
        self.save_json(self.presets_file, bundle.to_dict())

    def load_chunks(self) -> dict[str, str]:
        data = self.load_json(self.chunks_file, {})
        return {str(k): str(v) for k, v in data.items()}

    def save_chunks(self, chunks: dict[str, str]):
        self.save_json(self.chunks_file, chunks)

    def load_lora_triggers(self) -> dict[str, str]:
        """Per-LoRA-filename default trigger words, remembered across
        sessions so you don't have to retype them every time you pick
        that LoRA again."""
        data = self.load_json(self.lora_triggers_file, {})
        return {str(k): str(v) for k, v in data.items()}

    def save_lora_triggers(self, triggers: dict[str, str]):
        self.save_json(self.lora_triggers_file, triggers)

    def load_draft(self) -> dict[str, Any]:
        return self.load_json(self.draft_file, {})

    def save_draft(self, draft: dict[str, Any]):
        try:
            self.save_json(self.draft_file, draft)
        except Exception:
            pass

    # -- Imported ComfyUI workflow -----------------------------------

    def load_workflow_raw(self) -> dict[str, Any] | None:
        data = self.load_json(self.workflow_file, None)
        return data if isinstance(data, dict) else None

    def save_workflow_raw(self, raw: dict[str, Any]):
        self.save_json(self.workflow_file, raw)

    def import_workflow_file(self, source_path: str) -> dict[str, Any]:
        """Reads a workflow.json the user picked and persists a copy into
        the app's own storage so it survives restarts."""
        src = Path(source_path)
        with src.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        self.save_workflow_raw(raw)
        return raw
