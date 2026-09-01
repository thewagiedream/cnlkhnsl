from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class AppConfig:
    server_url: str = "http://127.0.0.1:8188"
    client_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    out_dir: str = ""

    # Imported workflow + which nodes get poked before each generation.
    workflow_path: str = ""
    prompt_node_id: str = ""
    prompt_field: str = "text"
    negative_node_id: str = ""
    negative_field: str = "text"
    lora_node_id: str = ""
    size_node_id: str = ""

    randomize_seed: bool = True
    jump_to_feed_on_start: bool = True
    auto_show_latest: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        base = cls()
        for k, v in data.items():
            if hasattr(base, k):
                setattr(base, k, v)
        return base

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PromptParts:
    char_amount: str = ""
    artists: str = ""
    details: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PromptParts":
        if not data:
            return cls()
        return cls(
            char_amount=str(data.get("char_amount", "")),
            artists=str(data.get("artists", "")),
            details=str(data.get("details", "")),
        )

    def combined(self) -> str:
        parts = [self.char_amount.strip(), self.artists.strip(), self.details.strip()]
        return ", ".join([p for p in parts if p])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoraSlotData:
    name: str = ""
    strength: float = 1.0
    trigger: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoraSlotData":
        return cls(
            name=str(data.get("name", "")),
            strength=float(data.get("strength", 1.0)),
            trigger=str(data.get("trigger", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JobData:
    prompt_parts: PromptParts = field(default_factory=PromptParts)
    prompt: str = ""
    negative_prompt: str = ""
    count: int = 1
    loras: list[LoraSlotData] = field(default_factory=list)
    width: int = 832
    height: int = 1216

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobData":
        parts = PromptParts.from_dict(data.get("prompt_parts"))

        raw_prompt = str(data.get("prompt", ""))
        if not raw_prompt:
            raw_prompt = parts.combined()

        return cls(
            prompt_parts=parts,
            prompt=raw_prompt,
            negative_prompt=str(data.get("negative_prompt", "")),
            count=int(data.get("count", 1)),
            loras=[LoraSlotData.from_dict(l) for l in data.get("loras", [])],
            width=int(data.get("width", 832)),
            height=int(data.get("height", 1216)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_parts": self.prompt_parts.to_dict(),
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "count": self.count,
            "loras": [l.to_dict() for l in self.loras],
            "width": self.width,
            "height": self.height,
        }


@dataclass
class PresetsBundle:
    presets: dict[str, dict[str, Any]] = field(default_factory=dict)
    queues: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PresetsBundle":
        if "presets" in data or "queues" in data:
            return cls(
                presets=dict(data.get("presets", {})),
                queues=dict(data.get("queues", {})),
            )
        # Backward compatibility with older presets.json formats.
        return cls(presets=dict(data), queues={})

    def to_dict(self) -> dict[str, Any]:
        return {
            "presets": self.presets,
            "queues": self.queues,
        }
