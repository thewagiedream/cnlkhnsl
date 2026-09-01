from __future__ import annotations

import copy
import random
import re
from typing import Any

# rgthree's lora-stacking nodes use human-readable class_types (with spaces),
# which is unusual for ComfyUI but is how rgthree registers them.
LORA_STACK_HINTS = ("lora loader stack", "power lora loader")

# Widget keys we recognize immediately as "this is a prompt box" - checked
# first, in this order, before falling back to a generic scan.
TEXT_WIDGET_KEYS = ("text", "text_g", "text_l", "string", "value", "prompt", "Text")

# When a node's text field isn't one of the names above (every custom node
# pack names theirs differently), we fall back to treating *any* plain
# string widget as a candidate - except these, which are almost always
# combo/selector values (checkpoint names, sampler names, etc.) rather
# than prompt text, so listing them would just add noise.
_NON_TEXT_KEYS = {
    "sampler_name", "scheduler", "ckpt_name", "vae_name", "lora_name",
    "control_net_name", "upscale_method", "resize_method", "weight_dtype",
    "filename_prefix", "clip_name", "clip_name1", "clip_name2",
    "style_preset", "device", "unet_name", "style", "font", "font_name",
}
_NON_TEXT_SUFFIXES = ("_name", "_method", "_dtype")


class WorkflowError(Exception):
    pass


class ComfyWorkflow:
    """
    Thin wrapper around a ComfyUI API-format workflow (the JSON exported via
    ComfyUI's "Save (API Format)" with Dev Mode enabled).

    This never changes the *structure* of the workflow -- it only pokes new
    values into the `inputs` dict of specific nodes before each queued
    prompt. Whatever you built in ComfyUI (samplers, resolution, steps,
    CFG, etc.) is left completely untouched.
    """

    def __init__(self, raw: dict[str, Any]):
        if not isinstance(raw, dict) or not raw:
            raise WorkflowError("Workflow file is empty or not a JSON object.")

        # Detect the common "wrong export" mistake: UI-format workflow.json
        # exports have top-level "nodes"/"links" arrays instead of a flat
        # node-id -> node dict.
        if "nodes" in raw and "links" in raw:
            raise WorkflowError(
                "This looks like a UI-format workflow export, not API format.\n\n"
                "In ComfyUI: Settings > Enable Dev Mode Options, then use the "
                "'Save (API Format)' button and import that file instead."
            )

        for node_id, node in raw.items():
            if not isinstance(node, dict) or "class_type" not in node:
                raise WorkflowError(
                    f"Node '{node_id}' doesn't look like API-format ComfyUI JSON "
                    "(missing 'class_type')."
                )

        self.raw = raw

    # ------------------------------------------------------------------
    # Introspection - used to populate the Settings dropdowns after import
    # ------------------------------------------------------------------

    def text_nodes(self) -> list[dict[str, Any]]:
        """
        Nodes with a plain string text widget - CLIPTextEncode and any
        custom "text box" node (WAS, ComfyUI-Custom-Scripts, etc). We check
        common widget key names first; if a node doesn't use any of those,
        we fall back to its first plain-string input that isn't an obvious
        combo/selector value, so unusual custom nodes still show up.

        A node's field only shows up here if it's a literal string in the
        JSON. If that field is wired from another node instead of typed in
        directly, ComfyUI serializes it as a link (a 2-item list), not a
        string, so it won't be picked up - point the prompt node setting at
        whichever node actually holds the literal text.
        """
        results = []
        for node_id, node in self.raw.items():
            class_type = str(node.get("class_type", ""))
            if any(hint in class_type.lower() for hint in LORA_STACK_HINTS):
                continue  # handled by lora_stack_nodes(), not a prompt box

            inputs = node.get("inputs", {})
            title = node.get("_meta", {}).get("title") or node["class_type"]

            matched_key = None
            for key in TEXT_WIDGET_KEYS:
                if isinstance(inputs.get(key), str):
                    matched_key = key
                    break

            if matched_key is None:
                for key, value in inputs.items():
                    if not isinstance(value, str):
                        continue
                    if key in _NON_TEXT_KEYS or key.endswith(_NON_TEXT_SUFFIXES):
                        continue
                    matched_key = key
                    break

            if matched_key is None:
                continue

            value = inputs[matched_key]
            preview = value.strip().replace("\n", " ")
            if len(preview) > 60:
                preview = preview[:57] + "..."
            results.append({
                "id": node_id,
                "title": title,
                "class_type": node["class_type"],
                "field": matched_key,
                "preview": preview,
            })
        return results

    def lora_stack_nodes(self) -> list[dict[str, Any]]:
        """rgthree-style lora stack / power-loader nodes."""
        results = []
        for node_id, node in self.raw.items():
            class_type = str(node.get("class_type", ""))
            if any(hint in class_type.lower() for hint in LORA_STACK_HINTS):
                title = node.get("_meta", {}).get("title") or class_type
                results.append({
                    "id": node_id,
                    "title": title,
                    "class_type": class_type,
                    "slots": self._lora_slot_count(node),
                })
        return results

    @staticmethod
    def _lora_slot_count(node: dict[str, Any]) -> int:
        inputs = node.get("inputs", {})
        indices = set()
        for key in inputs:
            m = re.match(r"lora_(\d+)$", key)
            if m:
                indices.add(int(m.group(1)))
        return len(indices)

    def seed_widgets(self) -> list[tuple[str, str]]:
        """[(node_id, field_name), ...] for every int seed-ish widget."""
        results = []
        for node_id, node in self.raw.items():
            inputs = node.get("inputs", {})
            for key in ("seed", "noise_seed"):
                if isinstance(inputs.get(key), int):
                    results.append((node_id, key))
        return results

    def latent_size_nodes(self) -> list[dict[str, Any]]:
        """Nodes with plain integer width & height widgets - EmptyLatentImage
        and custom "resolution preset" nodes alike."""
        results = []
        for node_id, node in self.raw.items():
            inputs = node.get("inputs", {})
            if isinstance(inputs.get("width"), int) and isinstance(inputs.get("height"), int):
                title = node.get("_meta", {}).get("title") or node["class_type"]
                results.append({
                    "id": node_id,
                    "title": title,
                    "class_type": node["class_type"],
                    "width": inputs["width"],
                    "height": inputs["height"],
                })
        return results

    # ------------------------------------------------------------------
    # Mutation - always operates on a fresh deep copy of the template
    # ------------------------------------------------------------------

    def build(
        self,
        *,
        prompt_node_id: str | None,
        prompt_field: str,
        prompt_text: str,
        negative_node_id: str | None,
        negative_field: str,
        negative_text: str,
        lora_node_id: str | None,
        loras: list[tuple[str, float]],
        randomize_seed: bool,
        size_node_id: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, Any]:
        wf = copy.deepcopy(self.raw)

        if prompt_node_id and prompt_node_id in wf:
            wf[prompt_node_id].setdefault("inputs", {})[prompt_field] = prompt_text

        if negative_node_id and negative_node_id in wf and negative_text:
            wf[negative_node_id].setdefault("inputs", {})[negative_field] = negative_text

        if lora_node_id and lora_node_id in wf:
            self._apply_loras(wf[lora_node_id], loras)

        if size_node_id and size_node_id in wf and width and height:
            wf[size_node_id]["inputs"]["width"] = int(width)
            wf[size_node_id]["inputs"]["height"] = int(height)

        if randomize_seed:
            for node_id, field in self.seed_widgets():
                if node_id in wf:
                    wf[node_id]["inputs"][field] = random.randint(0, 2 ** 32 - 1)

        return wf

    @staticmethod
    def _apply_loras(node: dict[str, Any], loras: list[tuple[str, float]]):
        """
        Fills the lora_XX / strength_XX (and strength_model_XX /
        strength_clip_XX variants) widgets already present on an rgthree
        lora-stack node. Never invents new keys - only ever touches keys
        that already exist on the node, so it adapts to whatever version
        of the rgthree node you have.
        """
        inputs = node.setdefault("inputs", {})
        slot_count = ComfyWorkflow._lora_slot_count(node)
        if slot_count == 0:
            return

        for i in range(1, slot_count + 1):
            idx = i - 1
            name, strength = (loras[idx] if idx < len(loras) else (None, 0.0))
            active = bool(name)

            for key in (f"lora_{i:02d}", f"lora_{i}"):
                if key in inputs:
                    inputs[key] = name if active else "None"
                    break

            for prefix in ("strength", "strength_model", "strength_clip"):
                for key in (f"{prefix}_{i:02d}", f"{prefix}_{i}"):
                    if key in inputs:
                        inputs[key] = float(strength) if active else 0.0
                        break
