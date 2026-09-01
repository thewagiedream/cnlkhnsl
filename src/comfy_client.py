from __future__ import annotations

import asyncio
import uuid
from typing import Any, Awaitable, Callable

import aiohttp


class ComfyClientError(Exception):
    pass


class ComfyClient:
    """
    Minimal client for a local/LAN ComfyUI server's default HTTP API.
    Uses polling (GET /queue, GET /history) instead of the websocket API,
    which is simpler and more resilient on flaky mobile connections.
    """

    def __init__(self, server_url: str, client_id: str | None = None):
        self.server_url = (server_url or "http://127.0.0.1:8188").rstrip("/")
        self.client_id = client_id or str(uuid.uuid4())

    def _url(self, path: str) -> str:
        return f"{self.server_url}{path}"

    async def get_object_info(self, session: aiohttp.ClientSession) -> dict[str, Any]:
        async with session.get(
            self._url("/object_info"), timeout=aiohttp.ClientTimeout(total=20)
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def get_available_loras(self, session: aiohttp.ClientSession) -> list[str]:
        """Reads the lora combo list straight from the server's node schema,
        so it always matches whatever's actually in your loras folder."""
        info = await self.get_object_info(session)

        for node_type, spec in info.items():
            if "lora loader stack" in node_type.lower() or "power lora loader" in node_type.lower():
                required = spec.get("input", {}).get("required", {})
                combo = required.get("lora_01")
                if combo and isinstance(combo, list) and combo and isinstance(combo[0], list):
                    return list(combo[0])

        for node_type, spec in info.items():
            if node_type.lower() in ("loraloader", "lora loader"):
                required = spec.get("input", {}).get("required", {})
                combo = required.get("lora_name")
                if combo and isinstance(combo, list) and combo and isinstance(combo[0], list):
                    return list(combo[0])

        return []

    async def queue_prompt(
        self, session: aiohttp.ClientSession, workflow: dict[str, Any]
    ) -> str:
        payload = {"prompt": workflow, "client_id": self.client_id}
        async with session.post(
            self._url("/prompt"), json=payload, timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise ComfyClientError(
                    f"ComfyUI rejected the prompt (HTTP {resp.status}): {text[:800]}"
                )
            data = await resp.json()
            prompt_id = data.get("prompt_id")
            if not prompt_id:
                raise ComfyClientError(f"ComfyUI didn't return a prompt_id: {data}")
            return prompt_id

    async def get_queue_position(
        self, session: aiohttp.ClientSession, prompt_id: str
    ) -> int | None:
        async with session.get(
            self._url("/queue"), timeout=aiohttp.ClientTimeout(total=15)
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

        pending = data.get("queue_pending", [])
        for i, item in enumerate(pending):
            if len(item) > 1 and item[1] == prompt_id:
                return i + 1
        return None

    async def get_history(
        self, session: aiohttp.ClientSession, prompt_id: str
    ) -> dict[str, Any] | None:
        async with session.get(
            self._url(f"/history/{prompt_id}"), timeout=aiohttp.ClientTimeout(total=20)
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
        return data.get(prompt_id)

    async def fetch_image(
        self,
        session: aiohttp.ClientSession,
        filename: str,
        subfolder: str,
        folder_type: str,
    ) -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        async with session.get(
            self._url("/view"), params=params, timeout=aiohttp.ClientTimeout(total=60)
        ) as resp:
            resp.raise_for_status()
            return await resp.read()

    async def wait_for_result(
        self,
        session: aiohttp.ClientSession,
        prompt_id: str,
        on_status: Callable[[str], Awaitable[None] | None] | None = None,
        poll_interval: float = 1.0,
        timeout: float = 1800.0,
    ) -> list[dict[str, str]]:
        """Polls /history until the prompt finishes, returning output images
        as [{"filename", "subfolder", "type"}, ...]."""
        elapsed = 0.0
        reported_generating = False

        while elapsed < timeout:
            history = await self.get_history(session, prompt_id)

            if history is not None:
                status = history.get("status", {})
                if status.get("status_str") == "error":
                    raise ComfyClientError(
                        f"ComfyUI reported an error: {status.get('messages', [])}"
                    )

                outputs = history.get("outputs", {})
                images: list[dict[str, str]] = []
                for node_output in outputs.values():
                    for img in node_output.get("images", []):
                        images.append({
                            "filename": img.get("filename", ""),
                            "subfolder": img.get("subfolder", ""),
                            "type": img.get("type", "output"),
                        })

                if images:
                    return images
                if status.get("completed") is True:
                    return []

            if not reported_generating:
                position = await self.get_queue_position(session, prompt_id)
                if position is not None and position > 0:
                    if on_status:
                        result = on_status(f"Queued (position {position})...")
                        if result is not None:
                            await result
                else:
                    reported_generating = True
                    if on_status:
                        result = on_status("Generating...")
                        if result is not None:
                            await result

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise ComfyClientError("Timed out waiting for ComfyUI to finish the prompt.")
