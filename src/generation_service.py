from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import aiohttp

from comfy_client import ComfyClient, ComfyClientError
from comfy_workflow import ComfyWorkflow
from image_service import ImageService
from models import AppConfig, JobData
from prompt_engine import PromptEngine

MaybeAsyncCallback = Callable[..., Any]


@dataclass
class GenerationCallbacks:
    on_status: MaybeAsyncCallback | None = None
    on_progress: MaybeAsyncCallback | None = None
    on_counts: MaybeAsyncCallback | None = None
    on_image_saved: MaybeAsyncCallback | None = None
    on_error: MaybeAsyncCallback | None = None
    on_batch_dir: MaybeAsyncCallback | None = None
    on_finished: MaybeAsyncCallback | None = None


class GenerationService:
    def __init__(self, prompt_engine: PromptEngine, callbacks: GenerationCallbacks):
        self.prompt_engine = prompt_engine
        self.callbacks = callbacks

    async def _call(self, cb: MaybeAsyncCallback | None, *args):
        if not cb:
            return
        result = cb(*args)
        if isinstance(result, Awaitable):
            await result

    async def process_jobs(
        self,
        config: AppConfig,
        jobs: list[JobData],
        current_session_batch_dir: str | None,
        workflow: ComfyWorkflow,
    ) -> str | None:
        client = ComfyClient(config.server_url, config.client_id)

        total_iterations = sum(max(1, int(job.count)) for job in jobs)
        generated_iterations = 0

        batch_dir = ImageService.create_or_reuse_session_batch_dir(
            config.out_dir,
            current_session_batch_dir,
        )
        await self._call(self.callbacks.on_batch_dir, str(batch_dir))
        image_counter = ImageService.next_png_counter(batch_dir)

        async with aiohttp.ClientSession() as session:
            for job_index, job in enumerate(jobs, start=1):
                count = max(1, int(job.count))

                for image_index in range(1, count + 1):
                    max_retries = 10
                    attempt = 0

                    while attempt <= max_retries:
                        await self._call(
                            self.callbacks.on_counts,
                            job_index, len(jobs),
                            image_index, count,
                            generated_iterations, total_iterations,
                        )
                        if attempt == 0:
                            await self._call(self.callbacks.on_status, "Queuing...")

                        resolved_main, resolved_neg = self.prompt_engine.resolve_job(job)

                        triggers = ", ".join(
                            l.trigger.strip() for l in job.loras if l.name and l.trigger.strip()
                        )
                        if triggers:
                            resolved_main = f"{triggers}, {resolved_main}" if resolved_main else triggers

                        wf = workflow.build(
                            prompt_node_id=config.prompt_node_id or None,
                            prompt_field=config.prompt_field or "text",
                            prompt_text=resolved_main,
                            negative_node_id=config.negative_node_id or None,
                            negative_field=config.negative_field or "text",
                            negative_text=resolved_neg,
                            lora_node_id=config.lora_node_id or None,
                            loras=[(l.name, l.strength) for l in job.loras if l.name],
                            randomize_seed=bool(config.randomize_seed),
                            size_node_id=config.size_node_id or None,
                            width=job.width,
                            height=job.height,
                        )

                        try:
                            prompt_id = await client.queue_prompt(session, wf)

                            async def status_cb(msg: str):
                                await self._call(self.callbacks.on_status, msg)

                            images = await client.wait_for_result(
                                session, prompt_id, on_status=status_cb,
                            )

                            if not images:
                                await self._call(
                                    self.callbacks.on_error,
                                    f"Job {job_index}, Image {image_index}: ComfyUI "
                                    "finished but produced no images. Make sure your "
                                    "workflow has a Save Image / Preview Image node.",
                                )
                                return str(batch_dir)

                            for img in images:
                                data = await client.fetch_image(
                                    session, img["filename"], img["subfolder"], img["type"],
                                )
                                filename = f"{image_counter:04d}.png"
                                filepath = batch_dir / filename

                                ImageService.save_generated_image(data, filepath)

                                await self._call(self.callbacks.on_image_saved, str(filepath))
                                image_counter += 1

                            generated_iterations += 1
                            pct = (generated_iterations / total_iterations) * 100.0
                            await self._call(self.callbacks.on_progress, pct)

                            break  # Success, exit retry loop

                        except (ComfyClientError, aiohttp.ClientError, asyncio.TimeoutError) as e:
                            err_str = str(e).lower()
                            is_network = (
                                isinstance(e, (aiohttp.ClientError, asyncio.TimeoutError))
                                or "timeout" in err_str
                                or "connection" in err_str
                                or "disconnect" in err_str
                            )

                            if is_network and attempt < max_retries:
                                attempt += 1
                                await self._call(
                                    self.callbacks.on_status,
                                    f"Network/Timeout error. Retrying {attempt}/{max_retries} in 5s..."
                                )
                                await asyncio.sleep(5)
                                continue
                            else:
                                await self._call(
                                    self.callbacks.on_error,
                                    f"Generation failed on Job {job_index}, Image {image_index}.\n\n{e}",
                                )
                                return str(batch_dir)

                    await asyncio.sleep(0)

        await self._call(self.callbacks.on_finished)
        return str(batch_dir)
