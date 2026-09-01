from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
import flet as ft

from comfy_client import ComfyClient
from comfy_workflow import ComfyWorkflow, WorkflowError
from generation_service import GenerationCallbacks, GenerationService
from image_service import ImageService
from models import AppConfig, JobData, LoraSlotData, PresetsBundle
from prompt_engine import PromptEngine
from storage_service import StorageService

# Palette
DARK_BG = "#0E0E13"
DARK_SURFACE = "#1F1F26"
CARD_BG = "#181820"
ACCENT = "#FF9C66"
ACCENT_LIGHT = "#FFB166"
ACCENT_BUSY = "#8C5934"
TEXT_MAIN = "#ECECF0"
TEXT_DIM = "#9A9AA5"


class ComfyCompanionApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.storage = StorageService()

        self.config: AppConfig = self.storage.load_config()
        self.bundle: PresetsBundle = self.storage.load_presets_bundle()
        self.chunks: dict[str, str] = self.storage.load_chunks()
        self.draft: dict[str, Any] = self.storage.load_draft()

        self.workflow_raw: dict[str, Any] | None = self.storage.load_workflow_raw()
        self.workflow: ComfyWorkflow | None = None
        self.workflow_error: str | None = None
        self._rebuild_workflow()

        self.available_loras: list[str] = list(self.draft.get("available_loras", []))
        self.lora_triggers: dict[str, str] = self.storage.load_lora_triggers()

        self.jobs: list[JobData] = [
            JobData.from_dict(j) for j in self.draft.get("jobs", [])
        ]
        self.current_loras: list[LoraSlotData] = [
            LoraSlotData.from_dict(l) for l in self.draft.get("current_loras", [])
        ]

        self.session_generated_images: list[str] = []
        self.current_preview_index: int = 0
        self.session_batch_dir: str | None = None
        self.is_running: bool = False
        self._show_status_bar: bool = True

        self.preview_src_cache: dict[str, str] = {}
        self.thumbnail_src_cache: dict[str, str] = {}

        # In current Flet, FilePicker is a "service" control (not added to
        # page.overlay) whose pick methods are awaited directly for a
        # result, instead of firing an on_result event.
        self.output_dir_picker = ft.FilePicker()
        self.workflow_file_picker = ft.FilePicker()

    async def initialize(self):
        self.page.services.extend([self.output_dir_picker, self.workflow_file_picker])
        self.build_ui()
        self.refresh_all_dynamic_views()
        self.page.update()
        # Belt-and-suspenders: make sure the Generate buttons are actually
        # painted with their label on first mount (see set_generate_button_busy).
        self._safe_update(self.generate_button, self.images_generate_button)

    @staticmethod
    def _ev(control: ft.Control, **handlers) -> ft.Control:
        """Attach event handlers after construction. Flet's generated
        __init__ signatures don't reliably accept every handler kwarg
        (varies by version/control), so handlers are always set as plain
        attributes instead of passed into the constructor."""
        for name, handler in handlers.items():
            setattr(control, name, handler)
        return control

    # -------------------------------------------------------------------
    # Small UI helpers
    # -------------------------------------------------------------------

    def _safe_update(self, *controls: ft.Control):
        for c in controls:
            try:
                c.update()
            except Exception:
                try:
                    self.page.update()
                except Exception:
                    pass

    def section(self, title: str, content: list[ft.Control]) -> ft.Card:
        return ft.Card(
            elevation=1,
            content=ft.Container(
                padding=16,
                content=ft.Column(
                    [
                        ft.Text(title, size=16, weight=ft.FontWeight.BOLD),
                        ft.Divider(height=8, color=ft.Colors.TRANSPARENT),
                        *content,
                    ],
                    spacing=8,
                ),
            ),
        )

    def show_snack(self, message: str, color: str | None = None):
        sb = ft.SnackBar(content=ft.Text(message), bgcolor=color)
        try:
            self.page.show_dialog(sb)
        except Exception:
            try:
                self.page.snack_bar = sb
                self.page.snack_bar.open = True
                self.page.update()
            except Exception:
                pass

    def show_error(self, message: str):
        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text("Error"),
            content=ft.Text(message),
        )
        dlg.actions = [self._ev(ft.TextButton("OK"), on_click=lambda e: self.page.pop_dialog())]
        self.page.show_dialog(dlg)

    def show_info(self, title: str, message: str):
        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text(title),
            content=ft.Text(message),
        )
        dlg.actions = [self._ev(ft.TextButton("OK"), on_click=lambda e: self.page.pop_dialog())]
        self.page.show_dialog(dlg)

    def prompt_dialog(self, title: str, label: str, on_submit, initial_value: str = ""):
        field = ft.TextField(label=label, value=initial_value, autofocus=True)
        dlg = ft.AlertDialog(modal=True, title=ft.Text(title), content=field)

        def submit(_):
            value = field.value.strip()
            self.page.pop_dialog()
            if value:
                on_submit(value)

        dlg.actions = [
            self._ev(ft.TextButton("Cancel"), on_click=lambda e: self.page.pop_dialog()),
            self._ev(ft.Button("OK"), on_click=submit),
        ]
        self.page.show_dialog(dlg)

    def two_field_dialog(
        self, title: str, label_a: str, label_b: str, on_submit,
        initial_a: str = "", initial_b: str = "",
    ):
        field_a = ft.TextField(label=label_a, value=initial_a, autofocus=True)
        field_b = ft.TextField(label=label_b, value=initial_b, multiline=True, min_lines=2)
        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text(title),
            content=ft.Column([field_a, field_b], tight=True, spacing=10),
        )

        def submit(_):
            a, b = field_a.value.strip(), field_b.value.strip()
            self.page.pop_dialog()
            if a:
                on_submit(a, b)

        dlg.actions = [
            ft.TextButton("Cancel", on_click=lambda e: self.page.pop_dialog()),
            ft.Button("OK", on_click=submit),
        ]
        self.page.show_dialog(dlg)

    def confirm_dialog(self, title: str, message: str, on_yes):
        dlg = ft.AlertDialog(modal=True, title=ft.Text(title), content=ft.Text(message))

        def yes(_):
            self.page.pop_dialog()
            on_yes()

        dlg.actions = [
            ft.TextButton("No", on_click=lambda e: self.page.pop_dialog()),
            ft.Button("Yes", on_click=yes),
        ]
        self.page.show_dialog(dlg)

    def safe_int(self, value: str, default: int) -> int:
        try:
            return int(str(value).strip())
        except Exception:
            return default

    def safe_float(self, value: str, default: float) -> float:
        try:
            return float(str(value).strip())
        except Exception:
            return default

    # -------------------------------------------------------------------
    # Workflow
    # -------------------------------------------------------------------

    def _rebuild_workflow(self):
        self.workflow = None
        self.workflow_error = None
        if not self.workflow_raw:
            return
        try:
            self.workflow = ComfyWorkflow(self.workflow_raw)
        except WorkflowError as e:
            self.workflow_error = str(e)

    def get_ready_workflow(self) -> ComfyWorkflow | None:
        if self.workflow is None:
            self.show_error(
                self.workflow_error
                or "Import a ComfyUI workflow (API format) in Settings first."
            )
            return None
        if not self.config.prompt_node_id:
            self.show_error("Pick which node the main prompt goes into, in Settings.")
            return None
        return self.workflow

    # -------------------------------------------------------------------
    # Main UI shell
    # -------------------------------------------------------------------

    def build_ui(self):
        self.build_settings_view()
        self.build_prompt_view()
        self.build_queue_view()
        self.build_library_view()
        self.build_images_view()

        self.screen_title = ft.Text("Prompt", size=18, weight=ft.FontWeight.BOLD, color=TEXT_MAIN)

        self.status_left = ft.Text("", size=12, color=TEXT_DIM)
        self.status_right = ft.Text("", size=12, color=TEXT_DIM)
        self.bottom_status = ft.Container(
            padding=ft.Padding.symmetric(horizontal=16, vertical=8),
            bgcolor=DARK_SURFACE,
            visible=False,
            content=ft.Row(
                [self.status_left, self.status_right],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
        )

        self.content_area = ft.Container(expand=True, content=self.prompt_view, padding=16)

        self.nav_bar = ft.NavigationBar(
            selected_index=0,
            destinations=[
                ft.NavigationBarDestination(icon=ft.Icons.EDIT_NOTE, label="Prompt"),
                ft.NavigationBarDestination(icon=ft.Icons.QUEUE, label="Queue"),
                ft.NavigationBarDestination(icon=ft.Icons.IMAGE, label="Images"),
                ft.NavigationBarDestination(icon=ft.Icons.LIBRARY_BOOKS, label="Library"),
                ft.NavigationBarDestination(icon=ft.Icons.SETTINGS, label="Settings"),
            ],
        )
        self.nav_bar.on_change = self.on_nav_change

        self.page.appbar = ft.AppBar(
            title=self.screen_title,
            bgcolor=DARK_SURFACE,
            center_title=False,
        )

        self.page.add(
            ft.Column(
                [
                    ft.Container(content=self.content_area, expand=True),
                    self.bottom_status,
                ],
                expand=True,
                spacing=0,
            )
        )
        self.page.navigation_bar = self.nav_bar

    def on_nav_change(self, e: ft.ControlEvent):
        idx = e.control.selected_index
        views = [
            ("Prompt", self.prompt_view),
            ("Queue", self.queue_view),
            ("Images", self.images_view),
            ("Library", self.library_view),
            ("Settings", self.settings_view),
        ]
        title, view = views[idx]
        self.screen_title.value = title
        self.content_area.content = view
        if idx == 1:
            self.refresh_queue_list()
        if idx == 3:
            self.refresh_presets_list()
            self.refresh_chunks_list()
        self._safe_update(self.screen_title, self.content_area)

    # -------------------------------------------------------------------
    # Settings view
    # -------------------------------------------------------------------

    def build_settings_view(self):
        self.server_url_field = ft.TextField(
            label="ComfyUI server URL", value=self.config.server_url,
            hint_text="http://127.0.0.1:8188",
        )
        self.out_dir_field = ft.TextField(label="Output folder", value=self.config.out_dir, expand=True)
        pick_out_dir_btn = self._ev(
            ft.IconButton(icon=ft.Icons.FOLDER_OPEN), on_click=self.pick_output_dir,
        )

        self.workflow_status_text = ft.Text(self._workflow_status_label(), color=TEXT_DIM, size=12)
        import_workflow_btn = self._ev(
            ft.Button("Import workflow.json"), on_click=self.pick_workflow_file,
        )

        self.prompt_node_dropdown = self._ev(
            ft.Dropdown(label="Main prompt node", options=[]), on_select=self.on_prompt_node_change,
        )
        self.negative_node_dropdown = self._ev(
            ft.Dropdown(label="Negative prompt node (optional)", options=[]), on_select=self.on_negative_node_change,
        )
        self.lora_node_dropdown = self._ev(
            ft.Dropdown(label="LoRA stack node (rgthree)", options=[]), on_select=self.on_lora_node_change,
        )
        self.size_node_dropdown = self._ev(
            ft.Dropdown(label="Image size node (optional)", options=[]), on_select=self.on_size_node_change,
        )

        self.randomize_seed_switch = ft.Switch(label="Randomize seed every image", value=self.config.randomize_seed)
        self.jump_to_feed_switch = ft.Switch(label="Jump to Images tab when generating", value=self.config.jump_to_feed_on_start)
        self.auto_show_latest_switch = ft.Switch(label="Auto-preview latest image", value=self.config.auto_show_latest)

        refresh_loras_btn = self._ev(
            ft.Button("Refresh LoRA list from server"),
            on_click=lambda e: self.page.run_task(self.refresh_lora_list, True),
        )

        save_btn = self._ev(
            ft.Button("Save settings"), on_click=lambda e: self.save_settings_from_ui(show_message=True),
        )

        self._populate_node_dropdowns()

        self.settings_view = ft.Column(
            [
                self.section("ComfyUI Server", [
                    self.server_url_field,
                    ft.Row([self.out_dir_field, pick_out_dir_btn]),
                ]),
                self.section("Workflow", [
                    ft.Row([import_workflow_btn]),
                    self.workflow_status_text,
                    self.prompt_node_dropdown,
                    self.negative_node_dropdown,
                    self.lora_node_dropdown,
                    self.size_node_dropdown,
                    ft.Text(
                        "Pick which node has the width/height widgets to enable "
                        "the size presets on the Prompt tab. Leave as (none) to "
                        "keep whatever size is baked into the workflow.",
                        color=TEXT_DIM, size=11,
                    ),
                ]),
                self.section("LoRAs", [
                    refresh_loras_btn,
                    ft.Text(f"{len(self.available_loras)} LoRA(s) cached from server.", color=TEXT_DIM, size=12),
                ]),
                self.section("Behavior", [
                    self.randomize_seed_switch,
                    self.jump_to_feed_switch,
                    self.auto_show_latest_switch,
                ]),
                save_btn,
            ],
            scroll=ft.ScrollMode.AUTO,
            spacing=12,
        )

    def _workflow_status_label(self) -> str:
        if self.workflow_error:
            return f"Import error: {self.workflow_error}"
        if self.workflow is None:
            return "No workflow imported yet."
        text_count = len(self.workflow.text_nodes())
        lora_count = len(self.workflow.lora_stack_nodes())
        return f"Workflow loaded: {text_count} text node(s), {lora_count} LoRA stack node(s) found."

    def _populate_node_dropdowns(self):
        text_options = [ft.DropdownOption(key="", text="(none)")]
        self._text_node_fields: dict[str, str] = {}
        if self.workflow:
            for n in self.workflow.text_nodes():
                label = f"[{n['id']}] {n['title']} — {n['preview']}"
                text_options.append(ft.DropdownOption(key=n["id"], text=label))
                self._text_node_fields[n["id"]] = n["field"]

        lora_options = [ft.DropdownOption(key="", text="(none)")]
        if self.workflow:
            for n in self.workflow.lora_stack_nodes():
                label = f"[{n['id']}] {n['title']} ({n['slots']} slots)"
                lora_options.append(ft.DropdownOption(key=n["id"], text=label))

        size_options = [ft.DropdownOption(key="", text="(none)")]
        if self.workflow:
            for n in self.workflow.latent_size_nodes():
                label = f"[{n['id']}] {n['title']} ({n['width']}x{n['height']})"
                size_options.append(ft.DropdownOption(key=n["id"], text=label))

        self.prompt_node_dropdown.options = text_options
        self.negative_node_dropdown.options = text_options
        self.lora_node_dropdown.options = lora_options
        self.size_node_dropdown.options = size_options

        self.prompt_node_dropdown.value = self.config.prompt_node_id or ""
        self.negative_node_dropdown.value = self.config.negative_node_id or ""
        self.lora_node_dropdown.value = self.config.lora_node_id or ""
        self.size_node_dropdown.value = self.config.size_node_id or ""

    def on_prompt_node_change(self, e):
        node_id = self.prompt_node_dropdown.value or ""
        self.config.prompt_node_id = node_id
        self.config.prompt_field = self._text_node_fields.get(node_id, "text")
        self.storage.save_config(self.config)

    def on_negative_node_change(self, e):
        node_id = self.negative_node_dropdown.value or ""
        self.config.negative_node_id = node_id
        self.config.negative_field = self._text_node_fields.get(node_id, "text")
        self.storage.save_config(self.config)

    def on_lora_node_change(self, e):
        self.config.lora_node_id = self.lora_node_dropdown.value or ""
        self.storage.save_config(self.config)

    def on_size_node_change(self, e):
        self.config.size_node_id = self.size_node_dropdown.value or ""
        self.storage.save_config(self.config)

    async def pick_output_dir(self, e):
        path = await self.output_dir_picker.get_directory_path()
        if path:
            self.out_dir_field.value = path
            self._safe_update(self.out_dir_field)

    async def pick_workflow_file(self, e):
        files = await self.workflow_file_picker.pick_files(
            allow_multiple=False, allowed_extensions=["json"],
        )
        if not files:
            return

        path = files[0].path
        try:
            raw = self.storage.import_workflow_file(path)
        except Exception as ex:
            self.show_error(f"Couldn't read that file as JSON.\n\n{ex}")
            return

        self.workflow_raw = raw
        self._rebuild_workflow()

        if self.workflow_error:
            self.show_error(self.workflow_error)
        else:
            # Reset node picks - the old node ids likely don't exist in a
            # different workflow.
            self.config.prompt_node_id = ""
            self.config.negative_node_id = ""
            self.config.lora_node_id = ""
            self.config.size_node_id = ""
            self.storage.save_config(self.config)
            self.show_snack("Workflow imported.")

        self.workflow_status_text.value = self._workflow_status_label()
        self._populate_node_dropdowns()
        self._safe_update(
            self.workflow_status_text, self.prompt_node_dropdown,
            self.negative_node_dropdown, self.lora_node_dropdown, self.size_node_dropdown,
        )


    async def refresh_lora_list(self, show_message: bool):
        client = ComfyClient(self.server_url_field.value or self.config.server_url)
        try:
            async with aiohttp.ClientSession() as session:
                loras = await client.get_available_loras(session)
        except Exception as ex:
            if show_message:
                self.show_error(f"Couldn't reach ComfyUI to list LoRAs.\n\n{ex}")
            return

        self.available_loras = loras
        self.draft["available_loras"] = loras
        self.storage.save_draft(self.draft)

        if hasattr(self, "lora_pick_dropdown"):
            self.lora_pick_dropdown.options = [
                ft.DropdownOption(key=name, text=name) for name in self.available_loras
            ]
            self._safe_update(self.lora_pick_dropdown)

        if show_message:
            self.show_snack(f"Found {len(loras)} LoRA(s) on the server.")

    def save_settings_from_ui(self, show_message: bool):
        self.config.server_url = self.server_url_field.value.strip() or "http://127.0.0.1:8188"
        self.config.out_dir = self.out_dir_field.value.strip()
        self.config.randomize_seed = self.randomize_seed_switch.value
        self.config.jump_to_feed_on_start = self.jump_to_feed_switch.value
        self.config.auto_show_latest = self.auto_show_latest_switch.value
        self.storage.save_config(self.config)
        if show_message:
            self.show_snack("Settings saved.")

    # -------------------------------------------------------------------
    # Prompt editor view
    # -------------------------------------------------------------------

    def build_prompt_view(self):
        self.prompt_field = ft.TextField(
            label="Prompt", multiline=True, min_lines=4, max_lines=10,
            hint_text="Use $chunks, {char}, and ||random|choices|| freely.",
            value=self.draft.get("current_prompt", ""),
        )
        self.negative_field = ft.TextField(
            label="Negative prompt (optional)", multiline=True, min_lines=2, max_lines=6,
            value=self.draft.get("current_negative", ""),
        )
        self.count_field = ft.TextField(
            label="Count", value=str(self.draft.get("current_count", "1")), width=100,
        )

        self.size_presets = [
            ("832 x 1216 (portrait)", 832, 1216),
            ("1024 x 1024 (square)", 1024, 1024),
            ("1216 x 832 (landscape)", 1216, 832),
        ]
        self.size_dropdown = ft.Dropdown(
            label="Image size",
            options=[
                ft.DropdownOption(key=f"{w}x{h}", text=label)
                for label, w, h in self.size_presets
            ],
            value=self.draft.get("current_size", "832x1216"),
        )

        # Persist prompt/negative/count/size as the user edits them, so
        # they survive closing and reopening the app - not just when a
        # job is queued.
        self._ev(self.prompt_field, on_blur=lambda e: self._save_draft_state())
        self._ev(self.negative_field, on_blur=lambda e: self._save_draft_state())
        self._ev(self.count_field, on_blur=lambda e: self._save_draft_state())
        self._ev(self.size_dropdown, on_select=lambda e: self._save_draft_state())

        self.lora_pick_dropdown = self._ev(
            ft.Dropdown(
                label="LoRA", expand=True,
                options=[ft.DropdownOption(key=n, text=n) for n in self.available_loras],
            ),
            on_select=self.on_lora_pick_change,
        )
        self.lora_strength_field = ft.TextField(label="Strength", value="1.0", width=100)
        self.lora_trigger_field = ft.TextField(
            label="Trigger words (auto-added to prompt)", hint_text="e.g. mylorastyle, trigger_word",
        )
        add_lora_btn = self._ev(ft.IconButton(icon=ft.Icons.ADD), on_click=self.on_add_lora_click)

        self.current_loras_column = ft.Column(spacing=4)
        self._rebuild_current_loras_view()

        self.generate_button = self._ev(
            ft.Button("Generate", bgcolor=ACCENT, color="#1A1109"),
            on_click=lambda e: self.generate_current_prompt_with_count(),
        )
        add_to_queue_btn = self._ev(
            ft.Button("Add to Queue"), on_click=lambda e: self.add_current_job_to_queue(),
        )
        save_as_preset_btn = self._ev(
            ft.TextButton("Save as preset"), on_click=lambda e: self.save_preset_dialog(),
        )

        self.prompt_view = ft.Column(
            [
                self.section("Prompt", [
                    self.prompt_field,
                    self.negative_field,
                    ft.Row([self.count_field, self.size_dropdown]),
                ]),
                self.section("LoRAs", [
                    ft.Row([self.lora_pick_dropdown, self.lora_strength_field, add_lora_btn]),
                    self.lora_trigger_field,
                    self.current_loras_column,
                ]),
                ft.Row([self.generate_button, add_to_queue_btn, save_as_preset_btn]),
            ],
            scroll=ft.ScrollMode.AUTO,
            spacing=12,
        )

    def on_lora_pick_change(self, e):
        name = self.lora_pick_dropdown.value or ""
        self.lora_trigger_field.value = self.lora_triggers.get(name, "")
        self._safe_update(self.lora_trigger_field)

    def on_add_lora_click(self, e):
        name = self.lora_pick_dropdown.value
        if not name:
            self.show_snack("Pick a LoRA first (or refresh the list in Settings).")
            return
        strength = self.safe_float(self.lora_strength_field.value, 1.0)
        trigger = (self.lora_trigger_field.value or "").strip()

        self.current_loras.append(LoraSlotData(name=name, strength=strength, trigger=trigger))

        # Remember this LoRA's trigger words for next time it's picked,
        # regardless of which job/preset it ends up in.
        if trigger:
            self.lora_triggers[name] = trigger
            self.storage.save_lora_triggers(self.lora_triggers)

        self._rebuild_current_loras_view()
        self._save_draft_state()

    def _rebuild_current_loras_view(self):
        self.current_loras_column.controls.clear()
        for i, lora in enumerate(self.current_loras):
            subtitle = f"x{lora.strength:.2f}"
            if lora.trigger:
                subtitle += f" · trigger: {lora.trigger}"
            self.current_loras_column.controls.append(
                ft.Row(
                    [
                        ft.Text(lora.name, expand=True, color=TEXT_MAIN),
                        ft.Text(subtitle, color=TEXT_DIM, size=11),
                        self._ev(
                            ft.IconButton(icon=ft.Icons.CLOSE, icon_size=16),
                            on_click=lambda e, idx=i: self.remove_current_lora(idx),
                        ),
                    ],
                )
            )
        self._safe_update(self.current_loras_column)

    def remove_current_lora(self, idx: int):
        if 0 <= idx < len(self.current_loras):
            del self.current_loras[idx]
            self._rebuild_current_loras_view()
            self._save_draft_state()

    def _selected_size(self) -> tuple[int, int]:
        raw = self.size_dropdown.value or "832x1216"
        try:
            w, h = raw.split("x")
            return int(w), int(h)
        except Exception:
            return 832, 1216

    def collect_job_from_editor(self, force_count: int | None = None) -> JobData | None:
        prompt = (self.prompt_field.value or "").strip()
        if not prompt:
            self.show_error("Enter a prompt first.")
            return None

        count = force_count if force_count is not None else self.safe_int(self.count_field.value, 1)
        count = max(1, count)
        width, height = self._selected_size()

        return JobData(
            prompt=prompt,
            negative_prompt=(self.negative_field.value or "").strip(),
            count=count,
            loras=[LoraSlotData(name=l.name, strength=l.strength, trigger=l.trigger) for l in self.current_loras],
            width=width,
            height=height,
        )

    def add_current_job_to_queue(self):
        job = self.collect_job_from_editor()
        if not job:
            return
        self.jobs.append(job)
        self._save_draft_state()
        self.refresh_queue_list()
        self.show_snack("Added to queue.")

    def generate_current_prompt_once(self):
        """Used by the Images tab's Generate button - always generates the
        current prompt exactly once, regardless of the Count field."""
        job = self.collect_job_from_editor(force_count=1)
        if not job:
            return
        self.start_jobs([job], show_status=False)

    def generate_current_prompt_with_count(self):
        """Used by the Prompt tab's Generate button - honors the Count
        field, generating that many images and showing a "total x/y"
        progress bar while it runs."""
        job = self.collect_job_from_editor()
        if not job:
            return
        self.start_jobs([job], show_status=True)

    def _save_draft_state(self):
        self.draft["jobs"] = [j.to_dict() for j in self.jobs]
        self.draft["current_loras"] = [l.to_dict() for l in self.current_loras]
        self.draft["current_prompt"] = self.prompt_field.value or ""
        self.draft["current_negative"] = self.negative_field.value or ""
        self.draft["current_count"] = self.count_field.value or "1"
        self.draft["current_size"] = self.size_dropdown.value or "832x1216"
        self.storage.save_draft(self.draft)

    # -------------------------------------------------------------------
    # Queue view
    # -------------------------------------------------------------------

    def build_queue_view(self):
        self.queue_list_column = ft.Column(spacing=6)
        self.queue_loop_field = ft.TextField(label="Loop queue", value="1", width=100)

        start_btn = ft.Button("Start Queue", on_click=lambda e: self.start_queue())
        clear_btn = ft.TextButton("Clear", on_click=lambda e: self.clear_queue())

        save_queue_btn = ft.TextButton("Save queue as...", on_click=lambda e: self.save_queue_dialog())
        self.saved_queues_column = ft.Column(spacing=4)

        self.char_replace_field = ft.TextField(
            label="Replace {char} with", hint_text="e.g. Alice", expand=True,
        )
        apply_char_replace_btn = ft.Button(
            "Apply", on_click=lambda e: self.apply_char_replacement(),
        )

        self.queue_view = ft.Column(
            [
                self.section("Queue", [
                    self.queue_list_column,
                    ft.Row([self.queue_loop_field, start_btn, clear_btn]),
                ]),
                self.section("Special tags", [
                    ft.Text(
                        "Type {char} anywhere in a prompt as a placeholder, then "
                        "swap every instance across the whole queue here.",
                        color=TEXT_DIM, size=11,
                    ),
                    ft.Row([self.char_replace_field, apply_char_replace_btn]),
                ]),
                self.section("Saved queues", [
                    save_queue_btn,
                    self.saved_queues_column,
                ]),
            ],
            scroll=ft.ScrollMode.AUTO,
            spacing=12,
        )

    def refresh_queue_list(self):
        self.queue_list_column.controls.clear()
        if not self.jobs:
            self.queue_list_column.controls.append(ft.Text("Queue is empty.", color=TEXT_DIM))
        for i, job in enumerate(self.jobs):
            preview = job.prompt[:60] + ("..." if len(job.prompt) > 60 else "")
            lora_names = ", ".join(l.name for l in job.loras) or "no LoRAs"
            self.queue_list_column.controls.append(
                ft.Container(
                    bgcolor=CARD_BG, border_radius=8, padding=10,
                    content=ft.Row(
                        [
                            ft.Column(
                                [
                                    ft.Text(preview, color=TEXT_MAIN, size=13),
                                    ft.Text(f"count={job.count} · {lora_names}", color=TEXT_DIM, size=11),
                                ],
                                expand=True, spacing=2,
                            ),
                            ft.IconButton(icon=ft.Icons.DELETE_OUTLINE, icon_size=18, on_click=lambda e, idx=i: self.remove_job(idx)),
                        ],
                    ),
                )
            )
        self._safe_update(self.queue_list_column)
        self.refresh_saved_queues_list()

    def remove_job(self, idx: int):
        if 0 <= idx < len(self.jobs):
            del self.jobs[idx]
            self._save_draft_state()
            self.refresh_queue_list()

    def clear_queue(self):
        self.jobs.clear()
        self._save_draft_state()
        self.refresh_queue_list()

    def save_queue_dialog(self):
        if not self.jobs:
            self.show_snack("Queue is empty.")
            return
        self.prompt_dialog("Save Queue", "Name", self.save_queue_as)

    def save_queue_as(self, name: str):
        self.bundle.queues[name] = [j.to_dict() for j in self.jobs]
        self.storage.save_presets_bundle(self.bundle)
        self.refresh_saved_queues_list()
        self.show_snack(f"Saved queue '{name}'.")

    def refresh_saved_queues_list(self):
        self.saved_queues_column.controls.clear()
        for name in sorted(self.bundle.queues.keys()):
            self.saved_queues_column.controls.append(
                ft.Row(
                    [
                        ft.Text(name, expand=True, color=TEXT_MAIN),
                        ft.TextButton("Load", on_click=lambda e, n=name: self.load_saved_queue(n)),
                        ft.IconButton(icon=ft.Icons.DELETE_OUTLINE, icon_size=16, on_click=lambda e, n=name: self.delete_saved_queue(n)),
                    ],
                )
            )
        self._safe_update(self.saved_queues_column)

    def load_saved_queue(self, name: str):
        data = self.bundle.queues.get(name, [])
        self.jobs = [JobData.from_dict(j) for j in data]
        self._save_draft_state()
        self.refresh_queue_list()
        self.show_snack(f"Loaded queue '{name}'.")

    def delete_saved_queue(self, name: str):
        self.bundle.queues.pop(name, None)
        self.storage.save_presets_bundle(self.bundle)
        self.refresh_saved_queues_list()

    def apply_char_replacement(self):
        value = (self.char_replace_field.value or "").strip()
        if not value:
            self.show_snack("Enter a value to replace {char} with.")
            return
        if not self.jobs:
            self.show_snack("Queue is empty.")
            return

        replaced = 0
        for job in self.jobs:
            if "{char}" in job.prompt:
                job.prompt = job.prompt.replace("{char}", value)
                replaced += 1
            if "{char}" in job.negative_prompt:
                job.negative_prompt = job.negative_prompt.replace("{char}", value)

        self._save_draft_state()
        self.refresh_queue_list()
        self.show_snack(f"Replaced {{char}} in {replaced} job(s).")

    def start_queue(self):
        if not self.jobs:
            self.show_error("Queue is empty.")
            return

        loops = max(1, self.safe_int(self.queue_loop_field.value, 1))
        jobs_to_run = []
        for _ in range(loops):
            for job in self.jobs:
                jobs_to_run.append(JobData.from_dict(job.to_dict()))

        self.start_jobs(jobs_to_run, show_status=True)

    # -------------------------------------------------------------------
    # Library view (memory presets + chunks)
    # -------------------------------------------------------------------

    def build_library_view(self):
        save_preset_btn = ft.TextButton("Save current prompt as preset...", on_click=lambda e: self.save_preset_dialog())
        self.presets_column = ft.Column(spacing=4)

        add_chunk_btn = ft.TextButton("Add chunk...", on_click=lambda e: self.add_chunk_dialog())
        self.chunks_column = ft.Column(spacing=4)

        self.library_view = ft.Column(
            [
                self.section("Memory Presets", [save_preset_btn, self.presets_column]),
                self.section("Chunks ($name -> text)", [add_chunk_btn, self.chunks_column]),
            ],
            scroll=ft.ScrollMode.AUTO,
            spacing=12,
        )

    def save_preset_dialog(self):
        job = self.collect_job_from_editor()
        if not job:
            return
        self.prompt_dialog("Save Preset", "Name", lambda name: self.save_preset_as(name, job))

    def save_preset_as(self, name: str, job: JobData):
        self.bundle.presets[name] = job.to_dict()
        self.storage.save_presets_bundle(self.bundle)
        self.refresh_presets_list()
        self.show_snack(f"Saved preset '{name}'.")

    def refresh_presets_list(self):
        self.presets_column.controls.clear()
        for name in sorted(self.bundle.presets.keys()):
            self.presets_column.controls.append(
                ft.Row(
                    [
                        ft.Text(name, expand=True, color=TEXT_MAIN),
                        ft.TextButton("Load", on_click=lambda e, n=name: self.load_preset(n)),
                        ft.TextButton("Add to Queue", on_click=lambda e, n=name: self.add_preset_to_queue(n)),
                        ft.IconButton(icon=ft.Icons.DELETE_OUTLINE, icon_size=16, on_click=lambda e, n=name: self.delete_preset(n)),
                    ],
                )
            )
        self._safe_update(self.presets_column)

    def load_preset(self, name: str):
        data = self.bundle.presets.get(name)
        if not data:
            return
        job = JobData.from_dict(data)
        self.prompt_field.value = job.prompt
        self.negative_field.value = job.negative_prompt
        self.count_field.value = str(job.count)
        self.size_dropdown.value = f"{job.width}x{job.height}"
        self.current_loras = [LoraSlotData(name=l.name, strength=l.strength, trigger=l.trigger) for l in job.loras]
        self._rebuild_current_loras_view()
        self._safe_update(self.prompt_field, self.negative_field, self.count_field, self.size_dropdown)
        self._save_draft_state()
        self.show_snack(f"Loaded preset '{name}'.")

    def delete_preset(self, name: str):
        self.bundle.presets.pop(name, None)
        self.storage.save_presets_bundle(self.bundle)
        self.refresh_presets_list()

    def add_preset_to_queue(self, name: str):
        data = self.bundle.presets.get(name)
        if not data:
            return
        job = JobData.from_dict(data)
        self.jobs.append(job)
        self._save_draft_state()
        self.refresh_queue_list()
        self.show_snack(f"Added preset '{name}' to queue.")

    def add_chunk_dialog(self):
        self.two_field_dialog("Add Chunk", "Name (without $)", "Text", self.add_chunk)

    def add_chunk(self, name: str, text: str):
        name = name.lstrip("$").strip()
        self.chunks[name] = text
        self.storage.save_chunks(self.chunks)
        self.refresh_chunks_list()

    def edit_chunk_dialog(self, name: str):
        self.two_field_dialog(
            "Edit Chunk", "Name (without $)", "Text",
            lambda new_name, new_text: self.edit_chunk(name, new_name, new_text),
            initial_a=name, initial_b=self.chunks.get(name, ""),
        )

    def edit_chunk(self, old_name: str, new_name: str, new_text: str):
        new_name = new_name.lstrip("$").strip()
        if old_name != new_name:
            self.chunks.pop(old_name, None)
        self.chunks[new_name] = new_text
        self.storage.save_chunks(self.chunks)
        self.refresh_chunks_list()

    def delete_chunk(self, name: str):
        self.chunks.pop(name, None)
        self.storage.save_chunks(self.chunks)
        self.refresh_chunks_list()

    def refresh_chunks_list(self):
        self.chunks_column.controls.clear()
        for name in sorted(self.chunks.keys()):
            preview = self.chunks[name][:40]
            self.chunks_column.controls.append(
                ft.Row(
                    [
                        ft.Text(f"${name}", color=ACCENT, size=13),
                        ft.Text(preview, expand=True, color=TEXT_DIM, size=12),
                        ft.IconButton(icon=ft.Icons.EDIT, icon_size=16, on_click=lambda e, n=name: self.edit_chunk_dialog(n)),
                        ft.IconButton(icon=ft.Icons.DELETE_OUTLINE, icon_size=16, on_click=lambda e, n=name: self.delete_chunk(n)),
                    ],
                )
            )
        self._safe_update(self.chunks_column)

    # -------------------------------------------------------------------
    # Images view
    # -------------------------------------------------------------------

    def build_images_view(self):
        # ft.Image with src=None renders as a big "Image must have src
        # specified" error box, so keep it invisible until there's
        # actually a src, and show this placeholder instead.
        self.image_placeholder = ft.Column(
            [
                ft.Icon(ft.Icons.IMAGE_OUTLINED, size=48, color=TEXT_DIM),
                ft.Text("No images yet", color=TEXT_DIM),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            visible=True,
        )
        self.main_preview_image = ft.Image(
            src=None, fit=ft.BoxFit.CONTAIN, border_radius=8, expand=True,
            visible=False,
            # gapless_playback keeps the previous frame on screen while the
            # new src decodes, instead of flashing blank in between - this
            # is what was causing the flicker when paging through images.
            gapless_playback=True,
        )
        self.thumbnails_row = ft.Row(spacing=6, scroll=ft.ScrollMode.AUTO)

        prev_btn = ft.IconButton(icon=ft.Icons.CHEVRON_LEFT, on_click=lambda e: self.prev_image())
        next_btn = ft.IconButton(icon=ft.Icons.CHEVRON_RIGHT, on_click=lambda e: self.next_image())

        self.images_generate_button = self._ev(
            ft.Button("Generate", bgcolor=ACCENT, color="#1A1109"),
            on_click=lambda e: self.generate_current_prompt_once(),
        )

        self.images_view = ft.Column(
            [
                ft.Container(
                    expand=True, bgcolor=CARD_BG, border_radius=8,
                    content=ft.Row(
                        [
                            prev_btn,
                            ft.Stack(
                                [self.image_placeholder, self.main_preview_image],
                                alignment=ft.Alignment.CENTER,
                                expand=True,
                            ),
                            next_btn,
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        expand=True,
                    ),
                ),
                ft.Row([self.images_generate_button], alignment=ft.MainAxisAlignment.CENTER),
                ft.Container(height=110, content=self.thumbnails_row),
            ],
            expand=True,
        )

    def get_cached_preview_src(self, path: str) -> str:
        if path not in self.preview_src_cache:
            self.preview_src_cache[path] = ImageService.image_file_to_resized_data_uri(path)
        return self.preview_src_cache[path]

    def get_cached_thumbnail_src(self, path: str) -> str:
        if path not in self.thumbnail_src_cache:
            self.thumbnail_src_cache[path] = ImageService.image_file_to_thumbnail_data_uri(path)
        return self.thumbnail_src_cache[path]

    def update_preview_display(self, index: int | None = None):
        if not self.session_generated_images:
            self.main_preview_image.src = None
            self.main_preview_image.visible = False
            self.image_placeholder.visible = True
            self._safe_update(self.main_preview_image, self.image_placeholder)
            return

        if index is None:
            index = self.current_preview_index
        index = max(0, min(index, len(self.session_generated_images) - 1))
        self.current_preview_index = index

        path = self.session_generated_images[index]
        try:
            self.main_preview_image.src = self.get_cached_preview_src(path)
            self.main_preview_image.visible = True
            self.image_placeholder.visible = False
        except Exception:
            self.main_preview_image.src = None
            self.main_preview_image.visible = False
            self.image_placeholder.visible = True
        self._safe_update(self.main_preview_image, self.image_placeholder)
        self._refresh_thumbnails_row()

    def _refresh_thumbnails_row(self):
        total = len(self.session_generated_images)
        if total == 0:
            self.thumbnails_row.controls.clear()
            self._safe_update(self.thumbnails_row)
            return

        existing = len(self.thumbnails_row.controls)
        appended = existing < total
        for i in range(existing, total):
            path = self.session_generated_images[i]
            try:
                img = ft.Image(
                    src=self.get_cached_thumbnail_src(path), width=96, height=96,
                    fit=ft.BoxFit.COVER, border_radius=6, gapless_playback=True,
                )
            except Exception:
                img = ft.Text("?", color=TEXT_DIM)

            self.thumbnails_row.controls.append(
                ft.Container(
                    padding=3, border_radius=8, bgcolor=CARD_BG, content=img,
                    on_click=lambda e, idx=i: self.update_preview_display(idx),
                )
            )

        if appended:
            self._safe_update(self.thumbnails_row)

        for i, container in enumerate(self.thumbnails_row.controls):
            desired = ACCENT if i == self.current_preview_index else CARD_BG
            if container.bgcolor != desired:
                container.bgcolor = desired
                self._safe_update(container)

    def prev_image(self):
        if self.session_generated_images:
            self.update_preview_display(self.current_preview_index - 1)

    def next_image(self):
        if self.session_generated_images:
            self.update_preview_display(self.current_preview_index + 1)

    def add_generated_image_to_feed(self, path: str):
        self.session_generated_images.append(path)
        if self.config.auto_show_latest:
            self.update_preview_display(len(self.session_generated_images) - 1)
        else:
            self._refresh_thumbnails_row()

    # -------------------------------------------------------------------
    # Generation orchestration
    # -------------------------------------------------------------------

    def set_generate_button_busy(self, busy: bool):
        label = "Generating..." if busy else "Generate"
        bgcolor = ACCENT_BUSY if busy else ACCENT
        buttons = [
            getattr(self, "generate_button", None),
            getattr(self, "images_generate_button", None),
        ]
        for btn in buttons:
            if btn is None:
                continue
            try:
                # Flet's Button has no `.text` field in this version - the
                # label lives in `.content`. Setting `.text` was a silent
                # no-op, which is also why the initial "Generate" label
                # only ever became visible after something else forced a
                # repaint of the button (e.g. a click).
                btn.content = label
                btn.disabled = busy
                btn.bgcolor = bgcolor
            except Exception:
                pass
        self._safe_update(*[b for b in buttons if b is not None])

    def start_jobs(self, jobs: list[JobData], show_status: bool = True):
        self._show_status_bar = show_status
        if self.is_running:
            self.show_snack("Generation is already running.")
            return

        self.save_settings_from_ui(show_message=False)

        workflow = self.get_ready_workflow()
        if workflow is None:
            return

        if not self.config.out_dir.strip():
            self.show_error("Please select or enter an output folder in Settings.")
            return

        self.is_running = True
        self.set_generate_button_busy(True)

        if self.config.jump_to_feed_on_start:
            try:
                self.nav_bar.selected_index = 2
                self.screen_title.value = "Images"
                self.content_area.content = self.images_view
            except Exception:
                pass

        self.status_left.value = "Starting..."
        self.status_right.value = ""
        self.bottom_status.visible = show_status
        self._safe_update(
            self.status_left, self.status_right, self.bottom_status,
            self.generate_button, self.nav_bar, self.content_area, self.screen_title,
        )

        self.page.run_task(self.run_generation_task, jobs, workflow)

    async def run_generation_task(self, jobs: list[JobData], workflow: ComfyWorkflow):
        prompt_engine = PromptEngine(self.chunks)

        callbacks = GenerationCallbacks(
            on_status=self.on_generation_status,
            on_progress=self.on_generation_progress,
            on_counts=self.on_generation_counts,
            on_image_saved=self.on_generation_image_saved,
            on_error=self.on_generation_error,
            on_batch_dir=self.on_generation_batch_dir,
            on_finished=self.on_generation_finished,
        )

        service = GenerationService(prompt_engine=prompt_engine, callbacks=callbacks)

        try:
            self.session_batch_dir = await service.process_jobs(
                config=self.config,
                jobs=jobs,
                current_session_batch_dir=self.session_batch_dir,
                workflow=workflow,
            )
        finally:
            self.is_running = False
            self.set_generate_button_busy(False)
            self.status_left.value = "Finished."
            self.bottom_status.visible = False
            self._safe_update(self.generate_button, self.status_left, self.bottom_status)

    def on_generation_status(self, message: str):
        if not self._show_status_bar:
            return
        self.status_left.value = message
        self.bottom_status.visible = True
        self._safe_update(self.status_left, self.bottom_status)

    def on_generation_counts(self, job_index, job_total, image_index, image_total, done, grand_total):
        self.status_left.value = f"Job {job_index}/{job_total} | Image {image_index}/{image_total}"
        self.status_right.value = f"Total: {done + 1}/{grand_total}"
        if not self._show_status_bar:
            return
        self.bottom_status.visible = True
        self._safe_update(self.status_left, self.status_right, self.bottom_status)

    def on_generation_progress(self, pct: float):
        pass

    def on_generation_image_saved(self, path: str):
        self.add_generated_image_to_feed(path)

    def on_generation_error(self, message: str):
        self.show_error(message)

    def on_generation_batch_dir(self, path: str):
        self.session_batch_dir = path

    def on_generation_finished(self):
        self.status_left.value = "Finished all jobs."
        self.status_right.value = ""
        self.bottom_status.visible = False
        self._safe_update(self.status_left, self.status_right, self.bottom_status)

    # -------------------------------------------------------------------
    # Bulk refresh
    # -------------------------------------------------------------------

    def refresh_all_dynamic_views(self):
        self.refresh_queue_list()
        self.refresh_presets_list()
        self.refresh_chunks_list()
        self.update_preview_display()
