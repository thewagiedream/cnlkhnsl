import flet as ft

from app_controller import ComfyCompanionApp

# Palette
DARK_BG = "#0E0E13"
DARK_SURFACE = "#1F1F26"
ACCENT = "#FF9C66"
ACCENT_LIGHT = "#FFB166"


async def main(page: ft.Page):
    try:
        page.title = "Comfy Companion"
        page.padding = 0
        page.theme_mode = ft.ThemeMode.DARK
        page.bgcolor = DARK_BG

        page.theme = ft.Theme(
            use_material3=True,
            color_scheme=ft.ColorScheme(
                primary=ACCENT,
                on_primary="#1A1109",
                primary_container=ACCENT_LIGHT,
                secondary=ACCENT_LIGHT,
                surface=DARK_SURFACE,
                on_surface="#ECECF0",
                outline="#3A3A45",
            ),
            scrollbar_theme=ft.ScrollbarTheme(
                thumb_color={ft.ControlState.DEFAULT: "#3A3A45"},
            ),
        )
        try:
            page.theme.scroll_clip_behavior = ft.ClipBehavior.HARD_EDGE
        except Exception:
            pass
        try:
            page.theme.page_transitions = ft.PageTransitionsTheme()
        except Exception:
            pass

        try:
            page.window.width = 420
            page.window.height = 860
        except Exception:
            pass

        app = ComfyCompanionApp(page)
        await app.initialize()

    except Exception as ex:
        import traceback
        tb = traceback.format_exc()
        page.controls.clear()
        page.add(
            ft.Container(
                expand=True,
                bgcolor="#0E0E13",
                padding=20,
                content=ft.Column(
                    scroll=ft.ScrollMode.AUTO,
                    controls=[
                        ft.Text("Startup Error", size=22,
                                color="#FF9C66", weight=ft.FontWeight.BOLD),
                        ft.Text(str(ex), color="#FFFFFF"),
                        ft.Text(tb, color="#BBBBBB", size=12, selectable=True),
                    ],
                ),
            )
        )
        page.update()


if __name__ == "__main__":
    ft.run(main)
