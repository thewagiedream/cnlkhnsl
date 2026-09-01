from __future__ import annotations

import base64
import io
from pathlib import Path

try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None


class ImageService:
    @staticmethod
    def mime_for_path(path: str | Path) -> str:
        suffix = Path(path).suffix.lower()

        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if suffix == ".webp":
            return "image/webp"
        if suffix == ".gif":
            return "image/gif"

        return "image/png"

    @staticmethod
    def image_file_to_data_uri(path: str | Path) -> str:
        path = Path(path)
        mime = ImageService.mime_for_path(path)
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def _resized_data_uri(
        path: str | Path,
        max_width: int,
        max_height: int,
        quality: int,
    ) -> str:
        path = Path(path)

        if PILImage is None:
            return ImageService.image_file_to_data_uri(path)

        try:
            with PILImage.open(path) as img:
                img.thumbnail((max_width, max_height), PILImage.Resampling.LANCZOS)

                if img.mode in ("RGBA", "LA", "P"):
                    bg = PILImage.new("RGB", img.size, (255, 255, 255))
                    if img.mode == "P":
                        img = img.convert("RGBA")
                    bg.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
                    img = bg
                else:
                    img = img.convert("RGB")

                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=quality, optimize=True)
                b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                return f"data:image/jpeg;base64,{b64}"

        except Exception:
            return ImageService.image_file_to_data_uri(path)

    @staticmethod
    def image_file_to_resized_data_uri(
        path: str | Path,
        max_width: int = 900,
        max_height: int = 900,
        quality: int = 85,
    ) -> str:
        """Lightweight JPEG data URI for the full-size UI preview."""
        return ImageService._resized_data_uri(path, max_width, max_height, quality)

    @staticmethod
    def image_file_to_thumbnail_data_uri(
        path: str | Path,
        width: int = 96,
        height: int = 128,
        quality: int = 75,
    ) -> str:
        """Tiny cached thumbnail source for the session feed strip."""
        return ImageService._resized_data_uri(path, width, height, quality)

    @staticmethod
    def create_or_reuse_session_batch_dir(
        out_dir: str,
        current_session_batch_dir: str | None,
    ) -> Path:
        if current_session_batch_dir:
            path = Path(current_session_batch_dir)
            path.mkdir(parents=True, exist_ok=True)
            return path

        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)

        existing = [
            p for p in root.iterdir()
            if p.is_dir() and p.name.isdigit() and len(p.name) == 4
        ]

        if existing:
            highest = max(int(p.name) for p in existing)
            next_name = f"{highest + 1:04d}"
        else:
            next_name = "0001"

        batch_dir = root / next_name
        batch_dir.mkdir(parents=True, exist_ok=True)
        return batch_dir

    @staticmethod
    def next_png_counter(batch_dir: Path) -> int:
        existing_pngs = [
            p for p in batch_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".png"
        ]
        return len(existing_pngs) + 1

    @staticmethod
    def save_generated_image(image: bytes | bytearray, filepath: Path):
        """Saves the raw PNG bytes returned by ComfyUI's /view endpoint."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(image, bytearray):
            image = bytes(image)

        if isinstance(image, bytes):
            filepath.write_bytes(image)
            return

        if PILImage is not None and isinstance(image, PILImage.Image):
            image.save(str(filepath))
            return

        raise RuntimeError(f"Could not save generated image to: {filepath}")

    @staticmethod
    def image_file_to_base64(path: str | Path) -> str:
        p = Path(path)
        return base64.b64encode(p.read_bytes()).decode("utf-8")
