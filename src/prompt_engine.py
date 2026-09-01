from __future__ import annotations

import random
import re

from models import JobData


class PromptEngine:
    def __init__(self, chunks: dict[str, str]):
        self.chunks = chunks

    def apply_chunks_recursive(self, text: str) -> str:
        if not text:
            return ""

        for _ in range(10):
            found = False
            # Replace longest chunk names first so $art doesn't
            # partially match inside $artist. Use a word boundary
            # so $art only matches when not followed by a name char.
            for k in sorted(self.chunks.keys(), key=len, reverse=True):
                v = self.chunks[k]
                pattern = r"\$" + re.escape(k) + r"(?![A-Za-z0-9_])"
                new_text = re.sub(pattern, lambda m: v, text)
                if new_text != text:
                    text = new_text
                    found = True

            if not found:
                break

        return text

    def _find_innermost_randomizer(self, text: str) -> tuple[int, int] | None:
        """
        Find the innermost ||...|| pair.

        We locate the LAST '||' opening marker, then the FIRST '||'
        closing marker that appears after it. This guarantees we always
        resolve the most deeply nested block first, so nested chunks like
        ||$r1|$r2|| (where $r1 = ||yellow|blue||) collapse correctly.
        """
        markers = [m.start() for m in re.finditer(r"\|\|", text)]
        if len(markers) < 2:
            return None

        open_pos = None
        for i in range(len(markers) - 1):
            open_pos = markers[i]
            close_pos = markers[i + 1]
            if close_pos >= open_pos + 2:
                inner = text[open_pos + 2:close_pos]
                if "||" not in inner:
                    return (open_pos, close_pos)

        return None

    def parse_randomizer(self, text: str) -> str:
        if not text:
            return ""

        for _ in range(200):
            found = self._find_innermost_randomizer(text)
            if found:
                open_pos, close_pos = found
                inner_content = text[open_pos + 2:close_pos]
                choices = inner_content.split("|")
                chosen = random.choice(choices) if choices else ""
                # Expand chunks only in the chosen value, not the whole
                # text, so chunk values containing ||...|| don't leak into
                # adjacent randomizer blocks.
                chosen = self.apply_chunks_recursive(chosen)
                text = text[:open_pos] + chosen + text[close_pos + 2:]
            else:
                expanded = self.apply_chunks_recursive(text)
                if expanded == text:
                    break
                text = expanded

        return text

    def resolve_job(self, job: JobData) -> tuple[str, str]:
        """Resolves chunks + ||randomizer|| blocks for a job's main and
        negative prompt text. Called fresh for every image so randomizer
        blocks re-roll each generation."""
        resolved_main = self.parse_randomizer(job.prompt)
        resolved_neg = self.parse_randomizer(job.negative_prompt)
        return resolved_main, resolved_neg
