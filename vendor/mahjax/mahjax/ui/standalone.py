# Copyright 2025 The Mahjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Showing states with no server: in a notebook's output, or as one HTML file.

The page is the server's replay screen with its script, its styles, the tile
art and every frame written into it, so the browser only draws what it carries.
"""

from __future__ import annotations

import base64
import gzip
import html
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Union

from .record import StateReplay

STATIC_DIR = Path(__file__).resolve().parent / "static"
TILE_DIR = Path(__file__).resolve().parents[1] / "_src" / "assets" / "tiles"
TILE_SETS = ("ja", "en")


def show(states: Any, *, height: int = 640) -> None:
    """Show a state, or play a list of states back, on the replay screen.

    ``states`` is one unbatched state, a list of them, or a state with a leading
    time axis, from any of the envs. In a notebook the screen is the
    cell's output, ``height`` pixels tall; anywhere else it is saved to a
    temporary HTML file, and the file's path is printed.
    """
    page = _page(StateReplay(states))
    if _in_notebook():
        from IPython.display import display

        # A raw bundle: IPython's HTML() warns on a lone iframe and points to its
        # IFrame, which takes a URL, not a page.
        display({"text/html": _iframe(page, height), "text/plain": "MahJax replay"}, raw=True)
        return
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", prefix="mahjax-", suffix=".html", delete=False
    ) as file:
        file.write(page)
    print(f"MahJax: open {file.name} in a browser")


def save_html(states: Any, path: Union[str, Path]) -> None:
    """Save the replay screen of ``states`` (as ``show`` takes them) to one HTML
    file, which any browser opens with no server."""
    Path(path).write_text(_page(StateReplay(states)), encoding="utf-8")


def _page(replay: StateReplay) -> str:
    data = {
        "replay": replay.to_dict(),
        "frames": replay.frames(0, replay.total),
        "tiles": _tiles(),
    }
    style = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    script = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    if "</style" in style or "</script" in script:
        raise RuntimeError("styles.css or app.js would close its inline tag early")
    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for served, inline in (
        ("<body>", '<body class="standalone">'),
        ('<link rel="stylesheet" href="/static/styles.css">', f"<style>\n{style}</style>"),
        (
            '<script type="module" src="/static/app.js"></script>',
            f'<script id="mahjax-data" type="application/octet-stream">{_pack(data)}</script>\n'
            f'<script type="module">\n{script}</script>',
        ),
    ):
        if page.count(served) != 1:
            raise RuntimeError(f"index.html no longer has {served!r} exactly once")
        page = page.replace(served, inline)
    return page


def _pack(data: Any) -> str:
    # A game's frames repeat themselves, so gzip leaves about 2% of the JSON.
    # The page unpacks it with the browser's DecompressionStream.
    raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(gzip.compress(raw, compresslevel=9, mtime=0)).decode("ascii")


def _tiles() -> Dict[str, Dict[str, str]]:
    # Both sets, so the page's tile switch works as it does on the server.
    return {
        name: {path.name: path.read_text(encoding="utf-8") for path in sorted((TILE_DIR / name).glob("*.svg"))}
        for name in TILE_SETS
    }


def _iframe(page: str, height: int) -> str:
    # The frame keeps the page's styles and script apart from the notebook's own.
    return (
        f'<iframe srcdoc="{html.escape(page)}" '
        f'style="width: 100%; height: {int(height)}px; border: 0;"></iframe>'
    )


def _in_notebook() -> bool:
    """A Jupyter kernel (JupyterLab, VS Code, Colab), which draws HTML; a terminal IPython does not."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and getattr(shell, "kernel", None) is not None
