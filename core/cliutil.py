"""click helpers shared by the CLIs: workspace lookup and ChArUco board options."""

from __future__ import annotations

import click

from . import workspace
from .board import BoardSpec


def workspace_or_die(photos: str | None) -> workspace.Workspace:
    try:
        return workspace.Workspace(workspace.resolve_photos_dir(photos))
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc))


def board_options(f):
    """--squares-x/-y, --square-mm, --marker-mm, --dictionary; None when not given."""
    d = BoardSpec()
    for opt in reversed([
        click.option("--squares-x", type=int, default=None, help=f"[{d.squares_x}]"),
        click.option("--squares-y", type=int, default=None, help=f"[{d.squares_y}]"),
        click.option("--square-mm", type=float, default=None,
                     help=f"Measured printed square size [{d.square_mm}]."),
        click.option("--marker-mm", type=float, default=None, help=f"[{d.marker_mm}]"),
        click.option("--dictionary", default=None, help=f"[{d.dictionary}]"),
    ]):
        f = opt(f)
    return f


def make_spec(saved: dict | None = None, **opts) -> BoardSpec:
    """Board spec from defaults, then a previously saved spec, then given options."""
    base = dict(BoardSpec().to_dict(), **(saved or {}))
    base.update({k: v for k, v in opts.items() if v is not None})
    return BoardSpec(**base)
