"""Subprocess entry point for model-written CadQuery: python -m scan2cad.cad.runner SCRIPT OUT_PREFIX

The script runs with `cq` (cadquery) and `math` preloaded and must assign `result`.
"""

from __future__ import annotations

import json
import math
import runpy
import sys
from pathlib import Path


def main() -> None:
    script, prefix = Path(sys.argv[1]), Path(sys.argv[2])
    import cadquery as cq

    from scan2cad import cad

    scope = runpy.run_path(str(script), init_globals={"cq": cq, "math": math})
    if "result" not in scope:
        sys.exit("script must assign the final model to a variable named `result`")
    shape = cad.to_shape(scope["result"])
    stats = cad.shape_stats(shape)
    cad.export_step(shape, prefix.with_suffix(".step"))
    cad.export_stl(shape, prefix.with_suffix(".stl"))
    stats.update(ok=True, step=str(prefix.with_suffix(".step")), stl=str(prefix.with_suffix(".stl")))
    prefix.with_suffix(".json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
