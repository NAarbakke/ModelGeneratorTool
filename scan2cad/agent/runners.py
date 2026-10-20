"""Drive the MCP harness with a model: Claude Agent SDK or Codex CLI, same server, same task.

Both runners see only the scan2cad tools (built-in file/shell tools are off or
sandboxed read-only), so a comparison measures the model, not its access.
Each run is logged to <scan>/agent_runs/<runner>-<model>-<time>.json.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ..workspace import ScanWorkspace

TOOLS = ("scan_info", "segment_scan", "list_regions", "region", "fit_region", "build_cad",
         "score_cad", "render_cad", "show_code", "list_candidates", "refit_faces", "export_cad")
TOOL_TIMEOUT_S = 1800  # segmentation and silhouette scoring take minutes on CPU
DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
FINAL_RE = re.compile(r"FINAL:\s*([A-Za-z0-9_-]+)")

TASK = """\
Reverse-engineer the scanned part into a parametric CadQuery model using the scan2cad tools.
Follow the workflow in the server instructions. Build candidates under short names (v1, v2, ...).
When done, call export_cad on your best candidate and end your reply with a line
`FINAL: <candidate name>` followed by its key dimensions and any doubts."""


def server_command(scan: Path) -> tuple[str, list[str]]:
    return sys.executable, ["-m", "scan2cad.cli", "serve", str(scan.resolve())]


def run_claude(scan: Path, model: str = DEFAULT_CLAUDE_MODEL, max_turns: int = 60,
               task: str = TASK) -> dict:
    import asyncio

    from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock,
                                  ToolUseBlock, query)

    cmd, args = server_command(scan)
    options = ClaudeAgentOptions(
        model=model,
        tools=[],  # no built-in tools: the harness is the only way to touch the scan
        mcp_servers={"scan2cad": {"type": "stdio", "command": cmd, "args": args}},
        strict_mcp_config=True,
        allowed_tools=[f"mcp__scan2cad__{t}" for t in TOOLS],
        max_turns=max_turns,
        setting_sources=[],  # ignore user/project settings, CLAUDE.md, hooks
        env={"MCP_TOOL_TIMEOUT": str(TOOL_TIMEOUT_S * 1000)},
    )

    async def go() -> dict:
        transcript, final_text, result = [], "", None
        async for msg in query(prompt=task, options=options):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        transcript.append({"text": block.text})
                        final_text = block.text
                    elif isinstance(block, ToolUseBlock):
                        transcript.append({"tool": block.name, "input": _clip(block.input)})
            elif isinstance(msg, ResultMessage):
                result = msg
        return {"final_text": final_text, "transcript": transcript,
                "cost_usd": getattr(result, "total_cost_usd", None),
                "turns": getattr(result, "num_turns", None),
                "is_error": getattr(result, "is_error", None)}

    return _finish(scan, "claude", model, asyncio.run, go)


def run_codex(scan: Path, model: str | None = None, task: str = TASK) -> dict:
    cmd, args = server_command(scan)
    run_dir = ScanWorkspace(scan).root / "agent_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    last = run_dir / "codex_last_message.txt"
    argv = [
        "codex", "exec", "--json", "--ephemeral", "--skip-git-repo-check",
        "--sandbox", "read-only", "-C", str(scan.resolve()), "-o", str(last),
        "-c", f"mcp_servers.scan2cad.command={json.dumps(cmd)}",
        "-c", f"mcp_servers.scan2cad.args={json.dumps(args)}",
        "-c", f"mcp_servers.scan2cad.tool_timeout_sec={TOOL_TIMEOUT_S}",
        "-c", "mcp_servers.scan2cad.startup_timeout_sec=120",
    ]
    if model:
        argv += ["-m", model]
    argv.append(task)

    def go() -> dict:
        proc = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8")
        events = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
        transcript = [_codex_event(e) for e in events]
        return {"final_text": last.read_text(encoding="utf-8") if last.exists() else "",
                "transcript": [t for t in transcript if t],
                "exit_code": proc.returncode, "stderr_tail": proc.stderr[-2000:]}

    return _finish(scan, "codex", model or "default", lambda result: result, go)


def _finish(scan: Path, runner: str, model: str, execute, go) -> dict:
    from .server import Harness

    t0 = time.time()
    out = execute(go())
    out.update(runner=runner, model=model, seconds=round(time.time() - t0, 1),
               finished=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    match = FINAL_RE.findall(out.get("final_text", ""))
    out["final_candidate"] = match[-1] if match else None
    harness = Harness(scan)
    if out["final_candidate"]:
        try:
            out["final_score"] = harness.score(out["final_candidate"])["summary"]
        except FileNotFoundError:
            out["final_score"] = None
    out["candidates"] = harness.candidates()
    run_dir = ScanWorkspace(scan).root / "agent_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = run_dir / f"{runner}-{re.sub(r'[^A-Za-z0-9_.-]', '_', model)}-{stamp}.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    out["log"] = str(path)
    return out


def _clip(value, limit: int = 4000):
    s = json.dumps(value, default=str)
    return value if len(s) <= limit else s[:limit] + "...(clipped)"


def _codex_event(e: dict) -> dict | None:
    item = e.get("item") or {}
    kind = item.get("type") or item.get("item_type")
    if kind == "mcp_tool_call":
        return {"tool": item.get("tool"), "input": _clip(item.get("arguments"))}
    if kind in ("agent_message", "assistant_message"):
        return {"text": item.get("text")}
    return None
