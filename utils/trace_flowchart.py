"""Build Graphviz flowcharts from runtime trace steps."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

from graph.trace import trace_step_title

_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_]")


def _safe_node_id(name: str) -> str:
    normalized = _SAFE_ID_RE.sub("_", name.strip().lower())
    return normalized or "unknown"


def _node_label(name: str, count: int) -> str:
    display = trace_step_title(name)
    if count <= 1:
        return display
    return f"{display}\\n({count}x)"


def build_trace_flowchart_dot(trace: list[dict[str, Any]]) -> str:
    """Return DOT graph representing the exact executed trace transitions."""
    lines = [
        "digraph PipelineTrace {",
        'rankdir="LR";',
        'labelloc="t";',
        'label="Runtime Flow";',
        'bgcolor="transparent";',
        'node [shape=box, style="rounded,filled", fillcolor="#1f2937", color="#4b5563", fontcolor="#f9fafb"];',
        'edge [color="#9ca3af", fontcolor="#d1d5db"];',
    ]
    if not trace:
        lines.append('empty [label="No trace yet", shape=note, fillcolor="#111827"];')
        lines.append("}")
        return "\n".join(lines)

    nodes = [str(step.get("node") or "unknown").strip() or "unknown" for step in trace]
    node_counts = Counter(nodes)
    edge_counts = Counter(zip(nodes, nodes[1:]))

    for node_name, count in node_counts.items():
        node_id = _safe_node_id(node_name)
        lines.append(f'{node_id} [label="{_node_label(node_name, count)}"];')

    for (source, target), count in edge_counts.items():
        src_id = _safe_node_id(source)
        dst_id = _safe_node_id(target)
        if count > 1:
            lines.append(f'{src_id} -> {dst_id} [label="{count}"];')
        else:
            lines.append(f"{src_id} -> {dst_id};")

    first_id = _safe_node_id(nodes[0])
    lines.append(f'{first_id} [fillcolor="#065f46", color="#10b981"];')
    if len(nodes) > 1:
        last_id = _safe_node_id(nodes[-1])
        lines.append(f'{last_id} [fillcolor="#1e3a8a", color="#60a5fa"];')

    lines.append("}")
    return "\n".join(lines)
