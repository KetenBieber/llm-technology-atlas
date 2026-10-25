"""Mandatory coverage gate for newly changed papers, full backlog scan separately.

Default: review all registered articles and exit nonzero if any coverage risk remains.
--changed: review papers modified in HEAD compared to HEAD^; intended for CI
           so preexisting backlog does not block improvements to unrelated papers.
This is a heuristic risk gate, NOT a scientific review or original-PDF verification.
"""
from __future__ import annotations

import argparse
import re
import runpy
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
registry = runpy.run_path(str(ROOT / "tools" / "build_site_sources.py"))["ARTICLES"]
parser = argparse.ArgumentParser()
parser.add_argument("--changed", action="store_true", help="Only audited registered papers changed in latest Git commit")
args = parser.parse_args()

allowed = {item["path"] for item in registry}
if args.changed:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMRT", "HEAD^", "HEAD", "--", "papers"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    changed = set(result.stdout.splitlines()) & allowed
    print("SCOPE changed registered papers:", len(changed))
else:
    changed = allowed
    print("SCOPE all registered papers:", len(changed))

risk = []
for item in registry:
    if item["path"] not in changed:
        continue
    path = ROOT / item["path"]
    s = path.read_text(encoding="utf-8")
    stem = path.name.split("-", 1)[0]
    flags = []
    if len(s) < 2700:
        flags.append("正文篇幅较短，需要核实推导/实现完整性")
    if not re.search(r"Figure\s*\d|Table\s*\d|Algorithm\s*\d|图\s*\d|表\s*\d", s, re.I):
        flags.append("未定位原论文 Figure/Table/Algorithm，需人工核对")
    if s.count("$$") < 4:
        flags.append("行间公式不足两组，需要核查推导是否可复原")
    if not any(x in s for x in ["手算", "例子", "例如", "复算", "数值", "Shape", "shape", "假设"]):
        flags.append("缺少可复算例子或张量 Shape")
    if not any(x in s for x in ["不能", "局限", "边界", "代价", "开销", "限制"]):
        flags.append("缺少工程代价或失效边界")
    if flags:
        risk.append({"id": stem, "path": item["path"], "flags": flags})

print("REGISTERED_ARTICLES", len(registry), "AUDITED", len(changed), "FLAGGED", len(risk))
for item in risk:
    print(item["id"], item["path"], " | ".join(item["flags"]))
if risk:
    print("FAIL: coverage risk; original PDF review and independent scientific verification still required")
    raise SystemExit(2)
print("PASS heuristic coverage gate; not a certificate of scientific accuracy")
