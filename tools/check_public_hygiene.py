from __future__ import annotations

from pathlib import Path
import re
import sys

FORBIDDEN = [
    r"V2 重写",
    r"重制版 A\d+",
    r"figure_manifests/",
    r"REWRITE_EXECUTION_SPEC",
    r"READING_STANDARD",
    r"QUALITY_AUDIT",
    r"V2_STATUS",
    r"本项目研究批注",
    r"公开发布前须复核",
    r"当前项目真实进度",
    r"Agent 工作",
    r"审计状态",
    r"\bPASS\b.*终审",
]

def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/generated")
    if not root.exists():
        print(f"missing generated docs: {root}", file=sys.stderr)
        return 2
    issues = []
    patterns = [(raw, re.compile(raw, re.I)) for raw in FORBIDDEN]
    for path in root.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for raw, pattern in patterns:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                issues.append((path, line, raw))
    if issues:
        print(f"PUBLIC_HYGIENE_FAIL count={len(issues)}")
        for path, line, raw in issues[:100]:
            print(f"{path}:{line}: {raw}")
        return 1
    print("PUBLIC_HYGIENE_PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
