from pathlib import Path
import re
import sys

BT = chr(96)
CODE_FENCE_RE = re.compile(
    r"(?s)(?:" + re.escape(BT * 3) + r"|~~~).*?(?:" + re.escape(BT * 3) + r"|~~~)"
)
INLINE_CODE_RE = re.compile(re.escape(BT) + r"[^" + re.escape(BT) + r"\n]*" + re.escape(BT))
MATH_HTML_RE = re.compile(
    r'<(?:div|span)[^>]*class="[^"]*\bmath\b[^"]*"[^>]*>.*?</(?:div|span)>',
    re.S | re.I,
)

def strip_code(text: str) -> str:
    return INLINE_CODE_RE.sub("", CODE_FENCE_RE.sub("", text))

def check_sources(root: Path):
    findings = []
    for path in sorted(root.rglob("*.md")):
        text = strip_code(path.read_text(encoding="utf-8", errors="replace"))
        for i, line in enumerate(text.splitlines(), 1):
            if re.search(r"(?<!\\)\\\(|(?<!\\)\\\)", line):
                findings.append((path, i, "legacy-inline-delimiter", line.strip()))
            if re.match(r"^\s*(?:>\s*)?\\[\[\]]\s*$", line):
                findings.append((path, i, "legacy-display-delimiter", line.strip()))
            if re.match(r"^\s+\$\$\s*$", line):
                findings.append((path, i, "indented-dollar-display", line.strip()))
            if re.match(r"^>\s*\$\$\s*$", line):
                findings.append((path, i, "blockquote-dollar-display", line.strip()))
            if "$$" in line and line.strip() != "$$" and not re.match(r"^\s*(?:::|#|>|[-*+]|\d+[.)])", line):
                findings.append((path, i, "inline-double-dollar", line.strip()))
    return findings

def check_html(root: Path):
    findings = []
    for path in sorted(root.rglob("*.html")):
        rel = path.relative_to(root).as_posix()
        if not rel.startswith("generated/papers/"):
            continue
        html = path.read_text(encoding="utf-8", errors="replace")
        body = MATH_HTML_RE.sub("", html)
        body = re.sub(r"<script\b.*?</script>", "", body, flags=re.S | re.I)
        body = re.sub(r"<style\b.*?</style>", "", body, flags=re.S | re.I)
        suspects = [
            ("literal-double-dollar", r"\$\$"),
            ("literal-inline-open", r"\\\("),
            ("literal-display-open", r"\\\["),
            ("literal-frac", r"\\frac"),
            ("literal-begin", r"\\begin\{"),
            ("literal-boxed", r"\\boxed"),
        ]
        for kind, pattern in suspects:
            m = re.search(pattern, body)
            if m:
                snippet = re.sub(r"\s+", " ", body[max(0, m.start()-60):m.start()+160])
                findings.append((path, 0, kind, snippet[:240]))
    return findings

def main():
    source_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/generated")
    html_root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("site")

    source_findings = check_sources(source_root)
    html_findings = check_html(html_root)
    findings = source_findings + html_findings

    if findings:
        print(f"MATH_RENDER_CHECK_FAIL count={len(findings)}")
        for path, line, kind, sample in findings[:200]:
            loc = f"{path}:{line}" if line else str(path)
            print(f"{kind} | {loc} | {sample}")
        raise SystemExit(1)

    print("MATH_RENDER_CHECK_PASS")
    print(f"SOURCE_ROOT={source_root}")
    print(f"HTML_ROOT={html_root}")

if __name__ == "__main__":
    main()
