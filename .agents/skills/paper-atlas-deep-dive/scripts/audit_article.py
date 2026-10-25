"""Read-only local article checker; not a scientific/visual review."""
from pathlib import Path
import re, sys, xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
LINK = re.compile(r"!?(?:\[[^\]]*\])\(([^)]+)\)")

def check(path):
    errors = []
    if not path.is_file(): return ["File not found"]
    try: path.relative_to(ROOT)
    except ValueError: return ["File outside repository"]
    text = path.read_text(encoding="utf-8")
    fence = None; math = False; h1 = 0; links = 0; display_blocks = 0
    for no, line in enumerate(text.splitlines(), 1):
        m = re.match(r"^\s*(`{3,}|~{3,})", line)
        if m:
            c = m.group(1)[0]
            fence = c if fence is None else None if fence == c else fence
            continue
        if fence: continue
        if line.startswith("# "): h1 += 1
        if line.strip() == "$$":
            if not math: display_blocks += 1
            math = not math
            continue
        elif line.strip() == "$":
            errors.append(f"L{no}: standalone single $ cannot delimit display math; use $$")
            continue
        if math: continue
        # A LaTeX tensor index can look like a Markdown link: E[g_t^2](1-\\beta).
        # Remove inline math before resolving Markdown links, not after.
        check_line = re.sub(r"(?<!\\)\$(?!\$).*?(?<!\\)\$(?!\$)", "", line)
        for m in LINK.finditer(check_line):
            url = m.group(1).split("#",1)[0].split("?",1)[0]
            if not url or url.startswith(("https:","http:","mailto:","data:")): continue
            links += 1; dest = (path.parent / url).resolve()
            try: dest.relative_to(ROOT)
            except ValueError: errors.append(f"L{no}: link escapes repository: {url}"); continue
            if not dest.is_file(): errors.append(f"L{no}: missing target: {url}")
            elif dest.suffix.lower() == ".svg":
                try: ET.parse(dest)
                except (ET.ParseError,OSError) as ex: errors.append(f"L{no}: invalid SVG {url}: {ex}")
    if math: errors.append("unclosed $$ block")
    if fence: errors.append("unclosed fenced block")
    if h1 != 1: errors.append(f"expected one H1, got {h1}")
    top = "\n".join(text.splitlines()[:48])
    if not re.search(r"https?://\S+", top):
        errors.append("missing original/official source URL in article introduction")
    if display_blocks < 1:
        errors.append("no display-math derivation; check technical depth (unless paper genuinely has no math)")
    print(f"{'FAIL' if errors else 'PASS'} {path.relative_to(ROOT)}: {h1} H1, {links} local links")
    for e in errors: print("  "+e)
    return errors

if __name__ == "__main__":
    if len(sys.argv) < 2: raise SystemExit("Usage: audit_article.py papers/path.md [...]")
    found = sum(bool(check((ROOT / arg).resolve())) for arg in sys.argv[1:])
    print("Static link/structure check only; scientific claims and visual rendering not certified.")
    raise SystemExit(1 if found else 0)
