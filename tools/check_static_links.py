from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

FORBIDDEN_ARTIFACT_PARTS = {
    ".internal",
    "figure_manifests",
    "quality_audit",
    "rewrite_execution_spec",
    "reading_standard",
    "v2_status",
    "paperlist",
}

class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        attr = "href" if tag in {"a", "link"} else "src" if tag in {"img", "script"} else None
        if attr is None:
            return
        value = dict(attrs).get(attr)
        if value:
            self.targets.append(value)

def local_target(source: Path, raw: str) -> Path | None:
    parsed = urlsplit(raw)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    target = (source.parent / unquote(parsed.path)).resolve()
    if raw.endswith("/"):
        target /= "index.html"
    return target

def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "site").resolve()
    if not root.is_dir():
        print(f"site directory does not exist: {root}", file=sys.stderr)
        return 2

    html_files = sorted(root.rglob("*.html"))
    missing = []
    forbidden = []
    checked = 0

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        normalized = path.relative_to(root).as_posix().lower()
        if any(part in normalized for part in FORBIDDEN_ARTIFACT_PARTS):
            forbidden.append(path.relative_to(root))

    for source in html_files:
        parser = LinkParser()
        parser.feed(source.read_text(encoding="utf-8"))
        for raw in parser.targets:
            target = local_target(source, raw)
            if target is None:
                continue
            checked += 1
            if not target.exists():
                missing.append((source.relative_to(root), raw))

    print(f"HTML_COUNT={len(html_files)}")
    print(f"LOCAL_REFERENCES={checked}")
    print(f"LOCAL_MISSING={len(missing)}")
    print(f"FORBIDDEN_ARTIFACTS={len(forbidden)}")
    for src, raw in missing[:100]:
        print(f"MISSING {src}: {raw}")
    for path in forbidden[:100]:
        print(f"FORBIDDEN {path}")
    return 1 if missing or forbidden else 0

if __name__ == "__main__":
    raise SystemExit(main())
