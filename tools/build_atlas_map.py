"""根据论文注册表生成只读、可搜索、可折叠的首页技术关系导图数据。"""
from __future__ import annotations

import json
import re
from pathlib import Path


def build_atlas_map(articles: list[dict], root: Path) -> Path:
    spec_path = root / "tools" / "atlas_map_topics.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    indexed = {}
    for item in articles:
        match = re.match(r"([ABC]\d{3})-", Path(item["path"]).name)
        if not match:
            raise ValueError(f"无法取得论文 ID：{item['path']}")
        uid = match.group(1)
        if uid in indexed:
            raise ValueError(f"重复论文 ID：{uid}")
        indexed[uid] = {
            "id": uid,
            "title": item["title"],
            "paper": item["paper"],
            "url": "generated/" + item["path"].removesuffix(".md") + ".html",
            "source": item["url"],
        }
    graph = {"version": spec["version"], "totalArticles": len(indexed), "views": {}}
    for view_id, view in spec["views"].items():
        seen = set()
        groups = []
        for group in view["groups"]:
            branches = []
            for branch in group["children"]:
                papers = []
                for uid in branch["papers"]:
                    if uid not in indexed:
                        raise ValueError(f"{view_id}/{branch['id']} 引用了未登记的论文 {uid}")
                    papers.append(indexed[uid])
                    seen.add(uid)
                if papers:
                    branches.append({"id": branch["id"], "title": branch["title"], "papers": papers})
            if branches:
                groups.append({"id": group["id"], "title": group["title"],
                               "description": group.get("description", ""), "children": branches})
        missing = [indexed[k] for k in sorted(indexed) if k not in seen]
        if missing:
            groups.append({"id": "uncategorized", "title": "待归类新论文",
                           "description": "新增论文已自动进入导图；可通过作者维护的 taxonomy 归入更精确的技术路线。",
                           "children": [{"id": "new", "title": "最近登记的研究", "papers": missing}]})
        graph["views"][view_id] = {"title": view["title"], "intro": view["intro"],
                                   "groups": groups, "uniquePapers": len(seen)}
    output = root / "docs" / "_static" / "atlas-map-data.json"
    output.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    from build_site_sources import ARTICLES, ROOT
    print(build_atlas_map(ARTICLES, ROOT))
