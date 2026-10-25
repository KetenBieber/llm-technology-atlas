# LLM Technology Atlas

## 论文拆解 Skill

仓库内置可复用的 Codex/Agent Skill：`.agents/skills/paper-atlas-deep-dive/SKILL.md`。
它把 B 类完整模型、A 类原始方法与 C 类工程系统论文分别规范，提供第一性推导、原图证据、自制解释图、跨论文跳转与静态审查。Skill 的附属资料位于同目录的 `references/`、`assets/` 与 `scripts/`；新会话可要求“使用 paper-atlas-deep-dive 拆解/重构某篇论文”。当前 Skill 已在仓库创建；是否被其他客户端自动发现，取决于其 Skills 发现与加载机制。


中文大模型技术文章站。公开内容只包含技术文章、技术导航与必要图示；论文拆解规范、内部审阅记录、原始源包和构建中间状态不会进入网站。

## 本地预览

### 首页只读交互导图

首页提供两个可切换的论文关系视角：**技术发展路线**与**技术难点 → 解决方案**。访客只能展开、搜索、缩放和进入文章，不具备节点编辑或写回仓库的功能。

- 作者在 `tools/atlas_map_topics.json` 维护分支归类与跨主题关联；同一论文可出现在不同的问题视角。
- `tools/build_site_sources.py` 读取已经登记的论文列表，并在每次站点构建时调用 `tools/build_atlas_map.py` 自动生成 `docs/_static/atlas-map-data.json`，文章标题与地址不需要在 JSON 重复填写。
- 今后新增论文只需注册到 `ARTICLES` 并按需修订分类；如果尚未分类，页面会自动显示“待归类新论文”，防止新文章从地图消失。
- 访问端读取静态 JSON，导图代码是 `docs/_static/atlas-map.js`，样式 `docs/_static/atlas-map.css`；不存储访问者修改。

```powershell
python tools/build_site_sources.py
python -m sphinx -b html -W --keep-going docs site
python tools/check_public_hygiene.py docs/generated
python tools/check_static_links.py site
```

生成后打开 `site/index.html`。

公开站点按“训练基础 → Transformer → MoE → 完整模型 → Reasoning Post-training”组织，并用 DeepSeek 与 Llama 3 / Qwen2.5 两组现代模型路线建立架构、数据、训练系统与后训练的横向对照；文章内部保留前置/后续/相关技术跳转。
