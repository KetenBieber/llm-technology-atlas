# LLM Technology Atlas

中文大模型技术文章站。公开内容只包含技术文章、技术导航与必要图示；论文拆解规范、内部审阅记录、原始源包和构建中间状态不会进入网站。

## 本地预览

```powershell
python tools/build_site_sources.py
python -m sphinx -b html -W --keep-going docs site
python tools/check_public_hygiene.py docs/generated
python tools/check_static_links.py site
```

生成后打开 `site/index.html`。

公开站点按“训练基础 → Transformer → MoE → 完整模型 → Reasoning Post-training”组织，并用 DeepSeek 与 Llama 3 / Qwen2.5 两组现代模型路线建立架构、数据、训练系统与后训练的横向对照；文章内部保留前置/后续/相关技术跳转。
