/* Author-maintained read-only research atlas. No visitors' edits, cookies, or local storage. */
(() => {
  "use strict";
  const host = document.querySelector("#atlas-map");
  if (!host) return;
  const svg = host.querySelector("svg");
  const canvas = document.createElementNS("http://www.w3.org/2000/svg", "g");
  const edges = document.createElementNS("http://www.w3.org/2000/svg", "g");
  const nodesLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
  canvas.append(edges, nodesLayer);
  svg.append(canvas);
  const selector = host.querySelector("[data-map-view]");
  const search = host.querySelector("[data-map-search]");
  const label = host.querySelector("[data-map-status]");
  const detail = host.querySelector("[data-map-detail]");
  const fitButton = host.querySelector("[data-map-fit]");
  const expandButton = host.querySelector("[data-map-expand]");
  const shrinkButton = host.querySelector("[data-map-shrink]");
  const NS = "http://www.w3.org/2000/svg";
  const W = 208, H = 48, DX = 250, DY = 69, PAD = 30;
  let data, view = "evolution", collapsed = new Set(), viewport = {x: 22,y: 30,k: 1};
  let dragging = false, startPoint, startView, nodeIndex = [], sizes = {width: 920,height: 560};

  const el = (type, attrs = {}) => {
    const n = document.createElementNS(NS, type);
    Object.entries(attrs).forEach(([k,v])=> n.setAttribute(k, String(v)));
    return n;
  };
  const textNode = (t, x, y, cls) => {
    const n = el("text", {x, y, class: cls});
    n.textContent = t;
    return n;
  };
  function describe(node) {
    detail.replaceChildren();
    const head = document.createElement("strong");
    head.textContent = node.title;
    detail.append(head);
    const para = document.createElement("p");
    para.textContent = node.description || node.paper || "点击分类节点展开研究路线；点击论文节点打开全文。";
    detail.append(para);
    if (node.url) {
      const link = document.createElement("a");
      link.href = node.url;
      link.textContent = "阅读论文拆解 →";
      detail.append(link);
      if (node.source) {
        const ext = document.createElement("a");
        ext.href = node.source; ext.target = "_blank"; ext.rel = "noopener noreferrer";
        ext.textContent = "原论文 ↗";
        detail.append(ext);
      }
    }
  }
  function initCollapsed() {
    collapsed = new Set();
    data.views[view].groups.forEach(group => collapsed.add(view + ":" + group.id));
  }
  function makeTree() {
    const info = data.views[view];
    const term = search.value.trim().toLocaleLowerCase();
    const root = {id: "root",title: info.title,depth:0, children:[]};
    let hits = 0;
    info.groups.forEach(g=>{
      const group = {id:g.id,title:g.title,description:g.description,depth:1,children:[]};
      g.children.forEach(c=>{
        const branch = {id:g.id+"/"+c.id, title:c.title,depth:2,children:[]};
        c.papers.forEach(p=>{
          const match = !term || [p.id,p.title,p.paper].join(" ").toLocaleLowerCase().includes(term);
          if (match) {branch.children.push({...p,depth:3,children:[]}); if(term) hits++;}
        });
        if (branch.children.length) group.children.push(branch);
      });
      if (group.children.length) root.children.push(group);
    });
    const key = node => view + ":" + node.id;
    const visible = (node) => {
      const has = node.children && node.children.length;
      return { ...node, hiddenChildren: !!(has && !term && collapsed.has(key(node))),
        children: (!has || (!term && collapsed.has(key(node)))) ? [] : node.children.map(visible),
        count: node.children?.length || 0 };
    };
    label.textContent = term ? (hits ? hits + " 个匹配节点 · 点击论文跳转" : "没有找到匹配的论文") :
      data.totalArticles + " 篇论文 · 分类可展开 · 论文可跳转";
    return visible(root);
  }
  function layout(root) {
    let y = PAD;
    const all = [], lines = [];
    function visit(node, depth) {
      node.depth = depth;
      node.x = depth*DX+PAD;
      if (!node.children.length) {node.y = y; y += DY;}
      else {
        node.children.forEach(c=> visit(c,depth+1));
        node.y = (node.children[0].y + node.children[node.children.length-1].y)/2;
      }
      all.push(node);
      node.children.forEach(c=> lines.push({parent:node,child:c}));
    }
    visit(root,0);
    return {all,lines,width:Math.max(...all.map(n=>n.x))+W+PAD,height:y+PAD};
  }
  function updateTransform() {
    canvas.setAttribute("transform", "translate("+viewport.x+" "+viewport.y+") scale("+viewport.k+")");
  }
  function fit(ly) {
    sizes.width = svg.clientWidth || 920;
    sizes.height = svg.clientHeight || 560;
    viewport.k = Math.min(1.10,Math.max(.35,Math.min((sizes.width-40)/ly.width,(sizes.height-40)/ly.height)));
    viewport.x = Math.max(8,(sizes.width-ly.width*viewport.k)/2);
    viewport.y = Math.max(8,(sizes.height-ly.height*viewport.k)/2);
    updateTransform();
  }
  function draw({reframe=true}={}) {
    if (!data) return;
    const ly = layout(makeTree());
    nodeIndex = ly.all;
    edges.replaceChildren();
    nodesLayer.replaceChildren();
    for (const {parent,child} of ly.lines) {
      edges.append(el("path",{
        d:"M"+(parent.x+W)+","+(parent.y+H/2)+
          " C"+(parent.x+W+56)+","+(parent.y+H/2)+
          " "+(child.x-56)+","+(child.y+H/2)+
          " "+child.x+","+(child.y+H/2),
        class:"atlas-edge"
      }));
    }
    for(const n of ly.all) {
      const isPaper = !!n.url;
      const g = el("g",{class:"atlas-node"+(isPaper?" atlas-paper":" atlas-depth-"+n.depth),
        transform:"translate("+n.x+","+n.y+")",tabindex:0,role:"button","aria-label":n.title+
          (isPaper?" 打开文章":n.count?" 展开或折叠":"")});
      const rect = el("rect",{rx:10,ry:10,width:W,height:H});
      const short = n.title.length>16 ? n.title.slice(0,15)+"…" : n.title;
      const title = textNode(short,13,21,"atlas-node-title");
      g.append(rect,title);
      if(isPaper) g.append(textNode(n.id,13,38,"atlas-node-hint"));
      else g.append(textNode(n.depth===0?"知识导航":n.hiddenChildren?"＋ 展开 "+n.count+" 个分支":n.count?"－ 折叠":"",13,38,"atlas-node-hint"));
      const act=()=>{
        describe(n);
        if (isPaper) {window.location.assign(n.url);return;}
        if(n.count) {
          const k=view+":"+n.id;
          if(collapsed.has(k)) collapsed.delete(k); else collapsed.add(k);
          draw();
        }
      };
      g.addEventListener("click",e=>{e.stopPropagation();act();});
      g.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();act();}});
      g.addEventListener("mouseenter",()=>describe(n));
      const tooltip = el("title");tooltip.textContent=n.title;g.append(tooltip);
      nodesLayer.append(g);
    }
    if(reframe)fit(ly);
    else updateTransform();
  }
  function zoom(multiplier) {
    viewport.k = Math.max(.25,Math.min(2.1,viewport.k*multiplier));
    updateTransform();
  }
  function bind() {
    selector.addEventListener("change",()=>{
      view=selector.value;search.value="";initCollapsed();
      describe({title:data.views[view].title,description:data.views[view].intro});draw();
    });
    search.addEventListener("input",()=>draw());
    fitButton.addEventListener("click",()=>draw());
    expandButton.addEventListener("click",()=>zoom(1.22));
    shrinkButton.addEventListener("click",()=>zoom(1/1.22));
    svg.addEventListener("wheel",e=>{
      if(!e.ctrlKey){return;} // normal page scrolling works; Ctrl+wheel zooms atlas
      e.preventDefault();zoom(e.deltaY<0?1.08:1/1.08);
    },{passive:false});
    svg.addEventListener("pointerdown",e=>{
      if(e.target.closest(".atlas-node"))return;
      dragging=true;startPoint={x:e.clientX,y:e.clientY};startView={...viewport};
      svg.setPointerCapture(e.pointerId);
    });
    svg.addEventListener("pointermove",e=>{
      if(!dragging)return;
      viewport.x=startView.x+e.clientX-startPoint.x;
      viewport.y=startView.y+e.clientY-startPoint.y;
      updateTransform();
    });
    ["pointerup","pointercancel","lostpointercapture"].forEach(ev=>
      svg.addEventListener(ev,()=>{dragging=false;}));
    window.addEventListener("resize",()=>draw());
  }
  fetch("_static/atlas-map-data.json", {cache:"no-cache"})
    .then(r=>{if(!r.ok) throw Error("HTTP "+r.status); return r.json();})
    .then(json=>{data=json; initCollapsed();bind();
      describe({title:data.views[view].title,description:data.views[view].intro});
      draw();
    })
    .catch(()=>{label.textContent="导图暂时无法加载，仍可使用下方学习路线与专题目录。";});
})();
