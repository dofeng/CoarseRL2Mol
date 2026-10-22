"use strict";

function createPathwayScene(data, mode, summary) {
  const area=byId("graphArea"),three=mode==="path3d" || mode==="pathdetail3d";
  const metric=byId("pathTimeMetric").value || "compact";
  state.pathwayLayouts ||= {};
  const key=`${mode}:${metric}`;
  const layout=state.pathwayLayouts[key] || (state.pathwayLayouts[key]=compactProcessLayout(data,{metric}));
  const width=three ? Math.max(400,area.clientWidth) : layout.width;
  const height=three ? Math.max(300,area.clientHeight-8) : layout.height;
  const changed=state.pathScene?.key!==key || state.pathScene?.raw!==state.pathway;
  state.pathScene={key,raw:state.pathway,data,layout,three,width,height};
  state.pathSuppressClick=false;
  state.graphHeight=height;state.pathwayWidth=width;
  if(changed) {
    state.pathwayRotation={yaw:0,pitch:.16};state.pathwayPan={x:0,y:0};
    state.pathwayScale=three ? Math.min(1,(height-140)/layout.height) : 1;
    if(three)state.pathwayPan.x=(layout.width/2-110)*state.pathwayScale-width/2+110;
    area.scrollLeft=0;area.scrollTop=0;
  }
  const nodes=layout.nodes.map(node=>{
    const event=node.kind==="event";
    const label=event ? {name:node.process ? `过程 ${node.event_id} · ${node.event_count} 次` : `E${node.event_id}@${node.frame}`,description:`帧 ${node.frame} → ${node.child_frame}`,color:"#344f66"} : speciesPresentation(node.species);
    if(node.activity_events)label.name+=` · 类内 ${node.activity_events.length} 次`;
    const radius=event ? 7 : 8+8*Math.sqrt(node.tracked/data.seed_atoms);
    node.radius=node.process ? 14 : radius;
    const shape=event ? `<rect width="${node.process?26:14}" height="18" rx="5" fill="${label.color}"/>` : `<circle r="${radius}" fill="${node.context ? '#e2e7ed' : label.color}" stroke="white" stroke-width="2"/>`;
    return `<g class="path-node" tabindex="0" role="button" data-path-node="${node.id}" aria-label="${escapeText(label.name+' '+label.description)}"><line class="path-residence" stroke="${label.color}" stroke-width="4" opacity=".38"/>${shape}<g class="path-node-label"><text class="path-label">${escapeText(label.name)}</text><text y="15" class="path-subtitle">${escapeText(label.description)}</text></g><title>${escapeText(label.name+' · '+label.description)}${event ? '' : ` · 帧 ${node.first_frame}–${node.last_frame} · ${node.tracked} 个追踪原子`}</title></g>`;
  }).join("");
  const links=data.links.map((link,index)=>{
    const main=link.role.startsWith("dominant");
    return `<path class="path-curve" data-path-link="${index}" stroke="${main ? '#ba5220' : link.tracked ? '#46789c' : '#929ba8'}" stroke-width="${main ? 2.8 : 1.3}" ${!link.tracked ? 'stroke-dasharray="5 4"' : ''} fill="none" marker-end="url(#pathArrow)"><title>${link.tracked} 个锚点原子 · ${link.process ? '阶段贡献，非直接反应' : main ? '局部主要延续分支' : '参与分支'}</title></path>`;
  }).join("");
  const ticks=layout.ticks.map(tick=>`<g class="path-time-tick"><line y1="6" y2="14" stroke="#9badbc"/><text class="path-subtitle">帧 ${tick.time}</text></g>`).join("");
  area.innerHTML=`<svg id="networkSvg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="从左到右的时间反应网络"><defs><marker id="pathArrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="#516478"/></marker></defs><rect width="${width}" height="${height}" fill="#f7f9fc"/><g class="path-edges">${links}</g><g class="path-nodes">${nodes}</g><g class="path-time-axis">${ticks}</g><text x="20" y="25" class="path-label">时间 → · ${metric==='frames' ? '真实帧距' : '紧凑刻度（非等比例时间）'} · ${summary ? '阶段块不是单步反应' : '细线段表示驻留区间'}</text></svg>`;
  const svg=byId("networkSvg");
  const byIdentifier=new Map(layout.nodes.map(node=>[node.id,node]));
  state.pathScene.svg=svg;
  state.pathScene.nodes=[...svg.querySelectorAll('[data-path-node]')].map(element=>{
    const node=byIdentifier.get(element.dataset.pathNode);
    const inspect=()=>{if(state.pathSuppressClick)return;state.pathwaySelected=node.id;inspectPathNode(node);updatePathwayProjection();};
    element.onclick=inspect;element.onkeydown=event=>{if(event.key==='Enter')inspect();};
    return {node,element,shape:element.querySelector('circle,rect'),residence:element.querySelector('.path-residence'),label:element.querySelector('.path-node-label')};
  });
  state.pathScene.links=[...svg.querySelectorAll('[data-path-link]')].map(element=>{
    const link=data.links[+element.dataset.pathLink];
    return {element,source:byIdentifier.get(link.source),target:byIdentifier.get(link.target)};
  });
  state.pathScene.ticks=[...svg.querySelectorAll('.path-time-tick')].map((element,index)=>({element,node:layout.ticks[index]}));
  byId('pathBrowseControls').hidden=!three;
  bindPathwayCamera(three,width,height);
  updatePathwayProjection();
  if(!state.pathResizeObserver && typeof ResizeObserver!=='undefined') {
    state.pathResizeObserver=new ResizeObserver(()=>{
      const scene=state.pathScene;
      if(!scene?.three || !scene.svg.isConnected)return;
      const width=Math.max(400,area.clientWidth),height=Math.max(300,area.clientHeight-8);
      if(width===scene.width && height===scene.height)return;
      scene.width=width;scene.height=height;state.pathwayWidth=width;state.graphHeight=height;
      scene.svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
      const background=scene.svg.querySelector(':scope > rect');
      background.setAttribute('width',width);background.setAttribute('height',height);
      state.pathDrag=null;bindPathwayCamera(true,width,height);schedulePathwayProjection();
    });
    state.pathResizeObserver.observe(area);
  }
}

function updatePathwayProjection(moving=false) {
  const scene=state.pathScene;
  if(!scene || !scene.svg.isConnected || !byId('representation').value.startsWith('path'))return;
  const {layout,width,height,three}=scene;
  const scale=state.pathwayScale||1,pan=state.pathwayPan||{x:0,y:0};
  const shapeScale=three ? Math.max(.2,Math.min(1.3,Math.sqrt(scale))) : 1;
  const depth=+(byId('pathDepth').value||100)/100;
  const project=node=>three ? spatialProjection({...node,z:node.z*depth},layout,width,height,state.pathwayRotation,scale,pan) : {sx:node.x,sy:node.y,depth:0};
  scene.project=project;
  sizeGraph(scene.svg,three ? 1 : scale);
  byId('graphScale').value=Math.max(5,scale*100);
  byId('pathScaleValue').textContent=`${(scale*100).toFixed(1)}%`;
  const positioned=scene.nodes.map(entry=>({...entry,point:project(entry.node)}));
  for(const {node,shape,residence,point} of positioned) {
    if(node.kind==='event'){
      shape.setAttribute('x',point.sx-(node.process?13:7)*shapeScale);shape.setAttribute('y',point.sy-9*shapeScale);
      shape.setAttribute('width',(node.process?26:14)*shapeScale);shape.setAttribute('height',18*shapeScale);
    } else {shape.setAttribute('cx',point.sx);shape.setAttribute('cy',point.sy);shape.setAttribute('r',node.radius*shapeScale);shape.setAttribute('stroke',state.pathwaySelected===node.id ? '#102d4e' : 'white');}
    const end=project({...node,x:node.endX});
    residence.setAttribute('x1',point.sx);residence.setAttribute('y1',point.sy);
    residence.setAttribute('x2',end.sx);residence.setAttribute('y2',end.sy);
  }
  for(const {element,source,target} of scene.links)if(source && target)element.setAttribute('d',spatialCurve(source,{...target,radius:target.radius*shapeScale},project));
  const occupied=[];
  positioned.sort((left,right)=>Number(right.node.id===state.pathwaySelected)-Number(left.node.id===state.pathwaySelected) || (right.node.tracked||0)-(left.node.tracked||0));
  for(const {node,label,point} of positioned) {
    label.style.display='none';
    if(moving || (scale<.3 && node.id!==state.pathwaySelected))continue;
    const labelWidth=155,labelHeight=32;
    for(const [offsetX,offsetY] of [[20,-28],[20,30],[-175,-28],[-175,30]]) {
      const box={x:point.sx+offsetX,y:point.sy+offsetY};
      if(box.x<5 || box.y<45 || box.x+labelWidth>width || box.y+labelHeight>height)continue;
      if(occupied.some(other=>Math.abs(box.x-other.x)<labelWidth && Math.abs(box.y-other.y)<labelHeight))continue;
      if(positioned.some(other=>other.node.id!==node.id && other.point.sx>box.x-12 && other.point.sx<box.x+labelWidth+12 && other.point.sy>box.y-12 && other.point.sy<box.y+labelHeight))continue;
      label.setAttribute('transform',`translate(${box.x} ${box.y+10})`);label.style.display='';occupied.push(box);break;
    }
  }
  if(!moving && three)for(const entry of positioned.sort((left,right)=>left.point.depth-right.point.depth))entry.element.parentNode.appendChild(entry.element);
  let previousTick=-Infinity;
  for(const {element,node} of scene.ticks) {
    const point=project(node);
    const visible=point.sx>40 && point.sx<width-70 && (point.sx-previousTick)*(three ? 1 : scale)>95;
    element.style.display=visible ? '' : 'none';
    if(visible){element.setAttribute('transform',`translate(${point.sx} 55)`);previousTick=point.sx;}
  }
  if(three) {
    const center=project({x:layout.width/2,y:layout.height/2,z:0});
    const span=Math.max(1,layout.width*Math.cos(state.pathwayRotation.yaw)*scale);
    byId('pathBrowse').value=Math.max(0,Math.min(1000,500+(width/2-center.sx)/span*1000));
  }
}

function fitPathway() {
  const scene=state.pathScene;if(!scene)return;
  state.pathwayRotation={yaw:0,pitch:.16};state.pathwayPan={x:0,y:0};
  const area=byId('graphArea');
  if(scene.three) {
    const points=scene.layout.nodes.flatMap(node=>[node,{...node,x:node.endX}]).map(node=>spatialProjection({...node,z:node.z*+(byId('pathDepth').value||100)/100},scene.layout,scene.width,scene.height,state.pathwayRotation,1,{x:0,y:0}));
    const minimumX=Math.min(...points.map(point=>point.sx)),maximumX=Math.max(...points.map(point=>point.sx));
    const minimumY=Math.min(...points.map(point=>point.sy)),maximumY=Math.max(...points.map(point=>point.sy));
    const scale=Math.min(1,(scene.width-130)/Math.max(1,maximumX-minimumX),(scene.height-160)/Math.max(1,maximumY-minimumY));
    state.pathwayScale=scale;
    state.pathwayPan={x:(scene.width/2-(minimumX+maximumX)/2)*scale,y:(scene.height/2-(minimumY+maximumY)/2)*scale};
  } else state.pathwayScale=Math.min(1,(area.clientWidth-20)/scene.layout.width,(area.clientHeight-20)/scene.layout.height);
  area.scrollLeft=0;area.scrollTop=0;updatePathwayProjection();
}

function focusPathway() {
  const scene=state.pathScene,node=scene?.nodes.find(entry=>entry.node.id===state.pathwaySelected)?.node;
  if(!node){notify('请先点击一个节点');return;}
  state.pathwayScale=Math.max(.7,state.pathwayScale||1);
  if(scene.three) {
    state.pathwayPan={x:0,y:0};updatePathwayProjection();
    const point=scene.project(node);state.pathwayPan={x:scene.width/2-point.sx,y:scene.height/2-point.sy};
    updatePathwayProjection();
  } else {
    updatePathwayProjection();const area=byId('graphArea');
    area.scrollLeft=node.x*state.pathwayScale-area.clientWidth/2;area.scrollTop=node.y*state.pathwayScale-area.clientHeight/2;
  }
}
