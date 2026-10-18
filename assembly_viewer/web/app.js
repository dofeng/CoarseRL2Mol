/* Durable ordered playback, one SVG, real coordinates, bounded online caches. */
(() => {
  "use strict";
  const $ = id => document.getElementById(id), NS = "http://www.w3.org/2000/svg";
  const phaseNames = {input:"输入资源",rigid:"刚性连接",flex:"柔性连接",ring:"脂肪环",side:"侧链",substitution:"SU26 / SU30 取代"};
  const colors = {input:"#81929c",rigid:"#5677a5",aromatic:"#5677a5",flex:"#32877d",ring:"#b28b40",side:"#b86660",substitution:"#8563a1"};
  const suColors = {5:"#cc4d99",6:"#994dcc",7:"#b6a313",8:"#808080",9:"#b31a1a",10:"#3366e6",11:"#2eab60",12:"#e63333",13:"#4199c5",19:"#e66666",20:"#b380e6",21:"#999999",22:"#cc801a",23:"#cf8b30",24:"#cf8b30",25:"#cf8b30",26:"#4d00e6",27:"#8000cc",28:"#ff3333",29:"#cc0000",30:"#bd9b0b"};
  const kindNames = {"input.ready":"输入资源已读取","formal.initialized":"装配初始化","stage.start":"阶段开始","stage.complete":"阶段完成","search.start":"开始搜索","search.visit":"实际试连接","search.result":"本轮结果候选","search.selected":"选中候选，等待确认","checkpoint.rollback":"回退到检查点","checkpoint.assessment":"前瞻判断","checkpoint.accepted":"检查点接受","formal.connection":"已确认连接","forward.start":"正在快速前瞻","forward.complete":"快速前瞻完成","forward.connection":"前瞻试连接","run.finished":"运行结束","run.error":"运行异常"};
  const embedded = $("embedded-trace") ? JSON.parse($("embedded-trace").textContent) : null;
  let context={}, defaults={}, run=null, manifest={}, catalog=null, phases=[], seq=0, latest=0;
  let queue=[], cache=new Map(), shown=null, current=null, playing=false, timer=null, stream=null, stopAt=null, generation=0, fetching=false;
  let view=null, bounds=null, drag=null, mode="live", seeking=false, selectedGid=null, selectedPhase=null;
  const message = text => $("message").textContent=text;
  async function api(path, data){
    const response=await fetch(path,data===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)});
    const value=await response.json(); if(!response.ok)throw Error(value.error||response.statusText); return value;
  }
  function traceUrl(endpoint,params={}){const query=new URLSearchParams(Object.entries({...params,...(run?{run}: {})}).filter(([,v])=>v!==undefined&&v!==null));return "/api/trace/"+endpoint+"?"+query;}
  async function trace(endpoint,params={}){
    if(!embedded)return api(traceUrl(endpoint,params));
    if(endpoint==="manifest")return {...embedded.manifest,latest_sequence:embedded.events.at(-1)?.seq||0};
    if(endpoint==="resources")return embedded.resources;
    if(endpoint==="phases")return embedded.phases;
    if(endpoint.startsWith("state/"))return embedded.states[endpoint.split("/")[1]];
    const kinds=embedded.timeline_kinds[params.mode||mode];
    if(endpoint==="events")return {phases:embedded.phases,events:embedded.events.filter(e=>e.seq>Number(params.after||0)&&(params.before==null||e.seq<=Number(params.before))&&kinds.includes(e.kind)).slice(0,Number(params.limit||250)),latest_sequence:embedded.events.at(-1)?.seq||0};
    if(endpoint==="position")return embedded.events.filter(e=>e.seq<=Number(params.at)&&kinds.includes(e.kind)).at(-1)||null;
    throw Error("未知离线接口");
  }
  async function state(id){
    if(id==null)return null;
    if(cache.has(id)){const value=cache.get(id);cache.delete(id);cache.set(id,value);return value;}
    const value=await trace("state/"+id);if(!value)throw Error("状态记录不存在: "+id);
    cache.set(id,value);if(cache.size>12)cache.delete(cache.keys().next().value);return value;
  }
  function svgElement(tag,attrs={},text){const n=document.createElementNS(NS,tag);for(const [k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;}
  const xy=([q,r])=>[Math.sqrt(3)/2*q*36,-(r-q/2)*36];
  const nodeRadius=8.5, labelSize=.46*2*nodeRadius;
  function applyView(){if(view)$("molecule-svg").setAttribute("viewBox",view.join(" "));}
  function fit(){if(bounds){view=[...bounds];applyView();}}
  function detail(node,preview=false){$("node-detail").textContent=JSON.stringify({GID:node.logical_gid??node.node_id,unit:node.unit_id,original_SU:node.original_su??node.original_su_type,planned_SU:node.planned_su??node.planned_su_type,assembly_target_SU:node.assembly_target_su??node.assembly_target_su_type,current_SU:preview?"未组装":node.su_type,label_state:node.residual_conversion?"residual_converted":node.label_state,local_axial:preview?node.local_axial:undefined,actual_position:preview?undefined:node.final_position,physical_slot:node.physical_slot_id,fixed_center_parent:node.su25_center_parent_id},null,2);}
  function drawGraph(nodes,edges,old=null,titles=[],hull=[],focusIds=[]){
    const svg=$("molecule-svg");svg.replaceChildren();let xs=[],ys=[];
    const positions={};for(const [gid,n]of Object.entries(nodes)){positions[gid]=xy(n.final_position);xs.push(positions[gid][0]);ys.push(positions[gid][1]);}
    for(const t of titles){const p=xy(t.position);svg.append(svgElement("text",{x:p[0],y:p[1],fill:colors[t.phase]||"#52667a","font-size":13,"font-weight":600},t.label));xs.push(p[0]);ys.push(p[1]);}
    if($("show-hull").checked&&hull.length>2)svg.append(svgElement("polygon",{points:hull.map(([x,y])=>[x*36,-y*36]).map(p=>p.join(",")).join(" "),fill:"#167d7608",stroke:"#167d76","stroke-dasharray":"5 4"}));
    for(const [key,e]of Object.entries(edges)){const a=positions[e.source_node_id],b=positions[e.target_node_id];if(!a||!b)continue;svg.append(svgElement("line",{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:old&&!old.edges[key]?"#129773":"#899aa8","stroke-width":old&&!old.edges[key]?3:1.8}));}
    for(const [gid,n]of Object.entries(nodes)){const p=positions[gid],changed=old&&JSON.stringify(old.nodes[gid])!==JSON.stringify(n);const g=svgElement("g",{class:"node","data-gid":n.logical_gid??n.node_id,transform:`translate(${p[0]} ${p[1]})`});const special=n.su25_center_parent_id!==undefined&&n.su25_center_parent_id!==null;
      const fill=suColors[n.su_type]||colors[n.phase]||"#68879e",label=String($("show-gid").checked?n.node_id:n.su_type);
      g.append(svgElement("circle",{r:nodeRadius,fill,stroke:changed?"#008f75":"white","stroke-width":changed?2.5:1,"stroke-dasharray":special?"2 2":"none"}));
      g.append(svgElement("text",{y:0,"text-anchor":"middle","dominant-baseline":"central",fill:"white","font-size":Math.min(labelSize,.78*2*nodeRadius/(.7*label.length)),"font-family":"DejaVu Sans, Arial, sans-serif","font-weight":"bold"},label));
      g.addEventListener("click",e=>{e.stopPropagation();selectedGid=n.logical_gid??n.node_id;detail(n,n.preview);});svg.append(g);
    }
    bounds=xs.length?[Math.min(...xs)-32,Math.min(...ys)-35,Math.max(...xs)-Math.min(...xs)+64,Math.max(...ys)-Math.min(...ys)+70]:[-50,-50,100,100];
    if($("auto-fit").checked&&focusIds.length){const ps=focusIds.map(id=>positions[id]).filter(Boolean);if(ps.length){const fx=ps.map(p=>p[0]),fy=ps.map(p=>p[1]);view=[Math.min(...fx)-45,Math.min(...fy)-45,Math.max(...fx)-Math.min(...fx)+90,Math.max(...fy)-Math.min(...fy)+90];applyView();}else fit();}
    else if($("auto-fit").checked||!view)fit();else applyView();
  }
  function renderResources(){
    if(!catalog)return;
    const nodes={},edges={},titles=[];let rowY=0;
    for(const phase of ["rigid","flex","ring","side"]){
      const units=catalog.units.filter(u=>u.phase===phase);if(!units.length)continue;
      titles.push({position:[0,rowY+2],label:phaseNames[phase]+" · "+units.length,phase});
      let x=0, y=rowY, rowHeight=0;
      for(const u of units){const ps=u.nodes.map(n=>xy(n.local_axial));if(!ps.length)continue;
        const minX=Math.min(...ps.map(p=>p[0])),maxX=Math.max(...ps.map(p=>p[0])),minY=Math.min(...ps.map(p=>p[1])),maxY=Math.max(...ps.map(p=>p[1]));
        const width=Math.max(140,maxX-minX+65),height=Math.max(120,maxY-minY+60);
        if(x+width>1100&&x>0){x=0;y-=rowHeight/36+1;rowHeight=0;}
        for(const n of u.nodes){const p=xy(n.local_axial),cx=p[0]-minX+x,cy=p[1]-minY-y*36;const q=cx/(Math.sqrt(3)/2*36),r=-cy/36+q/2;
          nodes[u.unit_id+":"+n.node_id]={...n,unit_id:u.unit_id,phase,preview:true,logical_gid:n.node_id,final_position:[q,r]};}
        for(const [i,e]of u.internal_edges.entries())edges[u.unit_id+":"+i]={source_node_id:u.unit_id+":"+e.source_node_id,target_node_id:u.unit_id+":"+e.target_node_id};
        titles.push({position:[(x+5)/(Math.sqrt(3)/2*36),y+1+(x+5)/(Math.sqrt(3)/2*36)/2],label:u.unit_id+(u.side_category?(u.side_category==="complex"?" [复杂]":" [简单]"):"")+(u.topology_notation?" · "+u.topology_notation:""),phase});
        rowHeight=Math.max(rowHeight,height);x+=width;
      }
      rowY=y-rowHeight/36-3;
    }
    if(catalog.substitutions.length)titles.push({position:[0,rowY],label:"取代请求 · "+catalog.substitutions.map(r=>"SU"+r.target_su+" / GID "+r.target_node_id).join("；"),phase:"substitution"});
    $("counts-info").textContent=`原始输入 ${catalog.node_count} 个节点 · 取代请求 ${catalog.substitutions.length} 个`;
    shown={nodes,edges,preview:true};drawGraph(nodes,edges,null,titles);$("canvas-title").textContent="全部输入拓扑资源";$("frame-type").textContent="INPUT RESOURCES";
    $("caption").textContent=`${catalog.units.length} 个拓扑单元 · ${catalog.node_count} 个输入节点 · ${catalog.contracts.length} 项连接合同。分区位置仅用于预览；边界端点由合同引用。`;
    inventory(null);
    if(selectedGid!==null){const n=Object.values(nodes).find(n=>n.node_id===selectedGid);if(n)detail(n,true);}
  }
  function connectedFocus(value,action){
    if(!$("focus-action").checked)return [];
    const seeds=[action?.source_node_id,action?.target_node_id].filter(id=>id!==undefined&&id!==null&&value.nodes[String(id)]).map(String);
    if(!seeds.length){const main=new Set(value.completion_report.main_component);return Object.keys(value.nodes).filter(id=>main.has(value.nodes[id].unit_id));}
    const neighbors=new Map();for(const e of Object.values(value.edges)){const a=String(e.source_node_id),b=String(e.target_node_id);if(!neighbors.has(a))neighbors.set(a,[]);if(!neighbors.has(b))neighbors.set(b,[]);neighbors.get(a).push(b);neighbors.get(b).push(a);}
    const seen=new Set(seeds),stack=[...seeds];while(stack.length){for(const id of neighbors.get(stack.pop())||[])if(!seen.has(id)){seen.add(id);stack.push(id);}}
    return [...seen];
  }
  function renderState(value,old,action){
    shown=value;$("counts-info").textContent=`当前 ${value.counts.nodes} 个已放置节点 · ${value.counts.edges} 条边 · ${value.counts.components} 个组件`;let nodes=value.nodes;
    if($("main-only").checked){const main=new Set(value.completion_report.main_component);nodes=Object.fromEntries(Object.entries(nodes).filter(([,n])=>main.has(n.unit_id)));}
    drawGraph(nodes,value.edges,old,[],value.backbone_hull||[],connectedFocus(value,action));inventory(value.completion_report);
    if(selectedGid!==null){const n=value.nodes[String(selectedGid)];if(n)detail(n);else $("node-detail").textContent=`GID ${selectedGid} 在本帧未放置或已移除。原始资源仍可从输入阶段查看。`;}
  }
  function inventory(report){
    $("inventory").replaceChildren();
    const rows=report?.stages||Object.fromEntries(Object.entries(catalog?.counts||{}).map(([k,v])=>[k,{total:v}]));
    for(const phase of ["rigid","flex","ring","side","substitution"]){const r=rows[phase]||(phase==="substitution"?{total:catalog?.substitutions.length||0}:null);if(!r)continue;const row=document.createElement("div");row.className="row";const name=document.createElement("span"),count=document.createElement("span");name.textContent=phase==="rigid"?(report?"芳香团簇（主组件内）":"输入芳香团簇"):phaseNames[phase];count.textContent=r.completed===undefined?`${r.total} 个`:`${r.completed} / ${r.total}`;row.append(name,count);$("inventory").append(row);}
    $("unconnected").textContent=report?JSON.stringify(Object.fromEntries(Object.entries(report.stages).map(([p,r])=>[phaseNames[p],r.unconnected])),null,2):"开始组装后显示";
  }
  function pause(){playing=false;clearTimeout(timer);$("play").textContent="播放";}
  function invalidate(){generation++;selectedGid=null;selectedPhase=null;$("node-detail").textContent="点击节点查看 GID、标签与真实坐标。";seekingControls(false);pause();queue=[];cache.clear();current=null;shown=null;seq=0;stopAt=null;view=null;if(stream){stream.close();stream=null;}}
  async function show(e,token=generation){
    if(!e)return;if(e.kind==="input.ready"){if(token!==generation)return;renderResources();}
    else if((mode!=="live"||!e.kind.startsWith("forward."))&&e.state_id!=null){const [s,old]=await Promise.all([state(e.state_id),state(e.parent_state_id)]);if(token!==generation)return;renderState(s,old,e.data.action);}
    else if(e.kind==="run.finished"&&manifest.final_state){const s=await state(manifest.final_state);if(token!==generation)return;renderState(s);}
    if(token!==generation)return;
    seq=e.seq;current=e;$("position").textContent=`${seq} / ${latest}`;$("scrubber").value=seq;
    const tentative=e.scope==="search",forward=e.scope==="forward";
    $("frame-type").textContent=e.kind==="input.ready"?"INPUT RESOURCES":tentative?"TRIAL CONNECTION":forward?"LOOKAHEAD ASSESSMENT":"ACCEPTED ASSEMBLY";
    if(e.kind!=="input.ready")$("canvas-title").textContent=(phaseNames[e.phase]||"组装")+" · "+(kindNames[e.kind]||e.kind);
    if(e.kind!=="input.ready")$("caption").textContent=`事件 ${e.seq} · 流程 ${e.elapsed.toFixed(2)} s · ${tentative?"搜索试算，尚未正式接受":forward?"在副本上评估后续可连接性":"正式过程"}`;
    const d=e.data;$("decision").textContent=kindNames[e.kind]||e.kind;
    const completion=d.forward_completion??d.forward_completion_rate;
    if(completion!==undefined)$("decision").textContent+=` · 前瞻完成率 ${(completion*100).toFixed(2)}%`;
    if(d.fallback)$("decision").textContent+=" · 已评估合法结果保底选择";
    $("event-detail").textContent=JSON.stringify(e,null,2);
    $("playback-note").textContent=`逐事件播放 · 当前 ${seq}，已记录到 ${latest}。暂停只影响画面；历史保留在磁盘。`;
    document.body.dataset.sequence=seq;document.body.dataset.event=e.kind;
    document.dispatchEvent(new CustomEvent("assembly-frame",{detail:{seq,kind:e.kind,state_id:e.state_id,round_id:e.round_id}}));
  }
  async function next(){
    if(fetching)return;fetching=true;const token=generation;
    try{
      if(!queue.length){const data=await trace("events",{after:seq,mode,limit:250,before:stopAt??undefined});if(token!==generation)return;queue=data.events;if(selectedPhase&&$("stage-only").checked){const p=(data.phases||phases).find(p=>p.phase===selectedPhase);if(p?.end_seq!=null){stopAt=p.end_seq;queue=queue.filter(e=>e.seq<=stopAt);}}latest=Math.max(latest,data.latest_sequence);}
      if(queue.length){const e=queue.shift();await show(e,token);if(stopAt!==null&&seq>=stopAt)pause();return true;}
      if(!["running","starting","preparing"].includes(manifest.status)||stopAt!==null)pause();return false;
    }finally{fetching=false;}
  }
  async function tick(){if(!playing)return;try{await next();}catch(e){message(e.message);pause();}if(playing)timer=setTimeout(tick,Number($("speed").value));}
  function play(){if(seeking)return;if(playing){pause();return;}playing=true;$("play").textContent="暂停画面";tick();}
  function seekingControls(flag){seeking=flag;$("play").disabled=flag;$("next").disabled=flag;}
  async function seek(at){pause();queue=[];const token=++generation;seekingControls(true);try{const e=await trace("position",{at,mode});if(token===generation){if(e)await show(e,token);else {seq=0;current=null;renderResources();$("position").textContent=`0 / ${latest}`;$("scrubber").value=0;}}}finally{if(token===generation)seekingControls(false);}}
  function phaseButtons(){const nav=$("phases");nav.replaceChildren();for(const p of phases){
    const b=document.createElement("button");b.textContent=phaseNames[p.phase]||p.phase;b.dataset.phase=p.phase;
    b.onclick=async()=>{pause();queue=[];selectedPhase=p.phase;const token=++generation;seekingControls(true);
      try{stopAt=$("stage-only").checked?p.end_seq:null;
        const e=await trace("position",{at:p.start_seq,mode:"formal"});
        if(token!==generation)return;
        await show(e,token);if(token!==generation)return;
        $("canvas-title").textContent=phaseNames[p.phase]+" · 阶段起点";
        $("caption").textContent="已恢复这个阶段开始前的状态，点击播放逐步回放。";
        nav.querySelectorAll("button").forEach(x=>x.classList.toggle("active",x===b));
      }catch(e){message(e.message);}finally{if(token===generation)seekingControls(false);}};
    nav.append(b);}}

  async function refresh(){
    const token=generation;
    const [m,p]=await Promise.all([trace("manifest"),trace("phases")]);if(token!==generation)return;
    const priorStatus=manifest.status;manifest=m;latest=m.latest_sequence;if(run&&priorStatus!==m.status)history().catch(e=>message(e.message));$("scrubber").max=latest;$("position").textContent=`${seq} / ${latest}`;$("run-status").textContent=`${m.status} · ${phaseNames[m.current_phase]||"输入"}`;
    if(JSON.stringify(phases)!==JSON.stringify(p)){phases=p;phaseButtons();}
    $("manifest").textContent=JSON.stringify(m,null,2);$("sample-note").hidden=!m.synthetic;
    $("start").disabled=m.status!=="ready"||!run||!inputMatches();
    if(m.error)message(JSON.stringify(m.error));
    if(run){$("final-download").hidden=!m.final_state;$("final-download").href=`/api/runs/${run}/final`;}
    if(playing)$("playback-note").textContent=`逐事件播放 · 当前 ${seq}，计算已记录到 ${latest}。所有连接均已保存在磁盘，播放暂停不停止计算。`;
  }
  function subscribe(){if(embedded)return;if(stream)stream.close();stream=new EventSource(traceUrl("stream",{after:seq}));stream.onmessage=()=>refresh().catch(e=>message(e.message));stream.addEventListener("finished",()=>{stream.close();stream=null;refresh().catch(e=>message(e.message));});stream.onerror=()=>{stream?.close();stream=null;};}
  function inputMatches(){return $("nodes-path").value.trim()===manifest.inputs?.nodes?.path&&$("topology-path").value.trim()===manifest.inputs?.topology?.path;}
  async function attach(id,live=false){invalidate();run=id||null;await refresh();if(run){$("nodes-path").value=manifest.inputs?.nodes?.path||"";$("topology-path").value=manifest.inputs?.topology?.path||"";$("start").disabled=manifest.status!=="ready";}catalog=await trace("resources");await show(await trace("position",{at:phases[0]?.start_seq||latest,mode:"formal"}));subscribe();if(live&&manifest.status!=="ready")play();}
  const numberFields=["connection_candidate_limit","forward_checkpoint_interval","forward_min_completion","forward_max_rollbacks","clearance_radius","c_puct","seed"];
  function configValues(){const c={...defaults};for(const k of numberFields)c[k]=Number($(k).value);c.reward_weights=$("reward_weights").value.split(/[,，\s]+/).filter(Boolean).map(Number);if(c.reward_weights.length!==5||c.reward_weights.some(x=>!Number.isFinite(x)))throw Error("请输入五项数字奖励权重");for(const k of ["spectrum_csv","element_composition_csv","g2s_checkpoint"])c[k]=$(k).value.trim()||null;for(const k of ["verbose","save_stage_images"])c[k]=$(k).checked;return c;}
  function fillConfig(c){defaults=c;for(const k of numberFields)$(k).value=c[k];$("reward_weights").value=c.reward_weights.join(", ");for(const k of ["spectrum_csv","element_composition_csv","g2s_checkpoint"])$(k).value=c[k]||"";for(const k of ["verbose","save_stage_images"])$(k).checked=Boolean(c[k]);}
  function samplePaths(){const root=context.project+"/test_results/"+$("sample").value;$("nodes-path").value=root+"/final_outputs/final_nodes_by_global_id.csv";$("topology-path").value=root+"/final_outputs/final_topology_specs.csv";$("output-path").value=root+"/rl_mtcs_nmr";}
  async function history(){if(context.fixed||embedded)return;const data=await api("/api/runs");const select=$("history");select.replaceChildren(new Option("选择已保存的记录",""));for(const r of data.runs)select.add(new Option(`${r.sample} · ${r.status} · ${r.created_at}`,r.id));if(run)select.value=run;}
  $("run-form").onsubmit=async e=>{e.preventDefault();pause();$("read").disabled=true;try{const payload={sample:$("sample").value,nodes_csv:$("nodes-path").value,topology_csv:$("topology-path").value,output_root:$("output-path").value,python:$("python").value,config:configValues()};const info=await api("/api/prepare",payload);invalidate();run=info.id;$("command").textContent=info.command;$("output-note").textContent=info.output_dir;$("input-status").textContent="正在读取和校验拓扑…";let poll=async()=>{try{const status=await api("/api/runs/"+info.id);if(run!==info.id)return;if(status.status==="preparing"){setTimeout(poll,400);return;}$("read").disabled=false;if(status.status==="error")throw Error(JSON.stringify(status.error));await attach(info.id);$("input-status").textContent="已读取，等待点击开始组装";await history();}catch(err){$("read").disabled=false;message(err.message);}};poll();}catch(err){$("read").disabled=false;message(err.message);}};
  $("start").onclick=async()=>{try{if(!inputMatches())throw Error("输入路径已变更，请先重新读取拓扑");const info=await api(`/api/runs/${run}/start`,{config:configValues(),python:$("python").value});generation++;queue=[];$("start").disabled=true;$("command").textContent=info.command;$("input-status").textContent="组装运行中";manifest.status="running";$("configuration").open=false;subscribe();if(!playing)play();}catch(e){message(e.message);}};
  $("history").onchange=async()=>{if(!$("history").value)return;try{const id=$("history").value,info=await api("/api/runs/"+id);if(info.config)fillConfig(info.config);$("command").textContent=info.command||"历史记录";$("output-note").textContent=info.output_dir;await attach(id,info.status==="running");}catch(e){message(e.message);}};
  $("sample").onchange=()=>{samplePaths();$("start").disabled=true;};for(const key of ["nodes-path","topology-path"])$(key).oninput=()=>{$("start").disabled=manifest.status!=="ready"||!inputMatches();};$("refresh-history").onclick=()=>history().catch(e=>message(e.message));
  $("open-trace").onclick=async()=>{try{const info=await api("/api/open",{path:$("open-path").value});await history();await attach(info.id);}catch(e){message(e.message);}};
  $("play").onclick=play;$("next").onclick=()=>{pause();next().catch(e=>message(e.message));};
  $("previous").onclick=()=>seek(Math.max(0,seq-1)).catch(e=>message(e.message));$("from-start").onclick=()=>{selectedPhase=null;stopAt=null;seek(phases[0]?.start_seq||1).catch(e=>message(e.message));};
  $("follow").onclick=()=>{selectedPhase=null;stopAt=null;if(!playing)play();};$("scrubber").onchange=()=>{selectedPhase=null;stopAt=null;seek(Number($("scrubber").value)).catch(e=>message(e.message));};
  $("mode").onchange=()=>{mode=$("mode").value;stopAt=null;seek(seq).catch(e=>message(e.message));};
  $("fit").onclick=fit;for(const id of ["main-only","auto-fit","focus-action","show-gid","show-hull"])$(id).onchange=()=>{if(shown?.preview)renderResources();else if(shown)renderState(shown,null,current?.data.action);};
  const svg=$("molecule-svg");svg.addEventListener("wheel",e=>{e.preventDefault();if(!view)return;const ratio=e.deltaY>0?1.15:1/1.15;view=[view[0]+view[2]*(1-ratio)/2,view[1]+view[3]*(1-ratio)/2,view[2]*ratio,view[3]*ratio];$("auto-fit").checked=false;applyView();},{passive:false});
  svg.onpointerdown=e=>{if(!view)return;if(e.target.closest(".node"))return;drag={x:e.clientX,y:e.clientY,view:[...view]};svg.setPointerCapture(e.pointerId);};svg.onpointermove=e=>{if(!drag)return;const box=svg.getBoundingClientRect();view=[drag.view[0]-(e.clientX-drag.x)*drag.view[2]/box.width,drag.view[1]-(e.clientY-drag.y)*drag.view[3]/box.height,drag.view[2],drag.view[3]];$("auto-fit").checked=false;applyView();};svg.onpointerup=()=>drag=null;
  function download(name,content,type){const a=document.createElement("a"),url=URL.createObjectURL(new Blob([content],{type}));a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  $("export-svg").onclick=()=>download(`assembly_${seq}.svg`,new XMLSerializer().serializeToString(svg),"image/svg+xml");
  async function exportRun(video=false){if(!run){message("离线回放可导出当前 SVG；其他导出请使用命令行 export。");return;}message("正在导出，请稍候…");try{const result=await api(`/api/runs/${run}/export`,{video,mode});$("exports").replaceChildren();for(const [label,url]of [["离线回放",result.html_url],["最终 SVG",result.svg_url],["最终 PDF",result.pdf_url],["过程视频",result.video_url]])if(url){const a=document.createElement("a");a.textContent=label;a.href=url;a.target="_blank";$("exports").append(a);}message(`SVG / PDF / PNG 阶段图已保存到 ${result.figure_dir}`);}catch(e){message(e.message);}}
  $("export").onclick=()=>exportRun();$("export-video").onclick=()=>exportRun(true);
  $("load-log").onclick=async()=>{try{$("log").textContent=run?(await api(`/api/runs/${run}/log`)).log:"此记录没有工作进程日志接口";}catch(e){message(e.message);}};
  async function boot(){
    if(embedded){context.fixed=true;$("configuration").hidden=true;document.querySelector(".history").hidden=true;await attach(null);return;}
    context=await api("/api/context");if(context.fixed){$("configuration").hidden=true;document.querySelector(".history").hidden=true;await attach(null);return;}
    fillConfig(context.defaults);$("python").value=context.python;const requested=new URLSearchParams(location.search).get("sample")||"YM-3";const samples=[...new Set([requested,...context.samples])];for(const name of samples)$("sample").add(new Option(name,name));$("sample").value=requested;samplePaths();await history();
  }
  setInterval(()=>{if((run||context.fixed)&&!embedded)refresh().catch(e=>message(e.message));},2000);
  boot().catch(e=>message(e.message));
})();
