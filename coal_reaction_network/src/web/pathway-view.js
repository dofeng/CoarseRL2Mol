"use strict";

async function startPathway(offset=0, fresh=false, reuse=false) {
  if(!state.db)return;
  const query=!fresh && (offset || reuse) && state.pathwayQuery ? {...state.pathwayQuery,offset} : {
    ...rangeQuery(),direction:byId("pathDirection").value,species:byId("species").value,
    fragment:byId("pathFragment").value || "0",target_species:byId("pathTarget").value,
    event_limit:byId("pathLimit").value,summary_stages:byId("summaryStages").value,offset};
  const task=await api("/api/pathway/start",{},query);
  state.pathwayTask=task.id;state.pathwayQuery=query;
  byId("pathOffset").value=offset;
  byId("pathStart").disabled=true;
  byId("pathCancel").disabled=false;
  byId("graphSummary").textContent="正在扫描所有相邻帧，追踪固定原子集合…";
  const poll=async()=>{
    try {
      if(state.pathwayTask!==task.id)return;
      const status=await api("/api/pathway/status",{id:task.id});
      const progress=status.progress;
      byId("graphSummary").textContent=progress ? `路径任务 ${status.status} · 帧 ${progress.frame} / ${progress.end} · 已发现 ${progress.relevant_events} 个相关事件` : status.status;
      if(status.status==="running") {state.pathwayTimer=setTimeout(poll,700);return;}
      byId("pathStart").disabled=false;byId("pathCancel").disabled=true;
      if(status.status!=="completed")throw new Error(status.error || status.status);
      state.pathway=await api("/api/pathway/result",{id:task.id});
      if(state.db!==query.db){state.pathway=null;throw new Error("数据库已切换，请在当前数据库重新构建路径");}
      state.pathwayLayout=null;
      state.pathwayLayouts={};state.pathwayPan={x:0,y:0};
      state.pathwayRotation={yaw:0,pitch:.16};state.pathwayScale=1;state.pathwaySelected=null;
      if(!byId("representation").value.startsWith("path"))byId("representation").value="path3d";
      renderPathway();
    } catch(error) {byId("pathStart").disabled=false;byId("pathCancel").disabled=true;notify(error.message,true);}
  };
  await poll();
}

function renderPathway() {
  state.pathDrag=null;
  const mode=byId("representation").value;
  const summary=mode==="pathflow" || mode==="path3d";
  const raw=state.pathway;
  const data=raw && summary && raw.overview ? {...raw,...raw.overview} : raw && mode==="pathdetail3d" && raw.compact ? {...raw,...raw.compact} : raw;
  state.coarseNetwork=null;
  byId("tableTab").textContent=summary ? "阶段过程" : "路径事件";
  byId("tableArea").innerHTML=data ? `<table><thead><tr><th>帧对</th><th>类型</th><th>${summary ? "过程摘要（非反应式）" : "完整反应式"}</th></tr></thead><tbody>${data.events.map((event,index)=>`<tr data-path-event="${index}"><td>${event.frame} → ${event.child_frame}</td><td>${escapeText(event.event_type)}</td><td>${escapeText(event.signature)}</td></tr>`).join("")}</tbody></table>` : '<p class="muted">请先构建路径。</p>';
  byId("tableArea").querySelectorAll("[data-path-event]").forEach(row=>row.onclick=()=>inspectPathNode(data.events[+row.dataset.pathEvent]));
  const three=mode==="path3d" || mode==="pathdetail3d";
  byId("graphArea").classList.toggle("pathway-3d",three);
  state.networkCursor=null;byId("nextPage").disabled=true;
  if(!data) {
    state.pathScene=null;byId('pathBrowseControls').hidden=true;
    byId("pathNext").disabled=true;
    byId("graphSummary").textContent="尚未构建当前数据库的时间路径";
    byId("graphArea").innerHTML='<div class="empty"><h3>选择时间区间与追踪对象</h3><p>选择类别或输入锚点帧片段 ID，然后点击「构建路径」。前向从起始帧追踪，反向从结束帧追溯。</p></div>';return;
  }
  createPathwayScene(data,mode,summary);
  byId("graphSummary").textContent=`实际区间 ${data.start}–${data.end} · 锚点帧 ${data.anchor} · 固定 ${data.seed_atoms} 个原子 · 显示事件 ${data.events.length ? data.offset+1 : 0}–${data.events.length ? data.offset+data.events.length : 0} / ${data.total_events} · 原子缺席累计 ${data.missing_atom_frame_occurrences} 次`;
  byId("pageInfo").textContent="橙色：每次事件的最大原子延续分支（并列均保留），不是唯一全局机理；分页之间不伪造连接。";
  if(summary) {
    byId("graphSummary").textContent=`全区间 ${data.start}–${data.end} · ${data.events.length} 个阶段 · 汇总 ${data.event_count ?? data.total_events} 个事件 · 点击过程展开证据`;
    byId("pageInfo").textContent="阶段节点是多事件过程，不是单步反应；边界类别为追踪原子群组。普通滚轮不缩放，左键＋滚轮局部缩放。";
  }
  byId("pathNext").disabled=summary || data.next_offset==null;
  document.querySelector(".legend").innerHTML='<h4>路径视觉编码</h4><p>节点颜色：C 区间<br>P 编号：固定 C/R/O 类别<br>方形 E：完整反应事件<br>橙色：主要延续分支<br>蓝色：次要分支 / 反应参与<br>灰虚线：共同参与者<br>面积：当前追踪原子数<br>点击查看完整证据</p>';
  if(summary)document.querySelector('.legend').innerHTML='<h4>阶段流程说明</h4><p>圆点：边界时刻同类原子群组<br>圆角块：多事件阶段<br>橙色：阶段边界最大群组<br>蓝色：其他群组<br>曲线不是直接反应式<br>具体原子去向见过程详情<br>稀有中间体请展开证据</p>';
}

function inspectPathNode(node) {
  if(node.process) {
    const transfers=(state.pathway.overview?.transfers || []).filter(item=>item.process_id===node.id);
    byId("detail").innerHTML=`<div class="detail-kicker">MULTI-EVENT PROCESS</div><h2>过程 ${node.event_id}</h2><p>帧 ${node.frame}–${node.child_frame} · ${node.event_count} 个事件</p><pre>${escapeText(JSON.stringify(node.event_types,null,2))}</pre><p class="muted">这是全区间的阶段摘要，不是单步反应。展开保留原始锚点，不重新定义追踪原子。逐事件页从本阶段首个事件开始，可能延伸到后续阶段。</p><h4>真实边界原子去向（非反应式）</h4>${transfers.map(item=>`<p>${escapeText(speciesPresentation(item.source).name)} → ${escapeText(speciesPresentation(item.target).name)}：${item.atoms} 原子</p>`).join('')}<button id="expandProcess" class="button primary" ${node.evidence_offset==null ? "disabled" : ""}>展开逐事件证据</button>`;
    byId("expandProcess").onclick=guard(async()=>{byId("representation").value="pathdetail3d";await startPathway(node.evidence_offset,false,true);});
    return;
  }
  if(node.kind==="event") {
    byId("detail").innerHTML=`<div class="detail-kicker">OBSERVED EVENT</div><h2>E${node.event_id} · 帧 ${node.frame} → ${node.child_frame}</h2><div class="equation">${escapeText(node.signature)}</div><p>${escapeText(node.event_type)} · ${escapeText(node.resolution)}</p><button id="pathEvidence" class="button primary">查看原始事件与片段</button>`;
    byId("pathEvidence").onclick=guard(async()=>{const result=await api("/api/events",{db:state.pathwayQuery.db,start:node.frame,end:node.child_frame,event_id:node.event_id,limit:1});const event=result.rows[0];if(event)inspectEvent(event);else notify("原始事件不存在，请检查数据库是否在查询后改变。",true);});
  } else {
    const label=speciesPresentation(node.species);
    byId("detail").innerHTML=`<div class="detail-kicker">TIME-RESOLVED STATE</div><h2>${escapeText(label.name)}</h2><div class="equation">${escapeText(label.description)}</div><p>帧 ${node.first_frame}–${node.last_frame}<br>追踪原子 ${node.tracked} / ${state.pathway.seed_atoms}<br>${node.context ? "共同参与者，不扩张锚点集合" : "原子谱系状态，不是唯一分子物种"}</p><button id="pathFragmentEvidence" class="button primary">查看片段实例</button>`;
    byId("pathFragmentEvidence").onclick=guard(()=>inspectFragment(node.first_frame,node.fragment_id));
    if(node.aggregate){byId("pathFragmentEvidence").hidden=true;byId("detail").insertAdjacentHTML("beforeend",'<p class="muted">阶段边界的同类原子群组，不是单个片段；请展开相邻过程查看实例证据。</p>');}
    if(node.activity_events) {
      byId("detail").insertAdjacentHTML("beforeend",`<h4>已折叠 ${node.activity_events.length} 次一对一类内变化</h4>${node.activity_events.map((event,index)=>`<button class="button ghost" data-activity="${index}">帧 ${event.frame} · E${event.event_id}</button>`).join("")}`);
      byId("detail").querySelectorAll('[data-activity]').forEach(button=>button.onclick=guard(async()=>{const evidence=node.activity_events[+button.dataset.activity];const result=await api('/api/events',{db:state.pathwayQuery.db,start:evidence.frame,end:evidence.frame+1,event_id:evidence.event_id,limit:1});if(result.rows[0])inspectEvent(result.rows[0]);}));
    }
  }
}

function exportPathway() {
  if(!state.pathway)return;
  const url=URL.createObjectURL(new Blob([JSON.stringify({query:state.pathwayQuery,...state.pathway},null,2)],{type:"application/json"}));
  const anchor=document.createElement("a");anchor.href=url;anchor.download="observed-pathway.json";anchor.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
