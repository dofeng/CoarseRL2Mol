"use strict";

const byId = identifier => document.getElementById(identifier);
const escapeText = value => String(value ?? "").replace(/[&<>"']/g, character => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[character]));
const state = {db:"", task:"", channels:[], networkCursor:null, eventsCursor:null, fragmentCursor:null, signature:"", species:"", timer:null, chart:null, generation:0, bounds:[0,0,1280,760], graphHeight:760};
const colors = {"C0":"#64748b","C1-5":"#008577","C6-20":"#3d6cc0","C21-40":"#ae5f91","C40+":"#bf7523"};
const eventNames = {fragmentation:"裂解",fusion:"融合",transformation:"转化",internal_rearrangement:"类内重排",many_to_many:"多对多",birth:"出现",death:"消失",mixed:"混合类型"};
const stages = {discovery:"校验输入", frames:"逐帧解析与分类", transitions:"相邻帧事件识别", tracks:"实例轨迹与类别驻留", annotations:"时间与质量标注", views:"反应通道聚合", coarse_flows:"C/R/O 物质流与守恒审计", export:"流式导出", done:"所有阶段结束"};
const formatNumber = value => Number(value || 0).toLocaleString("zh-CN");
const formatSide = nodes => Object.entries(nodes || {}).map(([species, amount]) => `${amount > 1 ? amount + " × " : ""}${species}`).join(" + ") || "∅";
const equation = channel => `${formatSide(channel.reactant_nodes)} → ${formatSide(channel.product_nodes)}`;
const sideHtml = nodes => Object.entries(nodes || {}).map(([species,amount])=>{
  const label=speciesPresentation(species);
  return `${amount>1 ? `<span class="coefficient">${amount} × </span>` : ""}<span class="species-token" title="${escapeText(label.description+' / '+species)}">${escapeText(label.name)}</span>`;
}).join(" + ") || "∅";

function syncControls() {
  const mode=byId("representation").value, pathway=mode.startsWith("path");
  document.querySelectorAll(".pathway-settings").forEach(element=>element.hidden=!pathway);
  document.querySelectorAll(".network-settings").forEach(element=>element.hidden=pathway);
  document.querySelectorAll(".weighted-settings").forEach(element=>element.hidden=mode==="all");
  document.querySelectorAll(".flow-setting").forEach(element=>element.hidden=mode==="all" || byId("flowWeight").value==="events");
  byId("applyFilters").hidden=pathway;
}

function fitNetwork(readable=false) {
  const svg=byId("networkSvg");
  if(!svg)return;
  const width=state.bounds[2];
  sizeGraph(svg,Math.max(readable ? .65 : .05,Math.min(1,(byId("graphArea").clientWidth-16)/width)));
  byId("graphArea").scrollLeft=0;
}

function notify(message, error=false) {
  byId("message").hidden = !message;
  byId("message").className = error ? "error" : "";
  byId("message").textContent = message;
}

async function api(path, query={}, body=null) {
  const url = new URL(path, location.origin);
  Object.entries(query).forEach(([key,value]) => { if (value !== "" && value != null) url.searchParams.set(key,value); });
  const response = await fetch(url, body ? {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)} : {});
  const result = await response.json();
  if (!response.ok || result.error) throw new Error(result.error || response.statusText);
  return result;
}

function guard(action) {
  return async event => {
    try { await action(event); } catch (error) { notify(error.message, true); }
  };
}

function rangeQuery() {
  return {db:state.db,start:byId("start").value,end:byId("end").value};
}

async function openDatabase() {
  const generation=state.datasetGeneration=(state.datasetGeneration || 0)+1;
  ++state.generation;
  state.chartGeneration=(state.chartGeneration || 0)+1;
  state.evidenceGeneration=(state.evidenceGeneration || 0)+1;
  const summary = await api("/api/summary", {db:byId("database").value});
  if(generation!==state.datasetGeneration)return;
  state.db = summary.database;
  state.summary=summary;
  state.signature="";state.species="";state.networkQuery=null;
  state.inspectedChannel=null;state.inspectedEvent=null;
  byId("detail").innerHTML='<div class="empty small"><h3>未选择对象</h3></div>';
  state.pathway=null;state.pathScene=null;
  byId("database").value = state.db;
  localStorage.setItem("coal-atlas-db", state.db);
  byId("datasetStatus").textContent = state.db.split("/").slice(-2).join("/");
  byId("end").value = summary.last_frame ?? 0;
  byId("start").value = summary.first_frame ?? 0;
  byId("frameNumber").max = summary.last_frame ?? 0;
  const metrics = [["完成帧",formatNumber(summary.completed_frames)],["类别",formatNumber(summary.species)],["反应事件",summary.reaction_events==null ? "--" : formatNumber(summary.reaction_events)],["计量通道",formatNumber(summary.reaction_channels)]];
  byId("metrics").innerHTML = metrics.map(([label,value]) => `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`).join("");
  byId("coverage").textContent = `${formatNumber(summary.completed_pairs)} / ${formatNumber(summary.expected_pairs)} 帧对 · ${formatNumber(summary.fragments)} 片段实例`;
  byId("coverage").title=summary.metadata.coverage_scope || "未验证文件编号完整性";
  document.querySelector(".status-dot")?.classList.toggle("ready",summary.metadata.views_ready==="true");
  byId("footerStatus").textContent=`帧 ${summary.first_frame}–${summary.last_frame} · ${summary.metadata.parser_version || "v3"}`;
  if (summary.legacy) {
    notify("旧版解析器结果不可用于最终分析。请更换数据库路径，重新处理原始文件。", true);
    return;
  }
  const species = await api("/api/species", {db:state.db});
  if(generation!==state.datasetGeneration)return;
  byId("species").innerHTML = '<option value="">全部物种</option>' + species.rows.map(row => `<option value="${escapeText(row.species_id)}">${escapeText(row.species_id)}</option>`).join("");
  const diagnostics = await api("/api/coarse-diagnostics",rangeQuery());
  if(generation!==state.datasetGeneration)return;
  state.coarseDiagnostics=diagnostics;
  populateCoarseBins(state.coarseDiagnostics);
  const options=species.rows.map(row=>{const label=speciesPresentation(row.species_id);return `<option value="${escapeText(row.species_id)}">${label.name} · ${escapeText(label.description)}</option>`;}).join("");
  byId("species").innerHTML='<option value="">请选择类别 / 全部</option>'+options;
  byId("pathTarget").innerHTML='<option value="">不限制</option>'+options;
  state.pathway=null;
  renderDiagnostics(state.coarseDiagnostics);
  state.diagnosticRange=JSON.stringify(rangeQuery());
  notify(summary.metadata.views_ready === "true" ? "" : "网络尚未完成", summary.failed_frames > 0 || summary.failed_pairs > 0);
  await Promise.all([loadNetwork(), loadChart()]);
}

async function runPipeline(event) {
  event.preventDefault();
  const payload = {input:byId("input").value,database:byId("database").value,pattern:byId("pattern").value,
    stage:byId("runStage").value,
    frame_order:byId("frameOrder").value,workers:+byId("workers").value,max_inflight:+byId("inflight").value,
    memory_limit_mib:+byId("memory").value,cutoff:+byId("cutoff").value,temporal_mode:byId("temporal").value,
    minimum_lifetime_frames:+byId("lifetime").value,reversal_window_frames:+byId("reversal").value,
    expected_frame_step:byId("frameStep").value,build_tracks:byId("tracksEnabled").checked,retry_failed:byId("retry").checked,
    config:byId("config").value};
  const result = await api("/api/run", {}, payload);
  state.task = result.task_id;
  byId("database").value = result.database;
  localStorage.setItem("coal-atlas-task", state.task);
  byId("run").disabled = true;
  byId("cancel").disabled = false;
  byId("setup").close();
  byId("jobBand").hidden=false;
  notify("计算任务已启动");
  pollTask();
}

async function pollTask() {
  byId("jobBand").hidden=false;
  clearTimeout(state.timer);
  if (!state.task) return;
  try {
    const result = await api("/api/task", {id:state.task});
    const progress = result.progress || {};
    const percent = progress.total ? 100 * (progress.completed || 0) / progress.total : 0;
    byId("jobStage").textContent = `${stages[progress.stage] || result.status} · ${result.status}`;
    byId("jobMeta").textContent = `${formatNumber(progress.completed)} / ${progress.total ? formatNumber(progress.total) : "待统计"} · RSS ${progress.rss_mib ?? "—"} MiB · 峰值 ${progress.peak_rss_mib ?? "—"} MiB · ${Math.round(progress.elapsed_seconds || 0)} s`;
    byId("jobProgress").value = percent;
    if (!progress.total && result.status === "running") byId("jobProgress").removeAttribute("value");
    if (["running","queued","cancelling"].includes(result.status)) {
      byId("run").disabled = true;
      byId("cancel").disabled = result.status === "cancelling";
      state.timer = setTimeout(pollTask, 1800);
    } else {
      byId("run").disabled = false;
      byId("cancel").disabled = true;
      localStorage.removeItem("coal-atlas-task");
      if (["completed","frames_completed"].includes(result.status)) {
        byId("database").value = result.database;
        await openDatabase();
      } else notify(progress.error || `任务状态：${result.status}。请检查 pipeline.log，修复原因后恢复。`, true);
    }
  } catch (error) { notify(error.message,true); state.timer = setTimeout(pollTask,5000); }
}

async function loadNetwork(cursor=null) {
  if (!state.db) return;
  syncControls();
  const generation = ++state.generation;
  byId("graphArea").classList.add("loading");
  try {
    const rangeKey=JSON.stringify(rangeQuery());
    if (state.diagnosticRange!==rangeKey) {
      const diagnostics=await api("/api/coarse-diagnostics",rangeQuery());
      if(generation!==state.generation)return;
      state.coarseDiagnostics=diagnostics;
      state.diagnosticRange=rangeKey;
      renderDiagnostics(diagnostics);
    }
    if(byId("representation").value.startsWith("path")) {renderPathway();return;}
    state.evidenceGeneration=(state.evidenceGeneration || 0)+1;
    state.signature="";state.inspectedChannel=null;state.inspectedEvent=null;
    byId("detail").innerHTML='<div class="empty small"><h3>未选择对象</h3></div>';
    byId("tableTab").textContent="通道表";
    byId("graphArea").classList.remove("pathway-3d");
    byId("graphArea").onpointerdown=null;
    byId("graphArea").onpointermove=null;
    byId("graphArea").onpointerup=null;
    byId("graphArea").onwheel=null;state.pathDrag=null;
    state.pathScene=null;byId('pathBrowseControls').hidden=true;
    const representation=byId("representation").value;
    if(representation!=="all") {
      const query=coarseFilters();
      const result=await api("/api/coarse-network",query);
      if(generation!==state.generation)return;
      state.coarseNetwork=result;
      state.coarseQuery=query;
      state.networkQuery={...query};
      state.channels=result.channels;
      state.networkCursor=null;
      byId("nextPage").disabled=true;
      const coverage=(100*result.achieved_coverage).toFixed(1);
      byId("graphSummary").innerHTML=result.ready ? `<span class="summary-primary">${result.channels.length} / ${formatNumber(result.total_channels)} 通道</span><span class="summary-separator">/</span><span>${formatNumber(result.displayed_events)} / ${formatNumber(result.total_events)} 事件</span><span class="coverage-meter"><i style="width:${coverage}%"></i></span><span>${coverage}% ${byId("flowWeight").selectedOptions[0].textContent}覆盖</span>${result.target_reached ? "" : '<span class="badge warn">显示上限 / 覆盖未达目标</span>'}` : escapeText(result.message);
      const flows=result.flow_totals;
      byId("pageInfo").textContent=flows && query.weight!=="events" ? `跨类 ${formatNumber(flows['cross_'+query.weight])} · 类内 ${formatNumber(flows['internal_'+query.weight])}${result.include_internal ? '（已计入）' : '（未计入）'}${result.edges_truncated ? ' · 流边截断' : ''}` : `帧 ${query.start}–${query.end} · 覆盖分母：当前筛选内全部事件`;
      if(representation==="flow")renderFlowGraph(result,state.coarseDiagnostics);
      else renderGraph();
      renderChannelTable();
      return;
    }
    state.coarseNetwork=null;
    const query=coarseFilters();
    const result = await api("/api/network", {...query,cursor});
    if (generation !== state.generation) return;
    state.channels = result.channels;
    state.networkQuery={...query};
    state.networkCursor = result.next_cursor;
    byId("nextPage").disabled = !result.next_cursor;
    const coverage = result.total_events ? (100 * result.displayed_events/result.total_events).toFixed(1) : "0.0";
    byId("graphSummary").textContent = result.ready ? `当前页 ${result.channels.length} / ${formatNumber(result.total_channels)} 条通道 · 当前页事件占筛选结果 ${coverage}% · 帧 ${byId("start").value}—${byId("end").value} · ${result.view}` : result.message;
    byId("pageInfo").textContent = "按事件次数分页 · 覆盖分母：最少次数阈值之后的筛选结果";
    renderGraph();
    renderChannelTable();
  } finally {
    if (generation === state.generation) byId("graphArea").classList.remove("loading");
  }
}

function speciesOrder(left,right) {
  const sequence = Object.keys(colors);
  return sequence.indexOf(left.split("_")[0])-sequence.indexOf(right.split("_")[0]) || left.localeCompare(right,undefined,{numeric:true});
}

function renderGraph() {
  const legend=document.querySelector(".legend");
  if(legend)legend.innerHTML='<h4>计量网络</h4><div><i style="--swatch:#008577"></i>反应物</div><div><i style="--swatch:#bf7523"></i>产物</div><p>完整系数 · C 区间着色</p>';
  if (!state.channels.length) {
    byId("graphArea").innerHTML='<div class="empty"><h3>没有匹配通道</h3></div>';
    return;
  }
  const sides=["reactant_nodes","product_nodes"];
  const species=sides.map(side=>[...new Set(state.channels.flatMap(channel=>Object.keys(channel[side])))].sort(speciesOrder));
  const ranks=sides.map((side)=>new Map());
  // Place categories near their participating channels to reduce crossings.
  sides.forEach((side,sideIndex)=>species[sideIndex].forEach(identifier=>{
    let sum=0,weight=0;
    state.channels.forEach((channel,index)=>{const amount=channel[side][identifier] || 0;sum+=index*amount;weight+=amount;});
    ranks[sideIndex].set(identifier,sum/Math.max(1,weight));
  }));
  species.forEach((items,index)=>items.sort((left,right)=>ranks[index].get(left)-ranks[index].get(right) || speciesOrder(left,right)));
  const channelSizes=state.channels.map(channel=>Math.max(46,20*(Math.max(...sides.map(side=>Object.keys(channel[side]).length))-1)+16));
  const rowHeights=channelSizes.map(size=>Math.max(76,size+24));
  const rowsHeight=rowHeights.reduce((sum,value)=>sum+value,0);
  const height=Math.max(660,rowsHeight+140,Math.max(...species.map(items=>items.length))*76+140);
  state.graphHeight=height;
  const position=(index,total)=>104+(index+.5)*(height-174)/total;
  const positions=species.map(items=>new Map(items.map((identifier,index)=>[identifier,position(index,items.length)])));
  const maximum=Math.max(1,...state.channels.map(channel=>channel.event_count));
  const edges=[],nodes=[],labels=[];
  let rowOffset=0;
  state.channels.forEach((channel,index)=>{
    const y=104+(rowOffset+rowHeights[index]/2)*(height-174)/rowsHeight;
    rowOffset+=rowHeights[index];
    sides.forEach((side,sideIndex)=>{
      const entries=Object.entries(channel[side]);
      entries.forEach(([identifier,amount],port)=>{
        const speciesY=positions[sideIndex].get(identifier);
        const portY=y+(port-(entries.length-1)/2)*20;
        const x1=sideIndex ? 608 : 221,x2=sideIndex ? 879 : 492;
        const y1=sideIndex ? portY : speciesY,y2=sideIndex ? speciesY : portY;
        const color=speciesPresentation(identifier).color;
        edges.push(`<path class="edge" data-channel="${index}" data-species="${escapeText(identifier)}" d="M ${x1} ${y1} C ${x1+116} ${y1}, ${x2-116} ${y2}, ${x2} ${y2}" stroke="${color}" stroke-width="${1+2.1*Math.log1p(channel.event_count)/Math.log1p(maximum)}" fill="none" opacity=".5" marker-end="url(#arrow)"><title>${escapeText(equation(channel))} / ${amount}</title></path>`);
        if(amount>1)labels.push(`<g class="edge" data-channel="${index}" data-species="${escapeText(identifier)}"><rect x="${sideIndex?625:452}" y="${portY-8}" width="24" height="17" rx="3" fill="#fcfdfb"/><text x="${sideIndex?637:464}" y="${portY+4}" text-anchor="middle" font-size="11" font-weight="650" fill="${color}">${amount}</text></g>`);
      });
    });
    const uncertain=channel.uncertain_events>0;
    nodes.push(`<g class="node reaction" data-channel="${index}" tabindex="0" role="button" aria-label="${escapeText(equation(channel))}"><rect x="496" y="${y-channelSizes[index]/2}" width="108" height="${channelSizes[index]}" rx="4" fill="${uncertain?'#fff9ee':'#f1f7f0'}" stroke="${uncertain?'#d5b77c':'#a4bca8'}"/><text x="550" y="${y-4}" text-anchor="middle" font-size="11" font-weight="600" fill="#455c49">${escapeText(eventNames[channel.event_type] || '通道')} ${String(index+1).padStart(2,'0')}</text><text x="550" y="${y+13}" text-anchor="middle" font-size="11" fill="#788d7b">${formatNumber(channel.event_count)} 次</text><title>${escapeText(equation(channel))}</title></g>`);
  });
  species.forEach((items,sideIndex)=>items.forEach(identifier=>{
    const y=positions[sideIndex].get(identifier),x=sideIndex?890:210,label=speciesPresentation(identifier);
    const textX=sideIndex?911:189,anchor=sideIndex?'start':'end';
    nodes.push(`<g class="node species" data-species="${escapeText(identifier)}" tabindex="0" role="button" aria-label="${escapeText(label.description)}"><circle cx="${x}" cy="${y}" r="11" fill="${label.color}" stroke="#fff" stroke-width="2"/><text x="${textX}" y="${y-2}" text-anchor="${anchor}" font-size="14" font-weight="600" fill="#354b3e">${escapeText(label.name)}</text><text x="${textX}" y="${y+16}" text-anchor="${anchor}" font-size="10" fill="#859486">${escapeText(label.description)}</text><title>${escapeText(identifier)}</title></g>`);
  }));
  byId("graphArea").innerHTML=`<svg xmlns="http://www.w3.org/2000/svg" id="networkSvg" viewBox="0 0 1100 ${height}" role="img" aria-label="完整反应计量网络"><defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="none" stroke="#809284" stroke-width="1"/></marker></defs><rect width="1100" height="${height}" fill="#fcfdfb"/><g font-family="system-ui,sans-serif"><text x="210" y="46" text-anchor="middle" font-size="11" fill="#8a9b8d">反应物</text><text x="550" y="46" text-anchor="middle" font-size="11" fill="#8a9b8d">计量通道</text><text x="890" y="46" text-anchor="middle" font-size="11" fill="#8a9b8d">产物</text><line x1="250" x2="455" y1="43" y2="43" stroke="#e4ebe0"/><line x1="645" x2="850" y1="43" y2="43" stroke="#e4ebe0"/>${edges.join("")}${labels.join("")}${nodes.join("")}</g></svg>`;
  const svg=byId("networkSvg");
  svg.querySelectorAll(".node").forEach(node=>{
    const select=guard(()=>node.classList.contains("reaction") ? inspectReaction(+node.dataset.channel) : inspectSpecies(node.dataset.species));
    node.addEventListener("click",select);
    node.addEventListener("keydown",event=>{if(event.key==="Enter" || event.key===" "){event.preventDefault();select(event);}});
  });
  bindGraphNavigation(svg);
}

function bindGraphNavigation(svg) {
  const box=svg.getAttribute("viewBox").split(/\s+/).map(Number);
  state.bounds=[0,0,box[2],state.graphHeight];
  updateViewBox();
  fitNetwork(true);
}

function updateViewBox() {
  byId("networkSvg")?.setAttribute("viewBox",state.bounds.join(" "));
}

function zoom(factor) {
  explicitGraphScale(1/factor);
}

function highlight(index=null,species=null) {
  byId("networkSvg")?.querySelectorAll(".edge").forEach(edge=>{
    if(edge.classList.contains("flow-edge")) {
      const flow=state.coarseNetwork?.edges[+edge.dataset.flow];
      if(index!=null) {edge.style.removeProperty("opacity");return;}
      const related=flow?.source===species || flow?.target===species;
      edge.style.opacity=related ? "1" : ".16";
      return;
    }
    edge.style.opacity = (index!=null ? +edge.dataset.channel===index : edge.dataset.species===species) ? "1" : ".55";
  });
}

function renderChannelTable() {
  const maximum=Math.max(1,...state.channels.map(channel=>channel.event_count));
  byId("tableArea").innerHTML = `<table><thead><tr><th>完整计量</th><th>事件类型</th><th class="numeric">次数</th><th class="numeric">守恒</th></tr></thead><tbody>${state.channels.map((channel,index)=>`<tr data-index="${index}"><td class="channel-equation">${sideHtml(channel.reactant_nodes)} → ${sideHtml(channel.product_nodes)}<span class="channel-meta">帧 ${channel.first_frame ?? '—'}–${channel.last_frame ?? '—'} · ${channel.activity==='internal' ? '类别数量不变' : '类别数量变化'}</span></td><td><span class="badge ${channel.uncertain_events ? 'warn' : 'neutral'}">${escapeText(eventNames[channel.event_type] || channel.event_type)}</span></td><td class="numeric">${formatNumber(channel.event_count)}<span class="frequency-bar"><i style="width:${100*channel.event_count/maximum}%"></i></span></td><td class="numeric">${(100*channel.mean_atom_conservation_ratio).toFixed(1)}%</td></tr>`).join("")}</tbody></table>`;
  byId("tableArea").querySelectorAll("tr[data-index]").forEach(row=>row.addEventListener("click",guard(()=>inspectReaction(+row.dataset.index))));
}

async function inspectReaction(index) {
  const channel = state.channels[index];
  if(!channel)return;
  state.inspectedChannel=index;
  state.evidenceQuery={...state.networkQuery,signature:channel.reaction_signature};
  state.signature = channel.reaction_signature;
  highlight(index);
  const net=channel.net_coefficients || {};
  const netHtml=Object.entries(net).map(([species,amount])=>`<span class="badge ${amount>0?'net-gain':'net-loss'}">${escapeText(speciesPresentation(species).name)} ${amount>0?'+':''}${amount}</span>`).join('') || '<span class="badge neutral">0 · 类别数量不变</span>';
  const types=Object.entries(channel.event_type_counts || {[channel.event_type]:channel.event_count}).map(([type,count])=>`<div class="accounting-note"><span>${escapeText(eventNames[type] || type)}</span><span>${formatNumber(count)}</span></div>`).join('');
  byId("detail").innerHTML = `<div class="detail-kicker">CHANNEL ${String(index+1).padStart(2,'0')}</div><h2>${escapeText(eventNames[channel.event_type] || channel.event_type)}</h2><div class="equation">${sideHtml(channel.reactant_nodes)}<strong>↓</strong>${sideHtml(channel.product_nodes)}</div><h4>每次事件的类别净变化</h4><div class="net-list">${netHtml}</div><dl><dt>事件次数</dt><dd>${formatNumber(channel.event_count)}</dd><dt>帧范围</dt><dd>${channel.first_frame ?? '—'}–${channel.last_frame ?? '—'}</dd><dt>原子守恒</dt><dd>${(100*channel.mean_atom_conservation_ratio).toFixed(2)}%</dd><dt>质量误差</dt><dd>${Number(channel.mean_mass_error).toPrecision(3)} u</dd><dt>复合 / 未解析</dt><dd>${formatNumber(channel.uncertain_events)}</dd></dl>${channel.cross_carbon!=null ? `<h4>累计碳转移</h4><dl><dt>跨类别</dt><dd>${formatNumber(channel.cross_carbon)}</dd><dt>类别内</dt><dd>${formatNumber(channel.internal_carbon)}</dd></dl>` : ''}<h4>事件类型分项</h4>${types}<h4>实例事件</h4><div id="eventList">加载中</div><button id="moreEvents" class="button ghost" hidden>下一页事件</button>`;
  await loadEvents();
}

async function loadEvents(cursor=null) {
  const generation=state.evidenceGeneration=(state.evidenceGeneration || 0)+1;
  const signature = state.signature;
  const result = await api("/api/events",{...state.evidenceQuery,signature,limit:20,cursor});
  if (generation!==state.evidenceGeneration || signature !== state.signature || !byId("eventList")) return;
  byId("eventList").innerHTML = result.rows.map((event,index)=>`<div class="event-card" data-index="${index}" tabindex="0"><strong>帧 ${event.parent_frame_id} → ${event.child_frame_id}</strong><br><small>${escapeText(event.resolution)} · 原子守恒 ${(100*event.atom_conservation_ratio).toFixed(1)}%</small></div>`).join("") || '<p class="muted">没有匹配事件</p>';
  byId("eventList").querySelectorAll(".event-card").forEach(card=>{
    const show=()=>inspectEvent(result.rows[+card.dataset.index]);
    card.addEventListener("click",show);
    card.addEventListener("keydown",event=>{if(event.key==="Enter")show();});
  });
  const more = byId("moreEvents");
  more.hidden = !result.next_cursor;
  more.onclick = guard(()=>loadEvents(result.next_cursor));
}

function inspectEvent(event) {
  state.evidenceGeneration=(state.evidenceGeneration || 0)+1;
  state.inspectedEvent=event;
  const members = [["反应物",event.parent_frame_id,event.parent_fragment_ids],["产物",event.child_frame_id,event.child_fragment_ids]];
  byId("detail").innerHTML = `<div class="detail-kicker">EVENT EVIDENCE</div><h2>帧 ${event.parent_frame_id} → ${event.child_frame_id}</h2><span class="badge">${escapeText(event.resolution)}</span><dl><dt>共享原子</dt><dd>${event.shared_atom_count}</dd><dt>原子守恒</dt><dd>${(event.atom_conservation_ratio*100).toFixed(2)}%</dd><dt>质量误差</dt><dd>${Number(event.mass_error).toPrecision(5)}</dd><dt>元素差额</dt><dd>${escapeText(JSON.stringify(event.element_balance))}</dd></dl>${members.map(([title,frame,identifiers])=>`<h4>${title}片段 · ${identifiers.length}</h4><div>${identifiers.slice(0,100).map(identifier=>`<button class="button ghost fragment-button" data-frame="${frame}" data-fragment="${identifier}">#${identifier}</button>`).join("")}</div>${identifiers.length>100?'<p class="muted">仅显示前100个；在逐帧片段页查询其余成员。</p>':""}`).join("")}`;
  byId("detail").querySelectorAll(".fragment-button").forEach(button=>button.addEventListener("click",guard(()=>inspectFragment(button.dataset.frame,button.dataset.fragment))));
  if(state.inspectedChannel!=null && state.channels[state.inspectedChannel]?.reaction_signature===event.reaction_signature) {
    byId("detail").insertAdjacentHTML("afterbegin",'<button id="backChannel" class="button ghost">← 通道</button>');
    byId("backChannel").onclick=guard(()=>inspectReaction(state.inspectedChannel));
  }
}

async function inspectSpecies(species) {
  state.evidenceGeneration=(state.evidenceGeneration || 0)+1;
  state.species = species;
  highlight(null,species);
  const [carbon,rings,oxygen] = species.split("_");
  byId("detail").innerHTML = `<div class="detail-kicker">PSEUDO SPECIES</div><h2>${escapeText(speciesPresentation(species).name)}</h2><dl><dt>碳数分类</dt><dd>${escapeText(carbon)}</dd><dt>独立环分类</dt><dd>${escapeText(rings)}</dd><dt>氧数分类</dt><dd>${escapeText(oxygen)}</dd></dl><button id="focusSpecies" class="button primary full">聚焦此类别</button><button id="clearFocus" class="button secondary full">全部类别</button>`;
  byId("focusSpecies").onclick = guard(async()=>{byId("species").value=species;await loadNetwork();});
  byId("clearFocus").onclick = guard(async()=>{byId("species").value="";await loadNetwork();});
  await loadChart(species);
}

async function loadChart(species="") {
  if (!state.db) return;
  const generation=state.chartGeneration=(state.chartGeneration || 0)+1;
  const chart=await api("/api/timeseries",{...rangeQuery(),species_id:species || byId("species").value,points:500});
  if(generation!==state.chartGeneration)return;
  state.chart=chart;
  renderChart();
}

function renderChart() {
  if (!state.chart) return;
  const data=state.chart;
  const label=speciesPresentation(data.species_id);
  byId("chartTitle").textContent = data.species_id ? `${label.name} · ${label.description}` : "类别组成";
  byId("chartNote").textContent = `帧均值 · 桶宽 ${data.bucket_frames} 帧 · 缺席类别计零`;
  if (!data.rows.length) {byId("chart").innerHTML='<p class="muted">当前范围没有状态数据。</p>';return;}
  const metric=byId("chartMetric").value;
  const first=data.rows[0].first_frame,last=data.rows.at(-1).last_frame;
  const fraction=metric.endsWith("_fraction");
  const maximum=fraction ? 1 : Math.max(1,...data.rows.map(row=>metric==="population"?row.population_max:row[metric]));
  const horizontal=frame=>60+(frame-first)/Math.max(1,last-first)*1100;
  const vertical=value=>175-value/maximum*145;
  const points=data.rows.map(row=>`${horizontal((row.first_frame+row.last_frame)/2)},${vertical(row[metric])}`).join(" ");
  const envelope=metric==="population"?`<polygon points="${data.rows.map(row=>`${horizontal((row.first_frame+row.last_frame)/2)},${vertical(row.population_max)}`).concat([...data.rows].reverse().map(row=>`${horizontal((row.first_frame+row.last_frame)/2)},${vertical(row.population_min)}`)).join(" ")}" fill="#268b831a"/>`:"";
  const guides=Array.from({length:5},(_,index)=>{const value=maximum*index/4;return `<line x1="60" y1="${vertical(value)}" x2="1160" y2="${vertical(value)}" stroke="#e6ece1"/><text x="48" y="${vertical(value)+4}" text-anchor="end" font-size="11" fill="#859882">${fraction ? (value*100).toFixed(0)+'%' : value.toFixed(maximum<10?1:0)}</text>`;}).join("");
  const area=`60,175 ${points} 1160,175`;
  byId("chart").innerHTML=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 210" role="img" aria-label="${escapeText(data.species_id)} ${metric} 时间曲线">${guides}<polygon points="${area}" fill="#0085770c"/>${envelope}<polyline points="${points}" fill="none" stroke="#008577" stroke-width="2"/><text x="60" y="200" fill="#879580" font-size="11">${first}</text><text x="1160" y="200" text-anchor="end" fill="#879580" font-size="11">${last}</text></svg>`;
}

async function loadFrame(cursor=null) {
  if (!state.db) return;
  const frame=byId("frameNumber").value;
  const result=await api("/api/fragments",{db:state.db,start:frame,end:frame,limit:100,cursor});
  const frames=await api("/api/frames",{db:state.db,start:frame,end:frame,limit:1});
  const states=await api("/api/species-states",{db:state.db,start:frame,end:frame,limit:500});
  const source=frames.rows[0];
  byId("frameStates").innerHTML=(source?`<p class="muted">${escapeText(source.source_path)} · timestep ${source.timestep} · ${escapeText(source.status)} ${escapeText(source.error || "")}</p>`:"")+states.rows.map(row=>`<span class="badge">${escapeText(row.species_id)} · ${row.population}</span>`).join("");
  byId("frameRows").innerHTML=`<table><thead><tr><th>片段</th><th>类别</th><th>原子数</th><th>质量</th><th>环数</th><th>估计自由基</th></tr></thead><tbody>${result.rows.map(row=>`<tr data-fragment="${row.fragment_id}"><td>#${row.fragment_id}</td><td>${escapeText(row.species_id)}</td><td>${row.atom_count}</td><td>${row.molecular_mass.toFixed(2)}</td><td>${row.ring_count}</td><td>${row.radical_sites}</td></tr>`).join("")}</tbody></table>`;
  byId("frameRows").querySelectorAll("tr[data-fragment]").forEach(row=>row.addEventListener("click",guard(()=>inspectFragment(frame,row.dataset.fragment))));
  byId("moreFragments").hidden=!result.next_cursor;
  byId("moreFragments").onclick=guard(()=>loadFrame(result.next_cursor));
}

async function inspectFragment(frame,fragment,offset=0) {
  const generation=state.evidenceGeneration=(state.evidenceGeneration || 0)+1;
  const result=await api("/api/fragment",{db:state.db,frame,fragment,offset,limit:500});
  if(generation!==state.evidenceGeneration)return;
  byId("detail").innerHTML=`<div class="detail-kicker">FRAGMENT INSTANCE</div><h2>帧 ${frame} · 片段 #${fragment}</h2><div class="equation">${escapeText(result.species_id)}</div><dl><dt>原子数</dt><dd>${result.atom_count}</dd><dt>分子质量</dt><dd>${result.molecular_mass.toFixed(3)}</dd><dt>独立环数</dt><dd>${result.ring_count}</dd><dt>估计自由基位点</dt><dd>${result.radical_sites}</dd></dl><h4>Atom IDs · ${offset+1}–${offset+result.atom_ids.length} / ${result.atom_count}</h4><div class="atom-list">${result.atom_ids.join(" · ")}</div><button id="moreAtoms" class="button ghost" ${result.next_offset==null?"hidden":""}>下一页原子 →</button><h4>连接拓扑指纹</h4><div class="atom-list">${escapeText(result.topology_hash)}</div>`;
  byId("moreAtoms").onclick=guard(()=>inspectFragment(frame,fragment,result.next_offset));
  if(state.inspectedEvent) {
    byId("detail").insertAdjacentHTML("afterbegin",'<button id="backEvent" class="button ghost">← 实例事件</button>');
    byId("backEvent").onclick=()=>inspectEvent(state.inspectedEvent);
  }
}

function setTab(tab) {
  for (const name of ["graph","table","frames"]) {
    byId(`${name}Area`).hidden=name!==tab;
    byId(`${name}Tab`).classList.toggle("active",name===tab);
    byId(`${name}Tab`).setAttribute("aria-selected",String(name===tab));
  }
  if(tab==="frames") guard(()=>loadFrame())();
}

byId("toggleSetup").onclick=()=>byId("setup").showModal();
byId("closeSetup").onclick=()=>byId("setup").close();
byId("database").onkeydown=guard(event=>{if(event.key==="Enter")return openDatabase();});
byId("openDb").onclick=guard(openDatabase);
byId("runForm").onsubmit=guard(runPipeline);
byId("cancel").onclick=guard(async()=>{await api("/api/cancel",{},{task_id:state.task});byId("cancel").disabled=true;});
byId("filterForm").onsubmit=guard(async event=>{event.preventDefault();await Promise.all([loadNetwork(),loadChart()]);});
byId("firstPage").onclick=guard(()=>loadNetwork());
byId("nextPage").onclick=guard(()=>loadNetwork(state.networkCursor));
byId("chartMetric").onchange=renderChart;
byId("representation").onchange=guard(()=>{setTab('graph');return loadNetwork();});
byId("flowWeight").onchange=syncControls;
byId("toggleFilters").onclick=()=>{
  const filters=document.querySelector(".filters");
  filters.classList.toggle("collapsed");
  byId("toggleFilters").setAttribute("aria-expanded",String(!filters.classList.contains("collapsed")));
};
byId("resetFilters").onclick=guard(async()=>{byId("filterForm").reset();byId("start").value=state.summary?.first_frame || 0;byId("end").value=state.summary?.last_frame || 0;await Promise.all([loadNetwork(),loadChart()]);});
byId("exportSelection").onclick=()=>byId("representation").value.startsWith("path") ? exportPathway() : exportCoarseSelection();
byId("zoomIn").onclick=()=>zoom(.8);
byId("zoomOut").onclick=()=>zoom(1.25);
byId("fit").onclick=()=>{if(byId("representation").value.startsWith("path"))fitPathway();else fitNetwork();};
byId("pathStart").onclick=guard(()=>startPathway(+byId("pathOffset").value,true));
byId("pathNext").onclick=guard(()=>startPathway(state.pathway.next_offset));
byId("pathCancel").onclick=guard(()=>api("/api/pathway/cancel",{},{id:state.pathwayTask}));
byId("pathExport").onclick=exportPathway;
byId("graphScale").oninput=()=>{const scale=+byId("graphScale").value/100;state.pathwayScale=scale;if(byId("representation").value.startsWith("path"))updatePathwayProjection();else sizeGraph(byId("networkSvg"),scale);};
byId('pathTimeMetric').onchange=()=>{if(byId('representation').value.startsWith('path'))renderPathway();};
byId('pathDepth').oninput=()=>updatePathwayProjection();
byId('pathFocus').onclick=()=>focusPathway();
byId('pathRotationMode').onchange=()=>{state.pathwayRotation={yaw:0,pitch:.16};updatePathwayProjection();};
byId('pathBrowse').oninput=()=>{
  const scene=state.pathScene;if(!scene?.three)return;
  const span=scene.layout.width*Math.cos(state.pathwayRotation.yaw)*(state.pathwayScale||1);
  state.pathwayPan={...(state.pathwayPan||{y:0}),x:(.5-byId('pathBrowse').value/1000)*span};
  schedulePathwayProjection();
};
byId("loadFrame").onclick=guard(()=>loadFrame());
for(const name of ["graph","table","frames"])byId(`${name}Tab`).onclick=()=>setTab(name);
byId("exportSvg").onclick=()=>{
  const svg=byId("networkSvg");
  if(!svg)return;
  const copy=svg.cloneNode(true);
  const exportWidth=byId("representation").value.startsWith("path") ? state.pathwayWidth : state.bounds[2];
  copy.setAttribute("viewBox",`0 0 ${exportWidth} ${state.graphHeight}`);
  copy.setAttribute("width",String(exportWidth));copy.setAttribute("height",String(state.graphHeight));
  copy.querySelectorAll(".edge").forEach(edge=>edge.style.removeProperty("opacity"));
  const description=document.createElementNS("http://www.w3.org/2000/svg","desc");
  description.textContent=byId("graphSummary").textContent+"; selected projection only; full stoichiometry in database; cumulative flow is not yield or rate";
  copy.prepend(description);
  const url=URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(copy)],{type:"image/svg+xml"}));
  const anchor=document.createElement("a");anchor.href=url;anchor.download="reaction-subnetwork.svg";anchor.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
};
if(localStorage.getItem("coal-atlas-db"))byId("database").value=localStorage.getItem("coal-atlas-db");
if(localStorage.getItem("coal-atlas-task")){state.task=localStorage.getItem("coal-atlas-task");pollTask();}
if(typeof fetch==="function") {
  syncControls();
  globalThis.lucide?.createIcons();
  openDatabase().catch(error=>{notify(error.message,true);byId("graphArea").innerHTML='<div class="empty"><h3>数据集未连接</h3></div>';});
}
