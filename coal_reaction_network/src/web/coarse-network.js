"use strict";

function coarseFilters() {
  return {...rangeQuery(),view:byId("view").value,species:byId("species").value,
    event_type:byId("eventType").value,resolution:byId("resolution").value,
    activity:byId("activity").value,
    minimum_count:byId("minCount").value,limit:byId("limit").value,
    weight:byId("flowWeight").value,coverage:byId("targetCoverage").value,
    include_internal:String(byId("includeInternal").checked),
    carbon_bin:byId("carbonFilter").value,ring_bin:byId("ringFilter").value,
    oxygen_bin:byId("oxygenFilter").value};
}

function populateCoarseBins(diagnostics) {
  for (const [name,element] of [["carbon","carbonFilter"],["ring","ringFilter"],["oxygen","oxygenFilter"]]) {
    const control=byId(element), previous=control.value;
    control.innerHTML='<option value="">全部</option>'+diagnostics.bins[name].map(bin=>
      `<option value="${escapeText(bin[0])}">${escapeText(bin[0])} [${bin[1]}, ${bin[2] ?? "∞"}]</option>`).join("");
    if ([...control.options].some(option=>option.value===previous)) control.value=previous;
  }
}

function renderDiagnostics(data) {
  const percent=value=>(100*(value || 0)).toFixed(2)+"%";
  const audit=data.audit;
  const totals=data.event_totals || {};
  byId("eventAccounting").innerHTML=`<div class="accounting-total"><span>非持续反应事件</span><strong>${formatNumber(totals.reactions)}</strong></div><div class="accounting-note"><span>类别数量变化</span><strong>${formatNumber(totals.population_changes)}</strong></div><div class="accounting-note"><span>类别数量不变的活动</span><strong>${formatNumber(totals.internal_activity)}</strong></div><div class="accounting-bar"><i style="width:${totals.reactions ? 100*totals.population_changes/totals.reactions : 0}%"></i></div><div class="accounting-note muted"><span>持续存在 · 不计为反应</span><span>${formatNumber(totals.persistence)}</span></div><div class="accounting-types">${data.activity.filter(row=>!row.event_type.includes('persistence')).map(row=>`<div class="accounting-note"><span>${escapeText(eventNames[row.event_type] || row.event_type)}</span><span>${formatNumber(row.event_count)}</span></div>`).join('')}</div><p class="muted">当前帧范围 · 原始事件 · 包含全部类别</p>`;
  const status=byId("auditStatus");
  const pending=!data.coarse_ready || !data.views_ready;
  status.textContent=pending ? "审计待完成" : audit?.unbalanced_events || audit?.unmatched_atom_occurrences ? "存在开放边界 / 不守恒事件" : "元素与类别数量校验通过";
  status.className=`badge ${pending || audit?.unbalanced_events || audit?.unmatched_atom_occurrences ? 'warn' : 'success'}`;
  byId("coarseDiagnostics").innerHTML=`<div class="diagnostic-meta"><span>${formatNumber(data.frame_count)} 帧 · ${data.species.length} 个出现类别</span><span>审计 ${formatNumber(audit?.audited_pairs)} 帧对 · 不平衡 ${formatNumber(audit?.unbalanced_events)} 事件 · 未匹配原子 ${formatNumber(audit?.unmatched_atom_occurrences)} 次</span></div><table><thead><tr><th>类别</th><th>C / R / O 区间</th><th class="numeric">平均片段数</th><th class="numeric">质量占比</th><th class="numeric">碳占比</th><th class="numeric">氧占比</th></tr></thead><tbody>${data.species.map(row=>{const label=speciesPresentation(row.species_id);return `<tr data-coarse-species="${escapeText(row.species_id)}"><td><span class="species-token" title="${escapeText(row.species_id)}">${escapeText(label.name)}</span></td><td>${escapeText(label.description)}</td><td class="numeric">${row.mean_population.toFixed(2)}</td><td class="numeric occupancy-cell">${percent(row.mean_mass_fraction)}<span class="frequency-bar"><i style="width:${100*row.mean_mass_fraction}%"></i></span></td><td class="numeric">${percent(row.mean_carbon_fraction)}</td><td class="numeric">${percent(row.mean_oxygen_fraction)}</td></tr>`;}).join('')}</tbody></table><p class="muted">占比按完整帧平均，缺席类别计零。R 为独立环数 E−V+1。${data.species_truncated ? '仅显示前 1000 类。' : ''}</p>`;
  byId("coarseDiagnostics").querySelectorAll("[data-coarse-species]").forEach(row=>row.onclick=guard(()=>inspectSpecies(row.dataset.coarseSpecies)));
}

function renderFlowGraph(data, diagnostics) {
  const legend=document.querySelector(".legend");
  if(legend)legend.innerHTML='<h4>氧区间</h4>'+diagnostics.bins.oxygen.map((bin,index)=>`<div><i style="--swatch:${["#008577","#3d6cc0","#bf7523","#ae5f91"][index%4]}"></i>${escapeText(bin[0])}</div>`).join('');
  if (!data.ready || !data.nodes.length) {
    byId("graphArea").innerHTML=`<div class="empty"><span>◈</span><h3>${data.ready ? "当前权重下暂无转移" : "物质流尚未构建"}</h3><p>${escapeText(data.message || "尝试事件次数、包含同类别转移或扩大时间窗。零碳流不等于零反应。")}</p></div>`;
    return;
  }
  const bins=diagnostics.bins;
  const rowHeight=Math.max(180,bins.oxygen.length*48+34);
  const height=Math.max(760,150+bins.ring.length*rowHeight);
  state.graphHeight=height;
  const columnWidth=1100/Math.max(1,bins.carbon.length);
  const palette=["#008577","#3d6cc0","#bf7523","#ae5f91"];
  const populations=new Map(diagnostics.species.map(row=>[row.species_id,row]));
  const positions=new Map(data.nodes.map(node=>{
    const carbon=bins.carbon.findIndex(bin=>bin[0]===node.carbon_bin);
    const rings=bins.ring.findIndex(bin=>bin[0]===node.ring_bin);
    const oxygen=bins.oxygen.findIndex(bin=>bin[0]===node.oxygen_bin);
    const fraction=populations.get(node.species_id)?.mean_mass_fraction || 0;
    return [node.species_id,{x:120+(carbon+.5)*columnWidth,y:120+rings*rowHeight+oxygen*48,
      radius:7+17*Math.sqrt(fraction),color:palette[oxygen%palette.length],node,fraction}];
  }));
  const grid=bins.carbon.map((bin,index)=>`<text x="${120+(index+.5)*columnWidth}" y="48" text-anchor="middle" font-size="15" font-weight="700" fill="#536678">${escapeText(bin[0])}</text><line x1="${120+index*columnWidth}" x2="${120+index*columnWidth}" y1="70" y2="${height-40}" stroke="#e6edf3"/>`).join("")+bins.ring.map((bin,index)=>`<text x="20" y="${124+index*rowHeight}" font-size="13" fill="#536678">${escapeText(bin[0])}</text><line x1="110" x2="1240" y1="${86+index*rowHeight}" y2="${86+index*rowHeight}" stroke="#e6edf3"/>`).join("");
  const maximum=Math.max(1,...data.edges.map(edge=>edge.weight));
  const edges=data.edges.map((edge,index)=>{
    const source=positions.get(edge.source),target=positions.get(edge.target);
    if (!source || !target) return "";
    let path;
    if(edge.source===edge.target) {
      path=`M ${source.x-7} ${source.y-source.radius} C ${source.x-55} ${source.y-65}, ${source.x+55} ${source.y-65}, ${source.x+7} ${source.y-source.radius-3}`;
    } else {
      const distance=Math.hypot(target.x-source.x,target.y-source.y);
      const horizontal=(target.x-source.x)/distance,vertical=(target.y-source.y)/distance;
      const bend=24;
      path=`M ${source.x+horizontal*source.radius} ${source.y+vertical*source.radius} Q ${(source.x+target.x)/2-vertical*bend} ${(source.y+target.y)/2+horizontal*bend} ${target.x-horizontal*(target.radius+5)} ${target.y-vertical*(target.radius+5)}`;
    }
    return `<path class="flow-edge edge" data-flow="${index}" d="${path}" fill="none" stroke="${source.color}" stroke-opacity=".43" stroke-width="${1+5*Math.sqrt(edge.weight/maximum)}" marker-end="url(#flowArrow)" tabindex="0" role="button"><title>${escapeText(edge.source)} → ${escapeText(edge.target)} · ${edge.weight.toFixed(2)} ${escapeText(data.weight)}</title></path>`;
  }).join("");
  const nodes=[...positions.entries()].map(([species,position])=>`<g class="node flow-node" data-species="${escapeText(species)}" tabindex="0" role="button" aria-label="${escapeText(species)}"><circle cx="${position.x}" cy="${position.y}" r="${position.radius}" fill="${position.color}" stroke="white" stroke-width="2.5"/><text x="${position.x+29}" y="${position.y+4}" font-size="11" fill="#435669">${escapeText(speciesPresentation(species).name)} · ${escapeText(position.node.oxygen_bin)}</text><title>${escapeText(speciesPresentation(species).description)} · ${escapeText(species)} · 平均质量占比 ${(position.fraction*100).toFixed(2)}%</title></g>`).join("");
  byId("graphArea").innerHTML=`<svg id="networkSvg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 ${height}" role="img" aria-label="C/R/O 类别物质流网络"><defs><marker id="flowArrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#71869a"/></marker></defs><rect width="1280" height="${height}" fill="#fcfdfb"/>${grid}${edges}${nodes}</svg>`;
  state.bounds=[0,0,1280,height];
  const svg=byId("networkSvg");
  for(const element of svg.querySelectorAll("[data-species]")) {
    const select=guard(()=>inspectSpecies(element.dataset.species));
    element.onclick=select;
    element.onkeydown=event=>{if(event.key==="Enter")select(event);};
  }
  for(const element of svg.querySelectorAll("[data-flow]")) {
    const select=guard(()=>inspectFlow(data.edges[+element.dataset.flow]));
    element.onclick=select;
    element.onkeydown=event=>{if(event.key==="Enter")select(event);};
  }
  bindGraphNavigation(svg);
}

async function inspectFlow(edge, cursor=null) {
  const generation=state.evidenceGeneration=(state.evidenceGeneration || 0)+1;
  const result=await api("/api/events",{...state.coarseQuery,flow_source:edge.source,flow_target:edge.target,limit:30,cursor});
  if(generation!==state.evidenceGeneration)return;
  byId("detail").innerHTML=`<div class="detail-kicker">ATOM-TRACED FLOW</div><h2>${escapeText(speciesPresentation(edge.source).name)} → ${escapeText(speciesPresentation(edge.target).name)}</h2><dl><dt>所选通道碳转移</dt><dd>${formatNumber(edge.carbon)}</dd><dt>氧转移</dt><dd>${formatNumber(edge.oxygen)}</dd><dt>质量转移</dt><dd>${edge.mass.toFixed(3)} u·次</dd></dl><p class="muted">证据沿用当前网络筛选，涵盖该类别对的全部通道。累计转移不是产率。</p><div id="flowEvidence">${result.rows.map((row,index)=>`<button class="button ghost full" data-flow-event="${index}">帧 ${row.parent_frame_id} → ${row.child_frame_id}<br>${sideHtml(row.reactant_nodes)} → ${sideHtml(row.product_nodes)}</button>`).join("") || '<p class="muted">没有匹配事件</p>'}</div><button id="moreFlowEvidence" class="button secondary" ${result.next_cursor ? "" : "hidden"}>下一页证据 →</button>`;
  byId("flowEvidence").querySelectorAll("[data-flow-event]").forEach(button=>button.onclick=()=>inspectEvent(result.rows[+button.dataset.flowEvent]));
  byId("moreFlowEvidence").onclick=guard(()=>inspectFlow(edge,result.next_cursor));
}

function exportCoarseSelection() {
  if(!state.networkQuery) {notify("尚未加载网络",true);return;}
  const payload={database:state.db,query:state.networkQuery,network:state.coarseNetwork || {channels:state.channels,next_cursor:state.networkCursor},
    definition:"C/R/O interval species; R=E-V+1; cumulative observed transfers, not rates or yields"};
  const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:"application/json"}));
  const anchor=document.createElement("a");anchor.href=url;anchor.download="cro-network-selection.json";anchor.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}
