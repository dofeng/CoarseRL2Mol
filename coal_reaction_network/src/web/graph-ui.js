"use strict";

function speciesPresentation(identifier) {
  const bins=state.coarseDiagnostics?.bins;
  const parts=String(identifier).split("_");
  if(!bins)return {name:identifier,description:identifier,color:"#326aa8"};
  const axes=[bins.carbon,bins.ring,bins.oxygen];
  const indices=axes.map((axis,index)=>axis.findIndex(bin=>bin[0]===parts[index]));
  if(indices.some(index=>index<0))return {name:identifier,description:identifier,color:"#326aa8"};
  const number=(indices[0]*axes[1].length+indices[1])*axes[2].length+indices[2]+1;
  const description=axes.map((axis,index)=>{
    const bin=axis[indices[index]], prefix=["C","R","O"][index];
    return bin[2]==null ? `${prefix}≥${bin[1]}` : bin[1]===bin[2] ? `${prefix}=${bin[1]}` : `${prefix} ${bin[1]}–${bin[2]}`;
  }).join(" · ");
  return {name:`P${String(number).padStart(3,"0")}`,description,
    color:["#64748b","#008577","#3d6cc0","#ae5f91","#bf7523"][indices[0]%5]};
}

function readableSpeciesLabels(svg) {
  if(!svg)return;
  svg.querySelectorAll("text").forEach(text=>{
    const value=text.textContent;
    if(!value.includes("_"))return;
    const label=speciesPresentation(value);
    text.textContent=`${label.name} · ${label.description}`;
    const title=document.createElementNS("http://www.w3.org/2000/svg","title");
    title.textContent=value;text.append(title);
  });
}

function sizeGraph(svg, scale=1) {
  if(!svg)return;
  const box=svg.getAttribute("viewBox").split(/\s+/).map(Number);
  svg.style.width=`${box[2]*scale}px`;
  svg.style.height=`${box[3]*scale}px`;
  svg.style.maxWidth="none";
  state.displayScale=scale;
}

function explicitGraphScale(factor) {
  if(state.pathway && byId("representation").value.startsWith("path")) {
    state.pathwayScale=Math.max(.05,Math.min(4,(state.pathwayScale || 1)*factor));
    updatePathwayProjection();return;
  }
  sizeGraph(byId("networkSvg"),Math.max(.05,Math.min(2.5,(state.displayScale || 1)*factor)));
}
