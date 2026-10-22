"use strict";

function compactProcessLayout(data, options={}) {
  const timestamp=node=>node.kind==="event" ? (node.frame+node.child_frame)/2 : node.first_frame;
  const nodes=[...data.states,...data.events].map(node=>({...node,time:timestamp(node)}));
  const byIdentifier=new Map(nodes.map(node=>[node.id,node]));
  const incoming=new Map(nodes.map(node=>[node.id,[]]));
  for(const link of data.links)incoming.get(link.target)?.push(link);
  const times=[...new Set(nodes.flatMap(node=>node.kind==="state" ? [node.time,node.last_frame] : [node.time,node.frame,node.child_frame]))].sort((left,right)=>left-right);
  if(!times.length)return {nodes:[],width:1000,height:650,metric:options.metric||'compact',ticks:[]};
  const ranks=new Map(times.map((time,index)=>[time,index]));
  const unit=Math.max(1,Math.min(...times.slice(1).map((time,index)=>time-times[index])));
  const position=time=>110+(options.metric==="frames" ? (time-times[0])/unit : ranks.get(time))*170;
  const reservations=new Map();
  const available=(lane,time)=>!reservations.has(lane) || reservations.get(lane)<time;
  const allocate=(preferred,time)=>{
    for(let distance=0;;distance++)for(const lane of distance ? [preferred+distance,preferred-distance] : [preferred])if(available(lane,time))return lane;
  };
  nodes.sort((left,right)=>left.time-right.time || (right.tracked||0)-(left.tracked||0) || left.id.localeCompare(right.id));
  for(const node of nodes) {
    const parents=incoming.get(node.id).slice().sort((left,right)=>(right.tracked||0)-(left.tracked||0) || left.source.localeCompare(right.source));
    const parent=parents.map(link=>byIdentifier.get(link.source)).find(candidate=>candidate && candidate.lane!==undefined);
    let depthParent=parent;
    let preferred=parent?.lane ?? 0;
    if(node.aggregate && parent?.process) {
      const sameCategory=incoming.get(parent.id).map(link=>byIdentifier.get(link.source)).find(candidate=>candidate?.species===node.species);
      if(sameCategory){preferred=sameCategory.lane;depthParent=sameCategory;}
    }
    node.lane=allocate(preferred,node.time);
    reservations.set(node.lane,node.kind==="state" ? node.last_frame : node.time);
    node.branchDepth=(depthParent?.branchDepth||0)+(parent && node.lane!==preferred ? 1 : 0);
    node.z=node.context ? -180 : node.branchDepth*150;
    node.x=position(node.time);
    node.endX=node.kind==="state" ? position(node.last_frame) : node.x;
  }
  const minimum=Math.min(0,...nodes.map(node=>node.lane));
  const maximum=Math.max(0,...nodes.map(node=>node.lane));
  for(const node of nodes)node.y=140+(node.lane-minimum)*115;
  const height=Math.max(650,(maximum-minimum)*115+280);
  const stretch=data.events.some(node=>node.process) ? Math.max(1,Math.min(3,height/900)) : 1;
  const stretchedPosition=time=>110+(position(time)-110)*stretch;
  for(const node of nodes){node.x=stretchedPosition(node.time);node.endX=node.kind==='state' ? stretchedPosition(node.last_frame) : node.x;}
  const width=Math.max(1000,stretchedPosition(times.at(-1))+220);
  const frameTimes=times.filter(time=>Number.isInteger(time));
  return {nodes,width,height,metric:options.metric||"compact",ticks:frameTimes.map(time=>({time,x:stretchedPosition(time),y:65,z:0}))};
}

function spatialProjection(node, layout, width, height, rotation, scale, pan) {
  const horizontal=node.x-layout.width/2;
  const vertical=node.y-layout.height/2;
  const rotatedX=horizontal*Math.cos(rotation.yaw)+node.z*Math.sin(rotation.yaw);
  const rotatedZ=-horizontal*Math.sin(rotation.yaw)+node.z*Math.cos(rotation.yaw);
  return {sx:width/2+rotatedX*scale+pan.x,
    sy:height/2+(vertical*Math.cos(rotation.pitch)-rotatedZ*Math.sin(rotation.pitch))*scale+pan.y,
    depth:vertical*Math.sin(rotation.pitch)+rotatedZ*Math.cos(rotation.pitch)};
}

function spatialCurve(source,target,project) {
  const startNode={...source,x:source.kind==="state" ? source.endX??source.x : source.x};
  const distance=target.x-startNode.x;
  const bow=Math.abs(target.y-startNode.y)<1 ? -Math.min(24,Math.abs(distance)*.12) : 0;
  const first=project({...startNode,x:startNode.x+distance*.42,y:startNode.y+bow,z:startNode.z+Math.abs(bow)});
  const second=project({...target,x:target.x-distance*.42,y:target.y+bow,z:target.z+Math.abs(bow)});
  const start=project(startNode),end=project(target);
  const length=Math.hypot(end.sx-second.sx,end.sy-second.sy)||1;
  const trim=Math.min(target.radius ? target.radius+3 : 0,length*.6);
  return `M ${start.sx} ${start.sy} C ${first.sx} ${first.sy}, ${second.sx} ${second.sy}, ${end.sx-(end.sx-second.sx)/length*trim} ${end.sy-(end.sy-second.sy)/length*trim}`;
}
