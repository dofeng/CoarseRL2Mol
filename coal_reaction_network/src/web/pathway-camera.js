"use strict";

function schedulePathwayProjection() {
  if(state.pathRenderPending)return;
  state.pathRenderPending=true;
  requestAnimationFrame(()=>{state.pathRenderPending=false;updatePathwayProjection(Boolean(state.pathDrag));});
}

function bindPathwayCamera(three,width,height) {
  const area=byId("graphArea");
  const schedule=schedulePathwayProjection;
  const clear=()=>{if(state.pathDrag){state.pathDrag=null;schedule();}};
  area.onpointerdown=event=>{
    if(event.button!==0)return;
    state.pathDrag={x:event.clientX,y:event.clientY,...(state.pathwayRotation||{yaw:0,pitch:.16}),
      pan:{...(state.pathwayPan||{x:0,y:0})},shift:event.shiftKey,node:Boolean(event.target.closest("[data-path-node]"))};
    state.pathSuppressClick=false;
    if(!state.pathDrag.node)area.setPointerCapture(event.pointerId);
  };
  area.onpointermove=event=>{
    const drag=state.pathDrag;if(!drag || !(event.buttons&1) || !three || drag.node)return;
    if(Math.hypot(event.clientX-drag.x,event.clientY-drag.y)<4)return;
    state.pathSuppressClick=true;
    if(drag.shift)state.pathwayPan={x:drag.pan.x+event.clientX-drag.x,y:drag.pan.y+event.clientY-drag.y};
    else {
      const limit=byId('pathRotationMode').value==='free' ? Math.PI : .4;
      state.pathwayRotation={yaw:Math.max(-limit,Math.min(limit,drag.yaw+(event.clientX-drag.x)*.003)),pitch:Math.max(-.55,Math.min(.55,drag.pitch+(event.clientY-drag.y)*.003))};
    }
    schedule();
  };
  area.onpointerup=clear;area.onpointercancel=clear;area.onlostpointercapture=clear;
  area.onwheel=event=>{
    if(!three && event.shiftKey){event.preventDefault();area.scrollLeft+=event.deltaY || event.deltaX;return;}
    if(!three || !state.pathDrag)return;
    event.preventDefault();state.pathSuppressClick=true;
    const svg=byId("networkSvg"),box=svg.getBoundingClientRect();
    const pointerX=(event.clientX-box.left)*width/box.width-width/2;
    const pointerY=(event.clientY-box.top)*height/box.height-height/2;
    const previous=state.pathwayScale||1;
    const next=Math.max(.05,Math.min(4,previous*Math.exp(-event.deltaY*.0015)));
    const ratio=next/previous,pan=state.pathwayPan||{x:0,y:0};
    state.pathwayPan={x:pointerX-(pointerX-pan.x)*ratio,y:pointerY-(pointerY-pan.y)*ratio};
    state.pathwayScale=next;byId("graphScale").value=Math.round(next*100);
    state.pathDrag.pan={...state.pathwayPan};state.pathDrag.x=event.clientX;state.pathDrag.y=event.clientY;
    Object.assign(state.pathDrag,state.pathwayRotation);
    schedule();
  };
  if(!state.pathGlobalRelease) {
    window.addEventListener("pointerup",clear);window.addEventListener("blur",clear);
    state.pathGlobalRelease=true;
  }
}
