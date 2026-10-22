"use strict";

const assert=require("node:assert/strict");
const fs=require("node:fs");
const path=require("node:path");
const vm=require("node:vm");

const elements=new Map();
function element(identifier) {
  if(!elements.has(identifier))elements.set(identifier,{
    innerHTML:"",value:"",dataset:{},clientWidth:900,style:{removeProperty(){}},
    classList:{add(){},remove(){},toggle(){}},
    addEventListener(){},setAttribute(){},getAttribute(){return "0 0 1280 760";},querySelectorAll(){return [];}
  });
  return elements.get(identifier);
}
const context=vm.createContext({
  console,
  document:{getElementById:element,querySelector:element},
  localStorage:{getItem(){return null;}}
});
for(const name of ["graph-ui.js","pathway-layout.js","pathway-camera.js","pathway-scene.js","pathway-view.js","coarse-network.js","app.js"]) {
  const filename=path.join(__dirname,"../src/web",name);
  vm.runInContext(fs.readFileSync(filename,"utf8"),context,{filename});
}
vm.runInContext(`
  const species="C1-5_R0_O0";
  const diagnostics={bins:{carbon:[["C1-5",1,5]],ring:[["R0",0,0]],oxygen:[["O0",0,0]]},
    species:[{species_id:species,mean_mass_fraction:1,mean_carbon_fraction:1,mean_oxygen_fraction:0,mean_population:1.5}],
    activity:[],frame_count:2,largest_category:species,coarse_ready:true,views_ready:true,
    event_totals:{reactions:1,population_changes:1,internal_activity:0,persistence:0},
    audit:{audited_pairs:1,unbalanced_events:0,unmatched_atom_occurrences:0}};
  const channel={reactant_nodes:{[species]:1},product_nodes:{[species]:2},event_count:1,
    uncertain_events:0,mean_atom_conservation_ratio:1};
  state.channels=[channel];
  state.coarseNetwork={ready:true,channels:[channel],weight:"carbon",
    nodes:[{species_id:species,carbon_bin:"C1-5",ring_bin:"R0",oxygen_bin:"O0"}],
    edges:[{source:species,target:species,weight:4,carbon:4,oxygen:0,mass:48.044}]};
  renderFlowGraph(state.coarseNetwork,diagnostics);
`,context);
assert.match(element("graphArea").innerHTML,/data-flow="0"/);
assert.match(element("graphArea").innerHTML,/flowArrow/);
assert.match(element("graphArea").innerHTML,/C1-5_R0_O0/);
vm.runInContext("renderDiagnostics(diagnostics);renderGraph();renderChannelTable();",context);
assert.match(element("coarseDiagnostics").innerHTML,/100.00%/);
assert.match(element("tableArea").innerHTML,/2 ×/);
assert.match(element("tableArea").innerHTML,/C1-5_R0_O0/);
assert.match(element("eventAccounting").innerHTML,/类别数量变化/);
assert.match(element("graphArea").innerHTML,/networkSvg/);
vm.runInContext("state.displayScale=.3;explicitGraphScale(.8);",context);
assert.equal(vm.runInContext("state.displayScale",context),.24);
console.log("C/R/O render smoke passed (mock DOM; not browser visual validation).");
