"use strict";
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const context=vm.createContext({});
vm.runInContext(fs.readFileSync(path.join(__dirname,'../src/web/pathway-layout.js'),'utf8'),context);
const result=vm.runInContext(`
  const data={states:[
    {id:'s0',kind:'state',tracked:10,first_frame:0,last_frame:8},
    {id:'s1',kind:'state',tracked:8,first_frame:9,last_frame:20},
    {id:'s2',kind:'state',tracked:2,first_frame:9,last_frame:20},
    {id:'s3',kind:'state',tracked:10,first_frame:21,last_frame:30}
  ],events:[{id:'e0',kind:'event',frame:8,child_frame:9},{id:'e1',kind:'event',frame:20,child_frame:21}],links:[
    {source:'s0',target:'e0',tracked:10},{source:'e0',target:'s1',tracked:8},{source:'e0',target:'s2',tracked:2},
    {source:'s1',target:'e1',tracked:8},{source:'s2',target:'e1',tracked:2},{source:'e1',target:'s3',tracked:10}
  ]};
  const before=JSON.stringify(data);
  const layout=compactProcessLayout(data);
  const project=node=>({sx:node.x,sy:node.y});
  const source=layout.nodes.find(node=>node.id==='s0'),target=layout.nodes.find(node=>node.id==='e0');
  ({layout,data,before,after:JSON.stringify(data),curve:spatialCurve(source,target,project),again:compactProcessLayout(data),real:compactProcessLayout(data,{metric:'frames'}),empty:compactProcessLayout({states:[],events:[],links:[]})});
`,context);
assert.deepEqual(result.layout,result.again);
assert.equal(result.before,result.after);
assert.match(result.curve,/ C /);
assert.doesNotMatch(result.curve,/NaN|Infinity/);
const nodes=new Map(result.layout.nodes.map(node=>[node.id,node]));
for(const link of result.data.links)assert.ok(nodes.get(link.source).endX<nodes.get(link.target).x);
assert.equal(nodes.get('s0').lane,nodes.get('s1').lane);
assert.equal(nodes.get('s1').lane,nodes.get('s3').lane);
assert.notEqual(nodes.get('s1').lane,nodes.get('s2').lane);
assert.notEqual(nodes.get('s1').z,nodes.get('s2').z);
assert.ok(nodes.get('s0').endX>nodes.get('s0').x);
assert.ok(result.real.width>result.layout.width);
assert.equal(result.empty.nodes.length,0);
const numbers=result.curve.match(/-?\d+(?:\.\d+)?/g).map(Number);
assert.ok(numbers[0]<=numbers[2] && numbers[2]<=numbers[4] && numbers[4]<=numbers[6]);
if(fs.existsSync('/tmp/coal-process-real.json')) {
  context.real=JSON.parse(fs.readFileSync('/tmp/coal-process-real.json','utf8'));
  const layouts=vm.runInContext('[compactProcessLayout(real),compactProcessLayout(real.compact),compactProcessLayout(real.overview)]',context);
  for(const [index,layout] of layouts.entries()) {
    const data=[context.real,context.real.compact,context.real.overview][index];
    const mapping=new Map(layout.nodes.map(node=>[node.id,node]));
    for(const link of data.links)assert.ok(mapping.get(link.source).endX<mapping.get(link.target).x,JSON.stringify(link));
    for(const node of layout.nodes)assert.ok(Number.isFinite(node.x+node.y+node.z));
  }
}
console.log('PASS: chronological lanes, split/merge inheritance, residence, monotone curves, frame metric and immutable evidence');
