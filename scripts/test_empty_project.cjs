const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync('static/index.html', 'utf8');
const script = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
new vm.Script(script);
function block(start, end) { return script.slice(script.indexOf(start), script.indexOf(end, script.indexOf(start))); }
const elements = new Map();
const el = id => { if (!elements.has(id)) elements.set(id, {hidden:false,textContent:'old',title:'old'}); return elements.get(id); };
let annotationResolve;
const sandbox = {console, clearTimeout, $:el,
  photo:{style:{width:'310px',height:'760px'},naturalWidth:105,removeAttribute(name){delete this[name];}},
  stageWrap:{scrollLeft:30,scrollTop:20},overlay:{width:310,height:760},
  imageLoadSeq:0,index:0,projectId:'old',items:[{name:'old.jpg'}],busy:false,
  points:[],saved:[],listEl:{children:[]},promptTimer:null,
  draw(){},renderShapes(){},setTexts(){},hydrate:x=>x,shapesKey:()=> '[]',ensureLabel:()=>false,persistLabels(){},
  toastCenter(t){sandbox.toast=t;},setStatus(t){sandbox.status=t;},leaveCurrent:async()=>true,
  fetch:()=>new Promise(resolve=>{annotationResolve=resolve;}),
};
vm.createContext(sandbox);
vm.runInContext(block('function wipeView() {','async function showProjects()'),sandbox);
vm.runInContext(block('async function openAt(i) {','window.addEventListener("beforeunload"'),sandbox);
(async()=>{
  const pending=sandbox.openAt(0);
  await Promise.resolve();
  sandbox.wipeView();
  sandbox.projectId='empty';
  annotationResolve({json:async()=>({shapes:[{label:'stale'}]})});
  await pending;
  assert.equal(el('stage').hidden,true);
  assert.equal(sandbox.photo.style.width,'');
  assert.equal(sandbox.photo.style.height,'');
  assert.equal(el('zoom').textContent,'适应窗口');
  assert.equal(sandbox.stageWrap.scrollLeft,0);
  assert.equal(sandbox.saved.length,0);
  assert.equal(sandbox.photo.src,undefined,'stale annotation response must not reload the previous image');
  assert(html.includes('#stage[hidden]'));
  console.log('PASS: empty canvas clears dimensions, zoom, scroll and rejects stale image response; inline JS syntax valid');
})().catch(e=>{console.error(e);process.exitCode=1;});
