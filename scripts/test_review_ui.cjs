const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const html = fs.readFileSync('static/index.html', 'utf8');
const section = (start, end) => html.slice(html.indexOf(start), html.indexOf(end, html.indexOf(start)));
const box = (id, x1, y1, x2, y2, label = 'car') => ({id, label, shape_type:'rectangle', points:[[x1,y1],[x2,y2]]});
const shapes = [
  box('upper-left', 100, 110, 150, 150),
  box('lower-left', 10, 210, 60, 250),
  box('upper-right', 200, 100, 250, 140),
  box('upper-middle', 140, 105, 190, 145),
  box('lower-right', 250, 200, 300, 240)
];
const strokes = [], handlers = {}, status = [];
const ctx = vm.createContext({
  saved:shapes, reviewMode:true, reviewView:'overview', selectedShapeId:null, selectedLabel:'',
  selectedVertex:null, reviewSignatures:new Map(), reviewOrder:[], reviewChecked:new Set(), history:[],
  $:()=>({}), renderShapes:()=>{}, setStatus:message=>status.push(message),
  setReviewView:view=>{ ctx.reviewView=view; },
  focusReviewBox:id=>{ ctx.reviewView='focus'; ctx.selectShape(id); return true; },
  selectedShape:()=>shapes.find(s=>s.id===ctx.selectedShapeId),
  ctx:{clearRect:()=>{ strokes.length=0; }},
  photo:{naturalWidth:1000}, overlay:{addEventListener:(type, fn)=>{ handlers[type]=fn; }},
  labelColor:()=>'#0088ff', hexA:()=>'dimmed',
  drawShape:(shape, color)=>strokes.push([shape.id, color]),
  shapeTop:()=>[0, 0], drawTag:()=>{}, drawHandles:()=>{},
  current:{polygons:[], boxes:[]}, points:[], polyDraft:[], draftBox:null,
  clearTimeout:()=>{}, promptTimer:null, pendingSelect:null, drag:null,
  index:0, promptMode:()=>'point', toImage:()=>[150, 115],
  shapeAt:()=>shapes.find(s=>s.id==='upper-middle'), showTab:()=>{}
});
vm.runInContext(section('function selectShape(', 'function refreshReview()')
  + section('function reviewBox(', '$("review-mode").onclick')
  + section('function rectOf(', 'function rectHandles(')
  + section('function deleteShape(', '// ---------- 图片列表')
  + section('function highlightOf(', 'function drawShape(')
  + section('function reviewRect()', 'async function segment(')
  + section('overlay.addEventListener("dblclick"', 'window.addEventListener("mousemove"'), ctx);
assert.deepEqual(Array.from(ctx.reviewQueue(),s=>s.id),
  ['upper-left','upper-middle','upper-right','lower-left','lower-right']);
assert(ctx.reviewBox(1)); assert.equal(ctx.selectedShapeId,'upper-left');
assert(ctx.reviewBox(1)); assert.equal(ctx.selectedShapeId,'upper-middle');
assert(ctx.reviewBox(-1)); assert.equal(ctx.selectedShapeId,'upper-left');
const focus=ctx.reviewRect();
ctx.reviewView='overview';
assert(ctx.shapeVisible(shapes[4],focus), 'overview shows even distant boxes');
ctx.reviewView='raw'; assert(!ctx.shapeVisible(shapes[0],focus));
ctx.reviewView='focus'; assert(ctx.shapeVisible(shapes[3],focus)); assert(!ctx.shapeVisible(shapes[4],focus));
ctx.reviewChecked.add('upper-left');
shapes[0].points[0][0]=99; ctx.reviewQueue();
assert(!ctx.reviewChecked.has('upper-left'), 'edited geometry must be rechecked');

ctx.reviewView='overview';
handlers.dblclick({});
assert.equal(ctx.selectedShapeId,'upper-middle');
assert.deepEqual(strokes.at(-1), ['upper-middle','#ffdf5d'], 'draw selection in yellow above the other boxes');

ctx.deleteShape('upper-middle');
assert.equal(ctx.selectedShapeId,'upper-right', 'deletion advances by position, even if the next box is earlier in storage');
assert.equal(status.at(-1),'已删除');
ctx.focusReviewBox('lower-right');
ctx.deleteShape('lower-right');
assert.equal(ctx.selectedShapeId,'lower-left');
ctx.deleteShape('upper-right');
assert.equal(ctx.selectedShapeId,'lower-left', 'deleting another box keeps the current selection');
ctx.deleteShape('lower-left');
assert.equal(ctx.selectedShapeId,'upper-left');
ctx.deleteShape('upper-left');
assert.equal(ctx.selectedShapeId,null);
assert.equal(ctx.reviewView,'overview');

shapes.push(box('anchor', 200, 100, 240, 140), box('left', 150, 115, 190, 155),
  box('next-row', 100, 130, 140, 170));
assert.deepEqual(Array.from(ctx.reviewQueue(),s=>s.id), ['left','anchor','next-row']);
ctx.focusReviewBox('anchor');
ctx.deleteShape('anchor');
assert.equal(ctx.selectedShapeId,'next-row');
assert.deepEqual(Array.from(ctx.reviewQueue(),s=>s.id), ['left','next-row'], 'deleting a row anchor keeps the scan order');
console.log('PASS: spatial order, visibility, yellow double-click selection, and deletion navigation');
