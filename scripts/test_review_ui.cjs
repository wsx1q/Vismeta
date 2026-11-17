const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const html = fs.readFileSync('static/index.html', 'utf8');
const section = (start, end) => html.slice(html.indexOf(start), html.indexOf(end, html.indexOf(start)));
const box = (id, x1, y1, x2, y2, label = 'car') => ({id, label, shape_type:'rectangle', points:[[x1,y1],[x2,y2]]});
const shapes = [box('normal',100,100,150,150), box('duplicate',102,102,150,150),
  box('other-class',100,100,150,150,'truck'), box('flat',10,10,20,10), box('small',30,45,32,49)];
const ctx = vm.createContext({
  saved:shapes, reviewMode:true, reviewView:'overview', selectedShapeId:'normal',
  reviewCache:{key:null}, reviewChecked:new Set(),
  selectedShape:()=>shapes.find(s=>s.id===ctx.selectedShapeId)
});
vm.runInContext(section('function reviewQueue()', 'function refreshReview()')
  + section('function rectOf(', 'function rectHandles(')
  + section('function reviewRect()', 'function draw()'), ctx);
const plan=ctx.reviewQueue();
assert.deepEqual(Array.from(plan.queue,s=>s.id), ['flat','normal','duplicate','small','other-class']);
assert.equal(plan.reasons.get('normal'),'疑似重复');
assert(!plan.reasons.has('other-class'));
const focus=ctx.reviewRect();
assert(ctx.shapeVisible(shapes[4],focus), 'overview shows even distant boxes');
ctx.reviewView='raw'; assert(!ctx.shapeVisible(shapes[0],focus));
ctx.reviewView='focus'; assert(ctx.shapeVisible(shapes[1],focus)); assert(!ctx.shapeVisible(shapes[4],focus));
ctx.reviewChecked.add('normal');
shapes[0].points[0][0]=99; ctx.reviewQueue();
assert(!ctx.reviewChecked.has('normal'), 'edited geometry must be rechecked');
console.log('PASS: suspicious box ranking, raw/overview/focus visibility, recheck after edits');
