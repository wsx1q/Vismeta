const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const script = fs.readFileSync('static/index.html', 'utf8').match(/<script>\s*([\s\S]*?)<\/script>/)[1];
new vm.Script(script);
function element() {
  const children = [], slots = new Map();
  return {children, style: {}, textContent: '', innerHTML: '',
    appendChild(node) { children.push(node); },
    querySelector(selector) { if (!slots.has(selector)) slots.set(selector, element()); return slots.get(selector); }};
}
let box, request;
const context = vm.createContext({
  document: {createElement: element}, $: () => box, AbortController, setTimeout, clearTimeout,
  fetch: (...args) => request(...args), restorePathImport() {},
  rowButton: (text, onclick) => ({text, onclick}), enterProject() {}, editProjectName() {}, removeProject() {},
});
vm.runInContext(script.slice(script.indexOf('let projectListSeq = 0;'), script.indexOf('async function enterProject')), context);
const metadata = {projects: [{id: 'p1', name: 'Example', builtin: false}]};
const response = data => ({ok: true, json: async () => data});
(async () => {
  box = element(); let finishStats;
  request = async url => url.includes('summary=0') ? response(metadata) : new Promise(resolve => { finishStats = resolve; });
  const loading = context.loadProjects();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(box.children.length, 1, 'project entry must appear while statistics are pending');
  assert.equal(box.children[0].querySelector('.actions').children[0].text, '进入');
  assert.equal(box.children[0].querySelector('.stat').textContent, '正在统计图片和标注…');
  finishStats(response({projects: [{id: 'p1', images: 10, annotated: 2, shapes: 5}]})); await loading;
  assert.match(box.children[0].querySelector('.stat').textContent, /图片 10/);
  box = element(); request = async () => { throw new Error('offline'); }; await context.loadProjects();
  assert.match(box.textContent, /加载失败/); assert.equal(box.children[0].text, '重试');
  box = element(); request = async url => { if (url.includes('summary=0')) return response(metadata); throw new Error('timeout'); };
  await context.loadProjects();
  assert.match(box.children[0].querySelector('.stat').textContent, /可直接进入/);
  assert.equal(box.children.at(-1).text, '重试统计');
  console.log('PASS: project entries appear before statistics; failures show a retry and preserve entry buttons');
})().catch(error => { console.error(error); process.exitCode = 1; });
