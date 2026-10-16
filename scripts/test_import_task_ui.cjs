const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const script=fs.readFileSync('static/index.html','utf8').match(/<script>\s*([\s\S]*?)<\/script>/)[1];
new vm.Script(script);
const elements=new Map();
const $=id=>{if(!elements.has(id))elements.set(id,{hidden:true,dataset:{},classList:{add(){this.shown=true;},remove(){this.shown=false;}},setAttribute(k,v){this[k]=v;}});return elements.get(id);};
let enters=0,polls=0,mayLeave=false;
const sandbox={$,fetch:async()=>({json:async()=>({jobs:[]})}),projectId:'other',projectName:'Other',store:key=>key==='path-import:task-project'?'job-123':key==='path-import-source:task-project'?'/data/migrate/demo':'',
 leaveCurrent:async()=>mayLeave,enterProject:async(id)=>{enters++;sandbox.projectId=id;},closeImport(){},updatePathModeHelp(){},renderPathProgress(){},monitorPathImport(){polls++;}};
const ctx=vm.createContext(sandbox);
vm.runInContext(script.slice(script.indexOf('let pathImportBusy = false;'),script.indexOf('function pathImportControls')),ctx);
(async()=>{
 ctx.updateImportTask({id:'job-123',status:'running',phase:'importing',processed:5,total:20},'task-project','安全带');
 assert.equal($('import-task').hidden,false);
 assert.match($('import-task').textContent,/25%.*5\/20/);
 assert.match($('import-task').title,/安全带/);
 await $('import-task').onclick(); assert.equal(enters,0,'respect unsaved changes');
 mayLeave=true;await $('import-task').onclick();
 assert.equal(enters,1);assert.equal($('pathbox').classList.shown,true);
 assert.equal($('server-path').value,'/data/migrate/demo');
 ctx.updateImportTask({id:'job-123',status:'complete',result:{count:20}},'task-project','安全带');
 assert.match($('import-task').textContent,/导入完成.*20/);
 ctx.updateImportTask({id:'job-123',status:'failed'},'task-project','安全带');
 assert.equal($('import-task').dataset.status,'failed');
 vm.runInContext('pathImportTask = null;',ctx);
 await ctx.restorePathImport([{id:'task-project',name:'安全带'}]);
 await ctx.restorePathImport([{id:'task-project',name:'安全带'}]);
 assert.equal(polls,1,'recover one poller after refresh');
 console.log('PASS: persistent progress badge, project navigation, unsaved-change guard, source path, completion/failure and refresh recovery');
})().catch(e=>{console.error(e);process.exitCode=1;});
