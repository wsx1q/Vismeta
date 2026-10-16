const assert = require('node:assert/strict');
const ImportFiles = require('../static/import-utils.js');

const file = (name, path = name) => ({name, importPath: path, size: 10});
const entry = (name) => ({name, isFile: true, file: resolve => resolve(file(name))});
(async () => {
  const files = Array.from({length: 205}, (_, i) => file(`${i}.jpg`, `dataset/images/train/${i}.jpg`));
  const labels = files.map((_, i) => file(`${i}.txt`, `dataset/labels/train/${i}.txt`));
  const config = file('data.yaml', 'dataset/data.yaml');
  const plan = ImportFiles.plan([...files, ...labels, config]);
  assert.equal(plan.batches.length, 2);
  assert.equal(plan.batches[0].filter(f => ImportFiles.image.test(f.name)).length, 200);
  assert.equal(plan.batches[1].filter(f => ImportFiles.image.test(f.name)).length, 5);
  assert(plan.batches.every(batch => batch.includes(config)));
  assert(plan.batches[1].some(f => f.importPath === 'dataset/labels/train/204.txt'));
  assert(!plan.batches[1].some(f => f.name === '0.txt'));
  const mixed = ImportFiles.plan([file('a.jpg'), file('dataset.zip'), file('classes.txt'), file('b.jpg')]);
  assert.equal(mixed.batches.length, 3);
  assert(mixed.batches.every(batch => batch.some(f => f.name === 'classes.txt')));

  const chunks = [Array.from({length: 100}, (_, i) => entry(`${i}.jpg`)), [entry('last.jpg')], []];
  const directory = {name: 'images', isDirectory: true, createReader: () => ({readEntries: resolve => resolve(chunks.shift())})};
  const dropped = await ImportFiles.readEntry(directory, 'dataset/');
  assert.equal(dropped.length, 101);
  assert.equal(dropped[100].importPath, 'dataset/images/last.jpg');
  const fallback = file('fallback.jpg');
  assert.deepEqual(await ImportFiles.droppedFiles({items: [], files: [fallback]}), [fallback]);
  console.log('PASS: batching, annotation pairing, metadata, archive boundaries, multi-chunk folder drops and fallback');
})().catch(error => {console.error(error); process.exitCode = 1;});
