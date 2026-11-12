/* Shared by the drag/drop UI and its Node tests; no browser framework required. */
(function (root) {
  const image = /\.(jpe?g|png|bmp|webp)$/i;
  const archive = /\.(zip|tar|tgz|tar\.gz|tar\.bz2|tar\.xz)$/i;
  const sidecar = /\.(txt|json|yaml|yml|names)$/i;
  const path = file => file.importPath || file.webkitRelativePath || file.name;
  const basename = file => path(file).split('/').pop();
  const isClasses = file => /\.(yaml|yml|names)$/i.test(file.name) || /^classes\.txt$/i.test(basename(file));
  const yoloMetadata = /^(?:data\.ya?ml|dataset\.ya?ml|classes\.txt|obj\.names)$/i;
  const splits = new Set(['train', 'val', 'valid', 'validation', 'test']);

  function selectMedia(files) {
    const media = files.filter(f => image.test(f.name) || archive.test(f.name));
    const paths = new Set(files.map(path));
    const metadata = new Set(), labels = new Set(), roots = new Set();
    const metadataRoots = new Set(), labelmeRoots = new Set();
    const dir = (parts, length) => parts.slice(0, length).join('/');
    for (const file of files) {
      const parts = path(file).split('/');
      if (yoloMetadata.test(parts.at(-1))) metadata.add(dir(parts, -1));
      if (/\.txt$/i.test(file.name) && !isClasses(file)) {
        parts.slice(0, -1).forEach((part, i) => { if (part === 'labels') labels.add(dir(parts, i + 1)); });
      }
    }
    for (const file of media) {
      if (!image.test(file.name)) continue;
      const name = path(file), parts = name.split('/');
      const i = parts.slice(0, -1).map(p => p.toLowerCase()).lastIndexOf('images');
      if (i < 0) continue;
      const parent = dir(parts, i), split = i > 0 && splits.has(parts[i - 1].toLowerCase());
      const root = split ? dir(parts, i - 1) : parent;
      const stem = name.replace(/\.[^.]+$/, '');
      if (paths.has(stem + '.json')) labelmeRoots.add(root);
      if (labels.has(parent ? parent + '/labels' : 'labels')
          || (!/^classes\./i.test(file.name) && paths.has(stem + '.txt'))) roots.add(root);
      if (metadata.has(parent) || metadata.has(root)) metadataRoots.add(root);
    }
    for (const root of metadataRoots) if (!labelmeRoots.has(root)) roots.add(root);
    const selected = media.filter(file => {
      if (archive.test(file.name)) return true;
      const parts = path(file).split('/');
      for (let i = parts.length - 1; i >= 0; i--) {
        if (!roots.has(dir(parts, i))) continue;
        return (i < parts.length - 1 && parts[i].toLowerCase() === 'images')
          || (i < parts.length - 2 && splits.has(parts[i].toLowerCase()) && parts[i + 1].toLowerCase() === 'images');
      }
      return true;
    });
    return {selected, skippedYolo: media.length - selected.length, totalMedia: media.length};
  }

  async function readEntry(entry, prefix = '') {
    const relative = prefix + entry.name;
    if (entry.isFile) {
      const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
      file.importPath = relative;
      return [file];
    }
    if (!entry.isDirectory) return [];
    const reader = entry.createReader();
    const files = [];
    // Chromium returns directory entries in chunks (often 100 at a time).
    for (;;) {
      const entries = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
      if (!entries.length) break;
      for (const child of entries) files.push(...await readEntry(child, relative + '/'));
    }
    return files;
  }

  async function droppedFiles(dataTransfer) {
    // Capture entries synchronously; DataTransfer becomes protected after the drop event.
    const entries = [...(dataTransfer.items || [])].map(item => item.webkitGetAsEntry?.()).filter(Boolean);
    const fallback = [...(dataTransfer.files || [])];
    if (!entries.length) return fallback;
    const files = [];
    for (const entry of entries) files.push(...await readEntry(entry));
    return files;
  }

  function plan(files, maxCount = 200, maxBytes = 400 * 1024 * 1024) {
    const {selected: media, skippedYolo, totalMedia} = selectMedia(files);
    const companions = files.filter(f => sidecar.test(f.name));
    const batches = [];
    let current = [], bytes = 0;
    const flush = () => { if (current.length) batches.push(current); current = []; bytes = 0; };
    for (const file of media) {
      if (archive.test(file.name)) { flush(); batches.push([file]); continue; }
      if (current.length >= maxCount || (current.length && bytes + file.size > maxBytes)) flush();
      current.push(file); bytes += file.size;
    }
    flush();
    return {
      batches: batches.map(batch => {
        const stems = new Set(batch.map(f => f.name.replace(/\.[^.]+$/, '')));
        return [...batch, ...companions.filter(f => isClasses(f) || stems.has(f.name.replace(/\.[^.]+$/, '')))];
      }),
      media, companions, skippedYolo, ignored: files.length - totalMedia - companions.length,
    };
  }

  const api = {path, image, archive, sidecar, isClasses, readEntry, droppedFiles, plan};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ImportFiles = api;
})(globalThis);
