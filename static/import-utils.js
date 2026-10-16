/* Shared by the drag/drop UI and its Node tests; no browser framework required. */
(function (root) {
  const image = /\.(jpe?g|png|bmp|webp)$/i;
  const archive = /\.(zip|tar|tgz|tar\.gz|tar\.bz2|tar\.xz)$/i;
  const sidecar = /\.(txt|json|yaml|yml|names)$/i;
  const path = file => file.importPath || file.webkitRelativePath || file.name;
  const basename = file => path(file).split('/').pop();
  const isClasses = file => /\.(yaml|yml|names)$/i.test(file.name) || /^classes\.txt$/i.test(basename(file));

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
    const media = files.filter(f => image.test(f.name) || archive.test(f.name));
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
      media, companions, ignored: files.length - media.length - companions.length,
    };
  }

  const api = {path, image, archive, sidecar, isClasses, readEntry, droppedFiles, plan};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ImportFiles = api;
})(globalThis);
