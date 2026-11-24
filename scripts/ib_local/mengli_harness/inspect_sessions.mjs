// Read-only inspection of this project's local mother sessions.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const baseDir = process.argv[2] ?? fileURLToPath(
  new URL('../../../data/harness/dsh', import.meta.url),
);
function walk(dir) {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap(entry => {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) return walk(full);
    return entry.isFile() && entry.name.endsWith('.jsonl') ? [full] : [];
  });
}
const files = walk(baseDir);
console.log('Local session JSONL files:', files.length);
for (const file of files) console.log(file);
