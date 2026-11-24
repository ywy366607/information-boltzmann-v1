// Read-only: require an explicit local session file; never select business data.
import fs from 'node:fs';
import zlib from 'node:zlib';

const file = process.argv[2];
if (!file) {
  console.error('Usage: node read_full_session.mjs <session.jsonl[.zstd]>');
  process.exitCode = 2;
} else {
  const bytes = fs.readFileSync(file);
  const data = file.endsWith('.zstd') ? zlib.zstdDecompressSync(bytes) : bytes;
  process.stdout.write(data);
}
