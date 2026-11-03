// Decode SAM 3.1 video output into a compact binary file for the Python renderer.
//
// Meta's TypeScript parser (@meta-sam/parser) decodes masks far faster than the pure
// Python port, so the render pipeline shells out to this script once per concept.
//
// Usage: node decode.mjs <raw_sam_output.txt> <out.bin>
//
// Output: "SCM1" | u32 count | count x record
// record: u32 frame | u32 objectId | i32 left, top, right, bottom | u32 h | u32 w
//         | ceil(h*w/8) bytes, row-major bits, MSB first (numpy.unpackbits order)
import { readFileSync, writeFileSync } from 'node:fs';
import { decodeMaskToRaster, parseVideoStream, recordsOfKind } from '@meta-sam/parser';

const [, , inPath, outPath] = process.argv;
if (!inPath || !outPath) {
  console.error('usage: node decode.mjs <raw.txt> <out.bin>');
  process.exit(2);
}

const text = readFileSync(inPath, 'utf8');

async function* events() {
  yield { type: 'response.output_text.done', item_id: 'saycut', output_index: 0, content_index: 0, text };
  yield { type: 'response.completed' };
}

const result = await parseVideoStream(events()).finalResult;

// Highest revision per (frame, object).
const latest = new Map();
for (const record of recordsOfKind(result.records, 'mask')) {
  if (!record.frame) continue;
  const key = `${record.frame.frameIndex}:${record.objectId}`;
  const prev = latest.get(key);
  if (!prev || record.revision > prev.revision) latest.set(key, record);
}

const chunks = [];
let size = 8;
for (const record of latest.values()) {
  const { width: w, height: h } = record.mask;
  const raster = decodeMaskToRaster(record.mask);
  const packed = new Uint8Array(Math.ceil((w * h) / 8));
  for (let i = 0; i < raster.length; i++) {
    if (raster[i]) packed[i >> 3] |= 0x80 >> (i & 7);
  }
  const header = Buffer.alloc(32);
  header.writeUInt32LE(record.frame.frameIndex, 0);
  header.writeUInt32LE(Number(record.objectId), 4);
  header.writeInt32LE(Math.round(record.bounds.left), 8);
  header.writeInt32LE(Math.round(record.bounds.top), 12);
  header.writeInt32LE(Math.round(record.bounds.right), 16);
  header.writeInt32LE(Math.round(record.bounds.bottom), 20);
  header.writeUInt32LE(h, 24);
  header.writeUInt32LE(w, 28);
  chunks.push(header, packed);
  size += header.length + packed.length;
}

const head = Buffer.alloc(8);
head.write('SCM1', 0, 'ascii');
head.writeUInt32LE(latest.size, 4);
writeFileSync(outPath, Buffer.concat([head, ...chunks], size));

const diagnostics = result.diagnostics.filter((d) => d.severity === 'error');
if (diagnostics.length) console.error(`diagnostics: ${diagnostics.length}`, diagnostics[0]);
console.log(JSON.stringify({ masks: latest.size, outcome: result.outcome.status }));
