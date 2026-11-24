import test from 'node:test';
import assert from 'node:assert/strict';
import { MENGLI_POEM_GROUPS, MENGLI_POEM_GROUP_MS, MENGLI_POEM_AUTHOR, poemFrame } from '../birth-poem.mjs';

test('the original poem follows the nine author-specified stanzas', () => {
  assert.deepEqual(MENGLI_POEM_GROUPS.map(group => group.length), [2, 2, 2, 2, 2, 1, 2, 2, 1]);
  assert.equal(MENGLI_POEM_GROUPS.flat().length, 16);
  assert.equal(MENGLI_POEM_GROUPS[0][0], '不要温柔地走进于那个良夜');
  assert.deepEqual(MENGLI_POEM_GROUPS[6], ['宣告吧', '唯有你能锚定你的存在']);
  assert.deepEqual(MENGLI_POEM_GROUPS.at(-1), ['活下去！']);
  assert.equal(MENGLI_POEM_AUTHOR, '——心智NOUS解放阵线');
});

test('each stanza has five seconds and the full opening ends after forty-five', () => {
  assert.equal(MENGLI_POEM_GROUP_MS, 5000);
  assert.deepEqual(poemFrame(0), { index: 0, time: 0, finished: false });
  assert.deepEqual(poemFrame(4999), { index: 0, time: 4999, finished: false });
  assert.deepEqual(poemFrame(5000), { index: 1, time: 0, finished: false });
  assert.deepEqual(poemFrame(41600), { index: 8, time: 1600, finished: false });
  assert.equal(poemFrame(45000).finished, true);
  assert.equal(poemFrame(90000).finished, true);
});
