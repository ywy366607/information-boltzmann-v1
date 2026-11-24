import { randomBytes, scrypt, timingSafeEqual } from 'node:crypto';
import { mkdir, readFile, rename, unlink, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { promisify } from 'node:util';

const derive = promisify(scrypt);
const validKey = key => typeof key === 'string' && key.length > 0 && Buffer.byteLength(key) <= 512;

function checkKey(key) {
  if (!validKey(key)) throw new Error('密钥须为非空文本，且不超过512字节');
}
function displayNameValue(name) {
  if (typeof name !== 'string' || !name.trim() || name.trim().length > 80) {
    throw new Error('昵称须为1至80个字符');
  }
  return name.trim();
}
async function keyRecord(key) {
  checkKey(key);
  const salt = randomBytes(16).toString('hex');
  return { salt, hash: (await derive(key, salt, 64)).toString('hex') };
}
async function matches(key, record) {
  if (!validKey(key)) return false;
  const candidate = await derive(key, record.salt, 64);
  return timingSafeEqual(candidate, Buffer.from(record.hash, 'hex'));
}

/** One local account; only the salted scrypt digest and display name are persisted. */
export async function openLocalAccount(file, { initialKey, displayName = '无月酱酱酱酱' } = {}) {
  let record;
  try {
    record = JSON.parse(await readFile(file, 'utf8'));
  } catch (error) {
    if (error.code !== 'ENOENT') throw new Error('无法读取本地账号');
    if (initialKey === undefined) throw new Error('本地账号不存在，须提供initialKey');
  }
  const save = async next => {
    const temporary = `${file}.${randomBytes(8).toString('hex')}.tmp`;
    try {
      await mkdir(path.dirname(file), { recursive: true });
      await writeFile(temporary, JSON.stringify(next) + '\n', { flag: 'wx', mode: 0o600 });
      await rename(temporary, file);
    } catch {
      throw new Error('无法保存本地账号');
    } finally {
      await unlink(temporary).catch(() => {});
    }
    record = next;
  };
  if (record === undefined) {
    await save({ displayName: displayNameValue(displayName), ...await keyRecord(initialKey) });
  } else if (!record || typeof record !== 'object' ||
    typeof record.displayName !== 'string' || !record.displayName.trim() || record.displayName.length > 80 ||
    !/^[a-f0-9]{32}$/.test(record.salt) || !/^[a-f0-9]{128}$/.test(record.hash)) {
    throw new Error('本地账号格式无效');
  }
  return {
    view: () => ({ displayName: record.displayName }),
    verify: key => matches(key, record),
    async setDisplayName(name) {
      await save({ ...record, displayName: displayNameValue(name) });
      return { displayName: record.displayName };
    },
    async changeKey(currentKey, newKey) {
      checkKey(currentKey);
      checkKey(newKey);
      if (!await matches(currentKey, record)) {
        const error = new Error('当前密钥不正确');
        error.code = 'INVALID_KEY';
        throw error;
      }
      await save({ ...record, ...await keyRecord(newKey) });
      // Existing native browser cookies retain their official lifetime.
      return { ok: true };
    },
  };
}
