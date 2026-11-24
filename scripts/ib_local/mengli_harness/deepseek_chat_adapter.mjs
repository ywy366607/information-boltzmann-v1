import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { DatabaseSync } from 'node:sqlite';

const packageRoot = process.env.MENGLI_DSH_PACKAGE_ROOT || 'F:/16、杂货行账册/ledger-dsh/node_modules/@deepseek-ai';
const { LlmAdapter, LlmError, attributionHeaders } = await import(
  pathToFileURL(path.join(packageRoot, 'dsh-llm/lib/index.js')).href);

export function normalizeUsage(usage) {
  if (!usage || !Number.isSafeInteger(usage.prompt_tokens) || usage.prompt_tokens < 0 ||
      !Number.isSafeInteger(usage.completion_tokens) || usage.completion_tokens < 0) return null;
  const reportedHit = usage.prompt_cache_hit_tokens ?? usage.prompt_tokens_details?.cached_tokens;
  const hit = reportedHit ?? 0;
  if (!Number.isSafeInteger(hit) || hit < 0 || hit > usage.prompt_tokens) return null;
  return {
    inputTokens: usage.prompt_tokens - hit,
    ...(reportedHit === undefined ? {} : { cacheReadTokens: hit }),
    outputTokens: usage.completion_tokens,
    totalTokens: usage.prompt_tokens + usage.completion_tokens,
  };
}

export function publicUsage(usage) {
  if (!usage) return null;
  return {
    prompt_tokens: usage.inputTokens + (usage.cacheReadTokens ?? 0) + (usage.cacheWriteTokens ?? 0),
    completion_tokens: usage.outputTokens,
    cache_hit_tokens: usage.cacheReadTokens ?? null,
    cache_miss_tokens: usage.cacheReadTokens === undefined ? null : usage.inputTokens,
    total_tokens: usage.totalTokens ?? null,
  };
}

// A single cache at the actual transport boundary. No session state is shared.
export class DeepSeekChatAdapter extends LlmAdapter {
  constructor({ cachePath, tokenMeter, fetchImpl = globalThis.fetch, onCall = () => {} }) {
    super();
    this.tokenMeter = tokenMeter;
    this.fetchImpl = fetchImpl;
    this.onCall = onCall;
    this.localKeys = new Set();
    this.sessionContextWindows = new Map();
    this.defaultContextWindow = 8192;
    this.db = new DatabaseSync(cachePath);
    this.db.exec('CREATE TABLE IF NOT EXISTS chat_wire_v1 (key TEXT PRIMARY KEY, response TEXT NOT NULL, created INTEGER NOT NULL)');
  }

  setContextWindow(sessionId, contextWindow) {
    if (sessionId && Number.isSafeInteger(contextWindow) && contextWindow >= 8192) {
      this.sessionContextWindows.set(sessionId, contextWindow);
      this.defaultContextWindow = contextWindow;
    }
  }

  contextWindowFor(sessionId) {
    return (sessionId && this.sessionContextWindows.get(sessionId)) || this.defaultContextWindow || 8192;
  }

  async resolveModel(provider, model) {
    if (model !== 'deepseek-flash') throw new LlmError('Only deepseek-flash is configured', 'INVALID_MODEL');
    return {
      provider, id: model, name: model, inputModalities: ['text'],
      // Harness run budget, not the provider model maximum context window.
      context: { contextWindow: this.defaultContextWindow || 8192 }, defaultMaxTokens: 120,
    };
  }

  async *stream(options) {
    if (options.model !== 'deepseek-flash' || options.tools?.length) {
      throw new LlmError('Unsupported model or tools in caregiver request', 'INVALID_REQUEST');
    }
    const messages = options.messages.map(message => {
      if (!['system', 'user', 'assistant'].includes(message.role) || message.content.some(block => block.type !== 'text')) {
        throw new LlmError('Caregiver wire accepts text conversation only', 'UNSUPPORTED_CONTENT');
      }
      return { role: message.role, content: message.content.map(block => block.text).join('') };
    });
    if (options.system) messages.unshift({ role: 'system', content: options.system });
    const maxTokens = options.maxTokens ?? (options.purpose === 'compaction' ? 1536 : options.purpose === 'reflection' ? 600 : 120);
    const contextBudget = options.contextWindow ?? this.contextWindowFor(options.sessionId);
    const headroom = contextBudget >= 131072 ? 4096 : contextBudget >= 32768 ? 2048 : 1024;
    const estimated = messages.reduce((sum, message) => sum + this.tokenMeter.estimateMessage({
      role: message.role, content: [{ type: 'text', text: message.content }],
    }), 0);
    if (estimated + maxTokens + headroom > contextBudget) {
      throw new LlmError(`Request exceeds the ${contextBudget}-token heuristic run budget after compaction`, 'CONTEXT_BUDGET');
    }
    const body = {
      model: options.model, messages, temperature: options.temperature ?? 0.7,
      max_tokens: maxTokens, stream: false, thinking: { type: 'disabled' },
    };
    const url = 'https://api.deepseek.com/chat/completions';
    const key = createHash('sha256').update(JSON.stringify({ namespace: 'mengli-chat-wire-v1', transport: url, body })).digest('hex');
    const row = this.db.prepare('SELECT response FROM chat_wire_v1 WHERE key=?').get(key);
    let response, source;
    const started = performance.now();
    const report = (usage, failed = false) => this.onCall({
      session_id: options.sessionId, purpose: options.purpose ?? 'conversation', source,
      usage: source === 'api' ? publicUsage(usage) : null,
      cached_usage: source === 'api' ? null : publicUsage(usage), failed,
      latency_seconds: (performance.now() - started) / 1000, estimated_input_tokens: estimated, cache_key: key,
    });
    if (row) {
      response = JSON.parse(row.response);
      source = this.localKeys.has(key) ? 'local' : 'shared';
    } else {
      const apiKey = process.env.DEEPSEEK_API_KEY?.trim();
      if (!apiKey) throw new LlmError('DEEPSEEK_API_KEY is not set in the environment', 'AUTH');
      source = 'api';
      let res;
      try {
        res = await this.fetchImpl(url, {
          method: 'POST', headers: { 'content-type': 'application/json', authorization: `Bearer ${apiKey}`, ...attributionHeaders() },
          body: JSON.stringify(body), signal: options.signal,
        });
      } catch {
        report(null, true);
        throw new LlmError('DeepSeek transport failed', options.signal?.aborted ? 'ABORTED' : 'TRANSPORT');
      }
      if (!res.ok) {
        report(null, true);
        throw new LlmError(`DeepSeek HTTP ${res.status}`, 'PROVIDER_HTTP', { status: res.status });
      }
      try { response = await res.json(); } catch {
        report(null, true);
        throw new LlmError('DeepSeek returned invalid JSON', 'PROVIDER_RESPONSE');
      }
    }
    const usage = normalizeUsage(response.usage);
    const choice = response.choices?.[0];
    if (choice?.finish_reason !== 'stop') {
      report(usage, true);
      throw new LlmError('DeepSeek reply did not finish normally', choice?.finish_reason === 'length' ? 'MAX_TOKENS' : 'PROVIDER_FINISH');
    }
    if (typeof choice.message?.content !== 'string' || !choice.message.content.trim() || choice.message?.tool_calls?.length) {
      report(usage, true);
      throw new LlmError('DeepSeek returned no usable text reply', 'EMPTY_REPLY');
    }
    if (!row) {
      // Concurrent processes return the first durable real sample, never overwrite it.
      this.db.prepare('INSERT OR IGNORE INTO chat_wire_v1 VALUES(?,?,?)').run(key, JSON.stringify(response), Date.now());
      response = JSON.parse(this.db.prepare('SELECT response FROM chat_wire_v1 WHERE key=?').get(key).response);
      this.db.exec('DELETE FROM chat_wire_v1 WHERE key IN (SELECT key FROM chat_wire_v1 ORDER BY created DESC LIMIT -1 OFFSET 2048)');
    }
    this.localKeys.add(key);
    if (this.localKeys.size > 2048) this.localKeys.delete(this.localKeys.values().next().value);
    report(usage);
    const raw = response.choices[0].message.content;
    yield { type: 'block-start', index: 0, blockType: 'text' };
    yield { type: 'text-delta', index: 0, text: raw };
    yield { type: 'block-end', index: 0, block: { type: 'text', text: raw } };
    if (usage && source === 'api') yield { type: 'usage', usage };
    yield { type: 'finish', reason: { kind: 'stop' }, replayState: { response: { source, cache_key: key } } };
  }

  close() { this.db.close(); }
}
