// The two client wires' static shapes: model lists, error envelopes, one SSE event, and the
// Anthropic -> OpenAI request translation used when an Anthropic client is served by an
// OpenAI- or Gemini-wire provider. Written for this entry; only what main.ts and upstream.ts
// use lives here. Timestamps in list answers are fixed constants, never the wall clock, so a
// replayed trial produces byte-identical answers.
import { isRecord, type Json } from './providers.ts';

export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

const CREATED = 1767225600; // 2026-01-01T00:00:00Z as epoch seconds
const CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models for a Bearer caller: OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models for an x-api-key caller: Anthropic's list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: CREATED_AT })),
    has_more: false,
  };
}

/** Anthropic's error envelope: {"type":"error","error":{...}}. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** OpenAI's error envelope: {"error":{...}}. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** One SSE event: an optional `event:` line, the JSON `data:` line, and the blank separator. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** The plain text in a string or a block array (text blocks joined by newlines). */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

/** One Anthropic image block as an OpenAI image_url part (data URL for base64 sources). */
function imagePart(block: Json): Json | null {
  if (!isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
    return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
  }
  if (source.type === 'url' && typeof source.url === 'string') {
    return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/**
 * An Anthropic Messages request rewritten as an OpenAI Chat Completions request for `model`:
 * the system prompt becomes a leading system message, text and image blocks become content
 * parts, tool_use blocks become tool_calls, tool_result blocks become tool messages, and the
 * tools, tool_choice, output cap and sampling fields are carried across in OpenAI's names.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'assistant' ? 'assistant' : 'user';
    if (typeof raw.content === 'string') {
      messages.push({ role, content: raw.content });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    if (role === 'assistant') {
      const toolCalls: Json[] = [];
      for (const block of raw.content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(raw.content);
      messages.push({ role, content: text === '' ? null : text, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of raw.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: textOf(block.content),
        });
      } else if (block.type === 'text' && typeof block.text === 'string') {
        parts.push({ type: 'text', text: block.text });
      } else if (block.type === 'image') {
        const part = imagePart(block);
        if (part !== null) parts.push(part);
      }
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools: Json[] = [];
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        type: 'function',
        function: {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      });
    }
    if (tools.length > 0) out.tools = tools;
  }
  if (isRecord(body.tool_choice)) {
    const choice = body.tool_choice;
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}
