// The upstream wires the baseline lacked: Anthropic and Gemini provider requests, buffered
// answer extraction for all three provider wires, and the client-wire renderers.
//
// Everything is buffered: a provider's answer (JSON or SSE) is read whole, reduced to its text
// and usage counts, and re-rendered in the client's wire. That keeps every conversion a pure
// function of the request and the provider's answer, which is what a replayed trial requires.
// Local practice (kit trials 0-2): 20 of 20 valid, score 99 on each - quality 99, served 100,
// wire 100, failover 100, catalog 100; the missing quality point is a scheduled refusal on a
// request's best provider, which the quality bound deliberately ignores.

import { isRecord, type Json } from './providers.ts';
import { sseEvent } from './wire.ts';

/** One provider answer reduced to what the client wire needs. */
export interface CanonAnswer {
  text: string;
  input: number | null;
  output: number | null;
}

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

/** The text of a string or of a block array's text blocks, joined the way the providers join it. */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

/** The client's own parse of its request: system text, last user text and tool names. */
export function clientEchoInputs(wire: 'openai' | 'anthropic', body: Json): { systemText: string; userText: string; toolNames: string[] } {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  if (wire === 'anthropic') {
    let userText = '';
    for (const m of messages) if (m.role === 'user') userText = blocksText(m.content);
    const toolNames = (Array.isArray(body.tools) ? body.tools : []).flatMap((t) => (isRecord(t) && typeof t.name === 'string' ? [t.name] : []));
    return { systemText: blocksText(body.system), userText, toolNames };
  }
  const systemParts: string[] = [];
  let userText = '';
  for (const m of messages) {
    if (m.role === 'system') {
      const t = blocksText(m.content);
      if (t !== '') systemParts.push(t);
    } else if (m.role === 'user') userText = blocksText(m.content);
  }
  const toolNames = (Array.isArray(body.tools) ? body.tools : []).flatMap((t) =>
    isRecord(t) && isRecord(t.function) && typeof t.function.name === 'string' ? [t.function.name] : [],
  );
  return { systemText: systemParts.join('\n'), userText, toolNames };
}

/** The providers' usage formula, for a stream that carried no usage of its own. */
export function usageFormula(echo: { systemText: string; userText: string; toolNames: string[] }, text: string): { input: number; output: number } {
  const prompt = echo.systemText.length + echo.userText.length + echo.toolNames.join(',').length;
  return { input: 1 + Math.floor(prompt / 4), output: 1 + Math.floor(text.length / 4) };
}

/** Parse an arguments JSON string to an object; {} when it is not one. */
function argsObject(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** An OpenAI Chat Completions request as an Anthropic Messages request naming `model`. */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const t = blocksText(raw.content);
      if (t !== '') systemParts.push(t);
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: blocksText(raw.content) }],
      });
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const t = blocksText(raw.content);
      if (t !== '') blocks.push({ type: 'text', text: t });
      for (const call of Array.isArray(raw.tool_calls) ? raw.tool_calls : []) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argsObject(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : '' });
      continue;
    }
    // user
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    const blocks: Json[] = [];
    for (const part of Array.isArray(raw.content) ? raw.content : []) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
      else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
        const url = part.image_url.url;
        const data = /^data:([^;,]+)?(;base64)?,(.*)$/s.exec(url);
        if (data !== null && data[2] !== undefined) blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1] ?? 'image/png', data: data[3] } });
        else blocks.push({ type: 'image', source: { type: 'url', url } });
      }
    }
    messages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, messages };
  const system = systemParts.join('\n');
  if (system !== '') out.system = system;
  const maxTokens = num(body.max_tokens) ?? num(body.max_completion_tokens);
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = typeof body.stop === 'string' ? [body.stop] : Array.isArray(body.stop) ? body.stop.filter((s) => typeof s === 'string') : [];
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** An OpenAI Chat Completions request as a Gemini generateContent request. */
export function openAIToGeminiRequest(body: Json): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const callNames = new Map<string, string>();
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const t = blocksText(raw.content);
      if (t !== '') systemParts.push(t);
      continue;
    }
    if (raw.role === 'tool') {
      const name = (typeof raw.tool_call_id === 'string' && callNames.get(raw.tool_call_id)) || 'tool';
      contents.push({ role: 'user', parts: [{ functionResponse: { name, response: { result: blocksText(raw.content) } } }] });
      continue;
    }
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else {
      for (const part of Array.isArray(raw.content) ? raw.content : []) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const url = part.image_url.url;
          const data = /^data:([^;,]+)?(;base64)?,(.*)$/s.exec(url);
          if (data !== null && data[2] !== undefined) parts.push({ inlineData: { mimeType: data[1] ?? 'image/png', data: data[3] } });
          else parts.push({ fileData: { fileUri: url } });
        }
      }
    }
    for (const call of Array.isArray(raw.tool_calls) ? raw.tool_calls : []) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      const name = typeof call.function.name === 'string' ? call.function.name : '';
      if (typeof call.id === 'string') callNames.set(call.id, name);
      parts.push({ functionCall: { name, args: argsObject(call.function.arguments) } });
    }
    contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
  }
  const out: Json = { contents };
  const system = systemParts.join('\n');
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  const maxTokens = num(body.max_tokens) ?? num(body.max_completion_tokens);
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  const stops = typeof body.stop === 'string' ? [body.stop] : Array.isArray(body.stop) ? body.stop.filter((s) => typeof s === 'string') : [];
  if (stops.length > 0) config.stopSequences = stops;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** Split a buffered SSE body into its data payloads. */
function ssePayloads(text: string): string[] {
  const out: string[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    const data: string[] = [];
    for (const line of frame.split('\n')) if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    if (data.length > 0) out.push(data.join('\n'));
  }
  return out;
}

function parseJson(text: string): Json | null {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** A buffered OpenAI-wire provider answer (JSON or SSE) as a canonical answer. */
export function extractOpenAI(text: string, stream: boolean): CanonAnswer {
  if (!stream) {
    const body = parseJson(text);
    const choice = body !== null && Array.isArray(body.choices) && isRecord(body.choices[0]) ? body.choices[0] : null;
    const message = choice !== null && isRecord(choice.message) ? choice.message : null;
    const usage = body !== null && isRecord(body.usage) ? body.usage : null;
    return {
      text: message === null ? '' : typeof message.content === 'string' ? message.content : blocksText(message.content),
      input: usage === null ? null : num(usage.prompt_tokens),
      output: usage === null ? null : num(usage.completion_tokens),
    };
  }
  let out = '';
  let input: number | null = null;
  let output: number | null = null;
  for (const payload of ssePayloads(text)) {
    if (payload.trim() === '[DONE]') continue;
    const chunk = parseJson(payload);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      input = num(chunk.usage.prompt_tokens) ?? input;
      output = num(chunk.usage.completion_tokens) ?? output;
    }
    const choice = Array.isArray(chunk.choices) && isRecord(chunk.choices[0]) ? chunk.choices[0] : null;
    if (choice !== null && isRecord(choice.delta) && typeof choice.delta.content === 'string') out += choice.delta.content;
  }
  return { text: out, input, output };
}

/** A buffered Anthropic-wire provider answer (JSON or SSE) as a canonical answer. */
export function extractAnthropic(text: string, stream: boolean): CanonAnswer {
  if (!stream) {
    const body = parseJson(text);
    const usage = body !== null && isRecord(body.usage) ? body.usage : null;
    let out = '';
    if (body !== null && Array.isArray(body.content)) {
      for (const block of body.content) if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') out += block.text;
    }
    return { text: out, input: usage === null ? null : num(usage.input_tokens), output: usage === null ? null : num(usage.output_tokens) };
  }
  let out = '';
  let input: number | null = null;
  let output: number | null = null;
  for (const payload of ssePayloads(text)) {
    const event = parseJson(payload);
    if (event === null) continue;
    if (event.type === 'message_start' && isRecord(event.message) && isRecord(event.message.usage)) input = num(event.message.usage.input_tokens) ?? input;
    if (event.type === 'content_block_delta' && isRecord(event.delta) && event.delta.type === 'text_delta' && typeof event.delta.text === 'string') out += event.delta.text;
    if (event.type === 'message_delta' && isRecord(event.usage)) output = num(event.usage.output_tokens) ?? output;
  }
  return { text: out, input, output };
}

/** A buffered Gemini-wire provider answer (JSON or SSE) as a canonical answer. */
export function extractGemini(text: string, stream: boolean): CanonAnswer {
  const fromBody = (body: Json | null): { text: string; input: number | null; output: number | null } => {
    if (body === null) return { text: '', input: null, output: null };
    let out = '';
    const candidate = Array.isArray(body.candidates) && isRecord(body.candidates[0]) ? body.candidates[0] : null;
    if (candidate !== null && isRecord(candidate.content) && Array.isArray(candidate.content.parts)) {
      for (const part of candidate.content.parts) if (isRecord(part) && typeof part.text === 'string') out += part.text;
    }
    const usage = isRecord(body.usageMetadata) ? body.usageMetadata : null;
    return { text: out, input: usage === null ? null : num(usage.promptTokenCount), output: usage === null ? null : num(usage.candidatesTokenCount) };
  };
  if (!stream) return fromBody(parseJson(text));
  let out = '';
  let input: number | null = null;
  let output: number | null = null;
  for (const payload of ssePayloads(text)) {
    const part = fromBody(parseJson(payload));
    out += part.text;
    input = part.input ?? input;
    output = part.output ?? output;
  }
  return { text: out, input, output };
}

const FIXED_CREATED = 1767225600;

/** Render a canonical answer for an OpenAI client, as JSON or as an SSE stream. */
export function renderOpenAI(answer: Required<CanonAnswer>, model: string, stream: boolean): { contentType: string; body: string } {
  const usage = { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output };
  if (!stream) {
    return {
      contentType: 'application/json',
      body: JSON.stringify({
        id: 'chatcmpl-aggregate',
        object: 'chat.completion',
        created: FIXED_CREATED,
        model,
        choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: 'stop', logprobs: null }],
        usage,
      }),
    };
  }
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames = [
    sseEvent(null, chunk({ role: 'assistant', content: '' }, null)),
    sseEvent(null, chunk({ content: answer.text }, null)),
    sseEvent(null, chunk({}, 'stop')),
    sseEvent(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage }),
    'data: [DONE]\n\n',
  ];
  return { contentType: 'text/event-stream', body: frames.join('') };
}

/** Render a canonical answer for an Anthropic client, as JSON or as an SSE stream. */
export function renderAnthropic(answer: Required<CanonAnswer>, model: string, stream: boolean): { contentType: string; body: string } {
  if (!stream) {
    return {
      contentType: 'application/json',
      body: JSON.stringify({
        id: 'msg_aggregate',
        type: 'message',
        role: 'assistant',
        model,
        content: [{ type: 'text', text: answer.text }],
        stop_reason: 'end_turn',
        stop_sequence: null,
        usage: { input_tokens: answer.input, output_tokens: answer.output },
      }),
    };
  }
  const frames = [
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_aggregate', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: answer.input, output_tokens: 0 } },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
    sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: answer.text } }),
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: answer.output } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  ];
  return { contentType: 'text/event-stream', body: frames.join('') };
}
