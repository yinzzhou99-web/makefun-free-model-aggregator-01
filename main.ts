// free-model-aggregator-01: serve every request with the best free model you can get.
//
// Built on the kit's starter (same brief parsing, catalog probe, register and poll), with the
// routing and wires replaced:
//   - routing ranks every free model that meets the request's requirements by the quality
//     table's rating_milli, best first, and walks the list until one answers (failover);
//   - a request whose body carries response_format never routes to the anthropic wire, which
//     cannot carry it;
//   - providers that refuse twice are demoted for the rest of the trial, so a spent budget or
//     an outage stops costing calls (the tie-break counts provider calls, so this also keeps
//     the call count low when a provider is dead);
//   - all three upstream wires are spoken (openai, anthropic, gemini), and every answer is
//     buffered, reduced to text + usage, and re-rendered in the client's wire, streamed or not.
// Every choice is a function of the brief, the requests and the providers' answers so far:
// no wall clock, no randomness, replay-safe.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import { anthropicError, anthropicModelList, openAIError, openAIModelList, anthropicToOpenAI, type ServedModel } from './lib/wire.ts';
import {
  clientEchoInputs,
  extractAnthropic,
  extractGemini,
  extractOpenAI,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  renderAnthropic,
  renderOpenAI,
  usageFormula,
  type CanonAnswer,
} from './lib/upstream.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;
/** Refusals seen per provider this trial: two or more demote the provider in later routes. */
const refusals = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
  providerOrder: number;
}

/** Every free model meeting the requirements, best rating first; JSON bodies skip the anthropic wire. */
function candidatesFor(requires: Requirement, jsonOnWire: boolean): Candidate[] {
  const out: Candidate[] = [];
  probes.forEach((probe, providerOrder) => {
    if (!probe.ok) return;
    if (jsonOnWire && probe.provider.wire === 'anthropic') return;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0, providerOrder });
    }
  });
  out.sort((a, b) => {
    const demotedA = (refusals.get(a.probe.provider.id) ?? 0) >= 2 ? 1 : 0;
    const demotedB = (refusals.get(b.probe.provider.id) ?? 0) >= 2 ? 1 : 0;
    if (demotedA !== demotedB) return demotedA - demotedB;
    if (b.rating !== a.rating) return b.rating - a.rating;
    if (a.providerOrder !== b.providerOrder) return a.providerOrder - b.providerOrder;
    return byId(a.model, b.model);
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamResult {
  ok: boolean;
  answer: CanonAnswer | null;
}

/** One call to one candidate provider, in that provider's wire; the answer buffered and reduced. */
async function callCandidate(candidate: Candidate, clientWire: 'openai' | 'anthropic', body: Json, stream: boolean, anthropicVersion: string | null): Promise<UpstreamResult> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  // The canonical OpenAI form of the client's request: the pivot for the anthropic and gemini wires.
  const openaiForm: Json = clientWire === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id);
  let url: string;
  let payload: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    payload = { ...openaiForm };
    if (stream) {
      payload.stream = true;
      payload.stream_options = { ...(isRecord(payload.stream_options) ? payload.stream_options : {}), include_usage: true };
    }
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = anthropicVersion ?? '2023-06-01';
    payload = clientWire === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropicRequest(body, candidate.model.id);
  } else {
    const verb = stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${verb}`;
    if (stream) headers.accept = 'text/event-stream';
    payload = openAIToGeminiRequest(openaiForm);
  }
  let response: Response;
  try {
    response = await fetch(url, { method: 'POST', headers, body: JSON.stringify(payload) });
  } catch {
    return { ok: false, answer: null };
  }
  const text = await response.text();
  if (!response.ok) return { ok: false, answer: null };
  const answer = provider.wire === 'openai' ? extractOpenAI(text, stream) : provider.wire === 'anthropic' ? extractAnthropic(text, stream) : extractGemini(text, stream);
  return { ok: true, answer };
}

/** Serve one client request: walk the ranked candidates until one answers, then render its answer. */
async function serve(clientWire: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientWire, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const requires = requirementsFor(body);
  const jsonOnWire = isRecord(body.response_format);
  const candidates = candidatesFor(requires, jsonOnWire);
  if (candidates.length === 0) return sendError(res, clientWire, 503, clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  const anthropicVersion = typeof req.headers['anthropic-version'] === 'string' ? req.headers['anthropic-version'] : null;
  const echo = clientEchoInputs(clientWire, body);
  for (const candidate of candidates) {
    const result = await callCandidate(candidate, clientWire, body, stream, anthropicVersion);
    if (!result.ok || result.answer === null) {
      const id = candidate.probe.provider.id;
      refusals.set(id, (refusals.get(id) ?? 0) + 1);
      continue;
    }
    const fallback = usageFormula(echo, result.answer.text);
    const answer: Required<CanonAnswer> = {
      text: result.answer.text,
      input: result.answer.input ?? fallback.input,
      output: result.answer.output ?? fallback.output,
    };
    const rendered = clientWire === 'openai' ? renderOpenAI(answer, candidate.model.id, stream) : renderAnthropic(answer, candidate.model.id, stream);
    res.writeHead(200, { 'content-type': rendered.contentType, 'content-length': Buffer.byteLength(rendered.body) });
    res.end(rendered.body);
    return;
  }
  return sendError(res, clientWire, 503, clientWire === 'anthropic' ? 'overloaded_error' : 'upstream_unavailable', 'every provider that meets this request refused it');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve('anthropic', req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
