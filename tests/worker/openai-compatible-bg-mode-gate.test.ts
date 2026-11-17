import { afterEach, beforeEach, describe, expect, it, mock, spyOn } from 'bun:test';
import { Database } from 'bun:sqlite';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'fs';
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { tmpdir } from 'os';
import { join } from 'path';
import { ModeManager } from '../../src/services/domain/ModeManager.js';
import { SessionStore } from '../../src/services/sqlite/SessionStore.js';
import { setIngestContext, ingestObservation } from '../../src/services/worker/http/shared.js';
import { OpenAICompatProvider, withoutPinnedAccount } from '../../src/services/worker/OpenAICompatProvider.js';
import { parseBgWriterState } from '../../src/services/worker/OpenAICompatibleProvider.js';
import { PaidSendBudget } from '../../src/services/worker/paid-send-budget.js';
import type { DatabaseManager } from '../../src/services/worker/DatabaseManager.js';
import type { SessionEventBroadcaster } from '../../src/services/worker/events/SessionEventBroadcaster.js';
import type { SessionManager } from '../../src/services/worker/SessionManager.js';
import type { ActiveSession, PendingMessageWithId } from '../../src/services/worker-types.js';
import { SettingsDefaultsManager } from '../../src/shared/SettingsDefaultsManager.js';
import { USER_SETTINGS_PATH } from '../../src/shared/paths.js';
import { logger, type Component } from '../../src/utils/logger.js';

const ZERO_VALUE_BUILTIN_TOOLS = [
  'Agent',
  'ToolSearch',
  'SendMessage',
  'StructuredOutput',
  'TaskStop',
  'Workflow',
  'Monitor',
] as const;

const mockMode = {
  name: 'code',
  prompts: {
    init: 'init prompt',
    observation: 'obs prompt',
    summary: 'summary prompt',
  },
  observation_types: [{ id: 'discovery' }, { id: 'change' }],
  observation_concepts: [],
};

const OBSERVATION_XML = `
<observation>
  <type>change</type>
  <title>Updated post-draft gatekeeper</title>
  <narrative>Added System One scoring before storing observation drafts.</narrative>
  <facts><fact>Threshold is 0.20</fact></facts>
  <concepts><concept>gatekeeper</concept></concepts>
  <files_read></files_read>
  <files_modified><file>src/services/worker/OpenAICompatibleProvider.ts</file></files_modified>
</observation>
`.trim();

const SUMMARY_XML = `
<summary>
  <request>Wire bg-mode and gatekeeper</request>
  <investigated>OpenAICompatibleProvider message loop</investigated>
  <learned>System One scores keep_observation with noul probability</learned>
  <completed>Implemented mode routing and gatekeeper</completed>
  <next_steps>Run verification suite</next_steps>
</summary>
`.trim();

function makeSession(overrides: Partial<ActiveSession> = {}): ActiveSession {
  return {
    sessionDbId: 77,
    contentSessionId: 'content-bg-mode-77',
    memorySessionId: 'mem-bg-mode-77',
    project: 'test-project',
    platformSource: 'claude',
    userPrompt: 'Fix bg-mode observation gate',
    abortController: new AbortController(),
    generatorPromise: null,
    lastPromptNumber: 2,
    startTime: Date.now(),
    cumulativeInputTokens: 0,
    cumulativeOutputTokens: 0,
    earliestPendingTimestamp: 1700000000000,
    claimedMessageIds: [1],
    conversationHistory: [],
    currentProvider: 'openai-compatible',
    consecutiveRestarts: 0,
    consecutiveInvalidOutputs: 0,
    consecutiveContextOverflows: 0,
    lastGeneratorActivity: Date.now(),
    ...overrides,
  };
}

function makeObs(
  id: number,
  overrides: {
    tool_name: string;
    tool_input: unknown;
    tool_response: unknown;
    prompt_number?: number;
  },
): PendingMessageWithId {
  return {
    _persistentId: id,
    _originalTimestamp: 1700000000000 + id,
    type: 'observation',
    tool_name: overrides.tool_name,
    tool_input: typeof overrides.tool_input === 'string' ? overrides.tool_input : JSON.stringify(overrides.tool_input),
    tool_response: typeof overrides.tool_response === 'string' ? overrides.tool_response : JSON.stringify(overrides.tool_response),
    prompt_number: overrides.prompt_number ?? 2,
  };
}

function makeSummary(id: number, lastAssistantMessage: string): PendingMessageWithId {
  return {
    _persistentId: id,
    _originalTimestamp: 1700000000000 + id,
    type: 'summarize',
    last_assistant_message: lastAssistantMessage,
    prompt_number: 2,
  };
}

interface RecordedChatRequest {
  url: string;
  headers: Record<string, string | string[] | undefined>;
  body: Record<string, unknown>;
}

interface RecordedSystemOneRequest {
  url: string;
  headers: Record<string, string | string[] | undefined>;
  body: Record<string, unknown>;
}

interface FakeChatReply {
  status?: number;
  headers?: Record<string, string>;
  body: Record<string, unknown>;
}

async function startFakeGateway(handlers: {
  onChat: (req: RecordedChatRequest) => FakeChatReply | Promise<FakeChatReply>;
  onSystemOne: (req: RecordedSystemOneRequest, res: ServerResponse) => void | Promise<void>;
  onConcurrency?: () => { status?: number; body: Record<string, unknown> };
}) {
  const chatRequests: RecordedChatRequest[] = [];
  const systemOneRequests: RecordedSystemOneRequest[] = [];

  const readJsonBody = (req: IncomingMessage): Promise<Record<string, unknown>> =>
    new Promise((resolve, reject) => {
      const chunks: Buffer[] = [];
      req.on('data', (chunk: Buffer) => chunks.push(chunk));
      req.on('end', () => {
        const text = Buffer.concat(chunks).toString('utf-8');
        try {
          resolve(text ? (JSON.parse(text) as Record<string, unknown>) : {});
        } catch (err) {
          reject(err);
        }
      });
      req.on('error', reject);
    });

  const server = createServer(async (req, res) => {
    const url = req.url ?? '/';
    if (req.method === 'GET' && url === '/v1/magpie/concurrency') {
      const reply = handlers.onConcurrency ? handlers.onConcurrency() : { status: 200, body: { active: [] } };
      res.writeHead(reply.status ?? 200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(reply.body));
      return;
    }
    const body = await readJsonBody(req);
    if (url.endsWith('/chat/completions')) {
      const record: RecordedChatRequest = { url, headers: req.headers, body };
      chatRequests.push(record);
      const reply = await handlers.onChat(record);
      res.writeHead(reply.status ?? 200, { 'Content-Type': 'application/json', ...reply.headers });
      res.end(JSON.stringify(reply.body));
      return;
    }
    if (url === '/v1/systemone') {
      const record: RecordedSystemOneRequest = { url, headers: req.headers, body };
      systemOneRequests.push(record);
      await handlers.onSystemOne(record, res);
      return;
    }
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'not found' }));
  });

  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', () => resolve()));
  const { port } = server.address() as { port: number };
  return {
    rootUrl: `http://127.0.0.1:${port}`,
    baseUrl: `http://127.0.0.1:${port}/v1`,
    concurrencyUrl: `http://127.0.0.1:${port}/v1/magpie/concurrency`,
    chatRequests,
    systemOneRequests,
    close: async () => {
      server.closeAllConnections();
      await new Promise<void>(resolve => server.close(() => resolve()));
    },
  };
}

function makeQueueAndDb(messages: PendingMessageWithId[]) {
  const pending = [...messages];
  let confirmCalls = 0;
  const storeObservations = mock((..._args: unknown[]) => ({
    observationIds: [101],
    summaryId: null as number | null,
    createdAtEpoch: Date.now(),
  }));

  const dbManager = {
    getSessionById: () => ({ memory_session_id: 'openai-compat-content-bg-mode-77-1' }),
    getSessionStore: () => ({
      storeObservations,
      ensureMemorySessionIdRegistered: mock(() => 'mem-bg-mode-77'),
      updateMemorySessionId: mock(() => {}),
      linkToolUsesToObservation: mock(() => 0),
    }),
    getChromaSync: () => ({
      syncObservation: mock(() => Promise.resolve()),
      syncSummary: mock(() => Promise.resolve()),
    }),
    getCloudSync: () => null,
  } as unknown as DatabaseManager;

  const sessionManager = {
    getMessageIterator: async function* () {
      while (pending.length > 0) {
        yield pending[0];
        if (pending.length > 0 && confirmCalls === 0) {
          // Guard against infinite loop if a failing implementation forgot to confirm.
          break;
        }
      }
    },
    confirmClaimedMessages: async () => {
      confirmCalls += 1;
      pending.shift();
      return 1;
    },
    resetProcessingToPending: async () => 0,
    getClaimedMessages: () => (pending.length > 0 ? [pending[0]] : []),
    deliverRequestedSessionWrapup: mock(() => {}),
  } as unknown as SessionManager;

  return {
    pending,
    get confirmCalls() { return confirmCalls; },
    storeObservations,
    dbManager,
    sessionManager,
  };
}

const ENV_KEYS = [
  'CLAUDE_MEM_PROVIDER',
  'CLAUDE_MEM_OPENAI_COMPAT_PRESET',
  'CLAUDE_MEM_OPENAI_COMPAT_BASE_URL',
  'CLAUDE_MEM_OPENAI_COMPAT_API_KEY',
  'CLAUDE_MEM_OPENAI_COMPAT_API_KEYS',
  'CLAUDE_MEM_OPENAI_COMPAT_MODEL',
  'CLAUDE_MEM_TIER_ROUTING_ENABLED',
  'CLAUDE_MEM_TIER_SUMMARY_MODEL',
  'CLAUDE_MEM_OBSERVER_CONTEXT_WINDOW',
  'CLAUDE_MEM_OBSERVE_BARE_PROMPTS',
  'CLAUDE_MEM_BG_MODE_FILE',
  'CLAUDE_MEM_MAGPIE_CONCURRENCY_URL',
] as const;

describe('CLAUDE_MEM_SKIP_TOOLS 7 zero-value built-in tools', () => {
  let store: SessionStore | undefined;
  let queued: Array<{ sessionDbId: number; data: unknown }>;
  let spies: ReturnType<typeof spyOn>[] = [];
  let savedSkipToolsEnv: string | undefined;
  let savedSettingsContent: string | null = null;

  beforeEach(() => {
    queued = [];
    savedSkipToolsEnv = process.env.CLAUDE_MEM_SKIP_TOOLS;
    delete process.env.CLAUDE_MEM_SKIP_TOOLS;
    savedSettingsContent = existsSync(USER_SETTINGS_PATH) ? readFileSync(USER_SETTINGS_PATH, 'utf-8') : null;
    writeFileSync(
      USER_SETTINGS_PATH,
      JSON.stringify({
        ...SettingsDefaultsManager.getAllDefaults(),
        CLAUDE_MEM_SKIP_TOOLS: [
          SettingsDefaultsManager.getAllDefaults().CLAUDE_MEM_SKIP_TOOLS,
          ...ZERO_VALUE_BUILTIN_TOOLS,
        ].join(','),
      }, null, 2),
      'utf-8',
    );
    store = new SessionStore(new Database(':memory:'));
    spies = [
      spyOn(logger, 'info').mockImplementation(() => {}),
      spyOn(logger, 'debug').mockImplementation(() => {}),
      spyOn(logger, 'warn').mockImplementation(() => {}),
      spyOn(logger, 'error').mockImplementation(() => {}),
      spyOn(logger, 'dataIn').mockImplementation(() => {}),
    ];
    setIngestContext({
      sessionManager: {
        queueObservation: async (sessionDbId: number, data: unknown) => {
          queued.push({ sessionDbId, data });
        },
      } as unknown as SessionManager,
      dbManager: { getSessionStore: () => store } as unknown as DatabaseManager,
      eventBroadcaster: { broadcastObservationQueued: mock(() => {}) } as unknown as SessionEventBroadcaster,
      ensureGeneratorRunning: mock(async () => {}),
    });
  });

  afterEach(() => {
    if (savedSkipToolsEnv === undefined) delete process.env.CLAUDE_MEM_SKIP_TOOLS;
    else process.env.CLAUDE_MEM_SKIP_TOOLS = savedSkipToolsEnv;
    if (savedSettingsContent === null) rmSync(USER_SETTINGS_PATH, { force: true });
    else writeFileSync(USER_SETTINGS_PATH, savedSettingsContent, 'utf-8');
    spies.forEach(s => s.mockRestore());
    store?.close();
    store = undefined;
  });

  it('skips all 7 zero-value built-in tools configured in settings.json while queuing normal tools', async () => {
    for (const tool of ZERO_VALUE_BUILTIN_TOOLS) {
      const res = await ingestObservation({
        contentSessionId: `skip-${tool}`,
        toolName: tool,
        toolInput: {},
        toolResponse: 'ok',
        cwd: '/workspace/project',
        platformSource: 'claude',
        toolUseId: `tu-${tool}`,
      });
      expect(res).toEqual({ ok: true, status: 'skipped', reason: 'tool_excluded' });
    }
    expect(queued).toHaveLength(0);

    const kept = await ingestObservation({
      contentSessionId: 'keep-edit',
      toolName: 'Edit',
      toolInput: { file_path: '/workspace/project/a.ts' },
      toolResponse: 'updated',
      cwd: '/workspace/project',
      platformSource: 'claude',
      toolUseId: 'tu-edit',
    });
    expect(kept.ok).toBe(true);
    expect('status' in kept).toBe(false);
    expect(queued).toHaveLength(1);
  });
});

describe('OpenAICompatibleProvider bg-mode routing & System One post-draft gatekeeper', () => {
  let savedEnv: Record<string, string | undefined>;
  let tempDir: string;
  let bgModePath: string;
  let spies: ReturnType<typeof spyOn>[];
  let infoLogs: Array<{ component: string; message: string; context?: unknown }>;
  let loudLogs: string[];
  // Errors do not survive JSON.stringify, so their text and their cause's text are spelled out.
  const logText = (args: unknown[]): string => args.map(arg => arg instanceof Error
    ? `${arg.message} ${arg.cause instanceof Error ? arg.cause.message : ''}`
    : JSON.stringify(arg) ?? '').join(' ');

  beforeEach(() => {
    savedEnv = {};
    for (const key of ENV_KEYS) {
      savedEnv[key] = process.env[key];
      delete process.env[key];
    }
    tempDir = mkdtempSync(join(tmpdir(), 'bg-mode-gate-test-'));
    mkdirSync(tempDir, { recursive: true });
    bgModePath = join(tempDir, 'bg-mode.json');

    infoLogs = [];
    loudLogs = [];
    spies = [
      spyOn(ModeManager, 'getInstance').mockImplementation(() => ({
        getActiveMode: () => mockMode,
        loadMode: () => {},
      }) as unknown as ModeManager),
      spyOn(logger, 'debug').mockImplementation(() => {}),
      spyOn(logger, 'info').mockImplementation((component: Component, message: string, context?: unknown) => {
        infoLogs.push({ component, message, context });
      }),
      spyOn(logger, 'warn').mockImplementation((...args: unknown[]) => { loudLogs.push(logText(args)); }),
      spyOn(logger, 'error').mockImplementation((...args: unknown[]) => { loudLogs.push(logText(args)); }),
      spyOn(logger, 'success').mockImplementation(() => {}),
      spyOn(logger, 'failure').mockImplementation(() => {}),
    ];

    process.env.CLAUDE_MEM_PROVIDER = 'openai-compatible';
    process.env.CLAUDE_MEM_OPENAI_COMPAT_PRESET = 'custom';
    process.env.CLAUDE_MEM_OPENAI_COMPAT_API_KEY = 'synthetic-gate-key';
    process.env.CLAUDE_MEM_OPENAI_COMPAT_MODEL = 'mem-observer';
    process.env.CLAUDE_MEM_TIER_ROUTING_ENABLED = 'false';
    process.env.CLAUDE_MEM_OBSERVER_CONTEXT_WINDOW = '200000';
    process.env.CLAUDE_MEM_OBSERVE_BARE_PROMPTS = 'false';
    process.env.CLAUDE_MEM_BG_MODE_FILE = bgModePath;
  });

  afterEach(() => {
    for (const key of ENV_KEYS) {
      if (savedEnv[key] === undefined) delete process.env[key];
      else process.env[key] = savedEnv[key];
    }
    spies.forEach(s => s.mockRestore());
    mock.restore();
    rmSync(tempDir, { recursive: true, force: true });
  });

  const gptState = {
    mode: 'quality',
    profile: 'gpt',
    qualified_tasks: { 'claude-mem/observation': { model: 'codex/gpt-6.1-sol', effort: 'high' } },
    quota_target: { provider: 'codex', account: 'someone@example.test' },
  };

  const writerReply = (): FakeChatReply => ({
    body: {
      choices: [{ message: { role: 'assistant', content: OBSERVATION_XML }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120 },
    },
  });

  async function writerGateway(onChat: (req: RecordedChatRequest) => FakeChatReply = writerReply) {
    const gw = await startFakeGateway({
      onChat,
      onSystemOne: (_req, res) => {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ answers: { keep_observation: { noul: 0.9 } } }));
      },
    });
    process.env.CLAUDE_MEM_OPENAI_COMPAT_BASE_URL = gw.baseUrl;
    process.env.CLAUDE_MEM_MAGPIE_CONCURRENCY_URL = gw.concurrencyUrl;
    return gw;
  }

  async function observe() {
    const q = makeQueueAndDb([makeObs(1, {
      tool_name: 'Edit', tool_input: { file_path: 'writer.ts' }, tool_response: 'updated',
    })]);
    const session = makeSession();
    await new OpenAICompatProvider(q.dbManager, q.sessionManager).startSession(session);
    return { q, session };
  }

  function recipe(req: RecordedChatRequest) {
    return { model: req.body.model, effort: req.body.reasoning_effort, account: req.headers['x-magpie-account'] };
  }

  function expectGpt(req: RecordedChatRequest) {
    expect(recipe(req)).toEqual({ model: 'codex/gpt-6.1-sol', effort: 'high', account: 'someone%40example.test' });
  }

  function expectStored(q: ReturnType<typeof makeQueueAndDb>) {
    expect(q.storeObservations.mock.calls[0][2]).toEqual([
      expect.objectContaining({ title: 'Updated post-draft gatekeeper', narrative: 'Added System One scoring before storing observation drafts.' }),
    ]);
    expect(q.confirmCalls).toBe(1);
  }

  async function positiveControl(gw: Awaited<ReturnType<typeof writerGateway>>) {
    writeFileSync(bgModePath, JSON.stringify(gptState));
    const before = gw.chatRequests.length;
    const result = await observe();
    expectGpt(gw.chatRequests[before]);
    expectStored(result.q);
    return result;
  }

  it('qualified GPT observation sends model, effort and pinned account, stores the draft, and preserves the session model', async () => {
    const gw = await writerGateway();
    try {
      const { session } = await positiveControl(gw);
      expect(gw.chatRequests).toHaveLength(1);
      expect(session.lastModelId).toBe('mem-observer');
      expect(process.env.CLAUDE_MEM_OPENAI_COMPAT_MODEL).toBe('mem-observer');
      expect(session.paidSendBudget?.spentPaidSends).toBe(1);
      expect(infoLogs.some(l => l.component === 'OBS_GATE' && l.message.includes('writer=codex/gpt-6.1-sol'))).toBe(true);
      expect(gw.systemOneRequests[0].body.model).toBe('group/obs-gate');
      expect(gw.systemOneRequests[0].body.reasoning_effort).toBeUndefined();
      expect(gw.systemOneRequests[0].headers['x-magpie-account']).toBeUndefined();
    } finally {
      await gw.close();
    }
  });

  const unchangedStates = [
    ['current with qualified task', JSON.stringify({ ...gptState, profile: 'current' })],
    ['missing profile', JSON.stringify({ ...gptState, profile: undefined })],
    ['unknown profile', JSON.stringify({ ...gptState, profile: 'other' })],
    ['missing task', JSON.stringify({ ...gptState, qualified_tasks: {} })],
    ['empty model', JSON.stringify({ ...gptState, qualified_tasks: { 'claude-mem/observation': { model: '' } } })],
    ['whitespace model', JSON.stringify({ ...gptState, qualified_tasks: { 'claude-mem/observation': { model: '  ' } } })],
    ['non-string model', JSON.stringify({ ...gptState, qualified_tasks: { 'claude-mem/observation': { model: 42 } } })],
    ['non-object task', JSON.stringify({ ...gptState, qualified_tasks: { 'claude-mem/observation': 'codex/gpt-6.1-sol' } })],
    ['array qualified_tasks', JSON.stringify({ ...gptState, qualified_tasks: [] })],
    ['null qualified_tasks', JSON.stringify({ ...gptState, qualified_tasks: null })],
    ['string qualified_tasks', JSON.stringify({ ...gptState, qualified_tasks: 'invalid' })],
    ['missing file', undefined],
    ['empty file', ''],
    ['bad JSON', '{broken'],
    ['array state', '[]'],
    ['null state', 'null'],
    ['scalar state', '42'],
  ] as const;

  for (const [name, raw] of unchangedStates) {
    it(`keeps the original recipe for ${name}, paired with qualified GPT`, async () => {
      const gw = await writerGateway();
      try {
        await positiveControl(gw);
        if (raw === undefined) rmSync(bgModePath);
        else writeFileSync(bgModePath, raw);
        const { q } = await observe();
        expect(gw.chatRequests).toHaveLength(2);
        expect(recipe(gw.chatRequests[1])).toEqual({ model: 'bg-mem-quality', effort: undefined, account: undefined });
        expectStored(q);
      } finally {
        await gw.close();
      }
    });
  }

  it('reader exception keeps the original recipe, paired with qualified GPT', async () => {
    const gw = await writerGateway();
    try {
      await positiveControl(gw);
      rmSync(bgModePath);
      mkdirSync(bgModePath);
      const { q } = await observe();
      expect(recipe(gw.chatRequests[1])).toEqual({ model: 'bg-mem-quality', effort: undefined, account: undefined });
      expectStored(q);
    } finally {
      await gw.close();
    }
  });

  it('explicit model bypasses GPT and bg routing, paired with qualified GPT', async () => {
    const gw = await writerGateway();
    try {
      await positiveControl(gw);
      process.env.CLAUDE_MEM_OPENAI_COMPAT_MODEL = 'user/explicit-model';
      const { q, session } = await observe();
      expect(recipe(gw.chatRequests[1])).toEqual({ model: 'user/explicit-model', effort: undefined, account: undefined });
      expect(session.lastModelId).toBe('user/explicit-model');
      expectStored(q);
    } finally {
      await gw.close();
    }
  });

  for (const [name, quota] of [
    ['provider mismatch', { provider: 'google', account: 'someone@example.test' }],
    ['missing quota target', undefined],
    ['empty account', { provider: 'codex', account: '' }],
    ['non-string account', { provider: 'codex', account: 42 }],
    ['empty provider', { provider: '', account: 'someone@example.test' }],
    ['array quota target', []],
  ] as const) {
    it(`omits account header for ${name}, paired with a matching quota target`, async () => {
      const gw = await writerGateway();
      try {
        await positiveControl(gw);
        writeFileSync(bgModePath, JSON.stringify({ ...gptState, quota_target: quota }));
        const { q } = await observe();
        expect(recipe(gw.chatRequests[1])).toEqual({ model: 'codex/gpt-6.1-sol', effort: 'high', account: undefined });
        expectStored(q);
      } finally {
        await gw.close();
      }
    });
  }

  it('valid task without effort still routes GPT without adding reasoning_effort', async () => {
    const gw = await writerGateway();
    try {
      await positiveControl(gw);
      writeFileSync(bgModePath, JSON.stringify({ ...gptState, qualified_tasks: { 'claude-mem/observation': { model: 'codex/gpt-6.1-sol' } } }));
      const { q } = await observe();
      expect(recipe(gw.chatRequests[1])).toEqual({ model: 'codex/gpt-6.1-sol', effort: undefined, account: 'someone%40example.test' });
      expectStored(q);
    } finally {
      await gw.close();
    }
  });

  function refusal(status: number): FakeChatReply {
    return { status, headers: { 'Retry-After': '0' }, body: { error: { message: status === 429 ? 'rate limited before work' : 'insufficient_quota' } } };
  }

  async function fallbackControl(status: number, refused: FakeChatReply = refusal(status)) {
    writeFileSync(bgModePath, JSON.stringify(gptState));
    const gw = await writerGateway(req => req.body.model === 'codex/gpt-6.1-sol' ? refused : writerReply());
    try {
      const { q, session } = await observe();
      expect(gw.chatRequests.length).toBeGreaterThan(1);
      for (const req of gw.chatRequests.slice(0, -1)) expectGpt(req);
      expect(gw.chatRequests.filter(r => r.body.model === 'bg-mem-quality')).toHaveLength(1);
      expect(recipe(gw.chatRequests.at(-1)!)).toEqual({ model: 'bg-mem-quality', effort: undefined, account: undefined });
      expect(new Set(gw.chatRequests.map(r => r.headers['x-client-request-id'])).size).toBe(1);
      expect(session.paidSendBudget?.spentPaidSends).toBe(1);
      expectStored(q);
      expect(infoLogs.some(l => l.component === 'OBS_GATE' && l.message.includes('writer=bg-mem-quality'))).toBe(true);
    } finally {
      await gw.close();
    }
  }

  it('429 refusal falls back without charging the rejected GPT sends', async () => {
    await fallbackControl(429);
  });

  it('402 insufficient quota is rejected before work and falls back with the same budget', async () => {
    await fallbackControl(402);
  });

  it('a watched account in other letters is sent percent-encoded and reaches the gateway', async () => {
    const gw = await writerGateway();
    try {
      writeFileSync(bgModePath, JSON.stringify({ ...gptState, quota_target: { provider: 'codex', account: '北冥' } }));
      const { q, session } = await observe();
      expect(gw.chatRequests).toHaveLength(1);
      expect(recipe(gw.chatRequests[0])).toEqual({ model: 'codex/gpt-6.1-sol', effort: 'high', account: '%E5%8C%97%E5%86%A5' });
      expectStored(q);
      expect(session.paidSendBudget?.spentPaidSends).toBe(1);
    } finally {
      await gw.close();
    }
  });

  it('a session cancelled while its GPT send is refused gains no default writer send, paired with 402 fallback', async () => {
    await fallbackControl(402);
    const session = makeSession();
    const gw = await writerGateway(() => {
      session.abortController.abort();
      return refusal(402);
    });
    try {
      writeFileSync(bgModePath, JSON.stringify(gptState));
      const q = makeQueueAndDb([makeObs(1, { tool_name: 'Edit', tool_input: {}, tool_response: 'updated' })]);
      await new OpenAICompatProvider(q.dbManager, q.sessionManager).startSession(session).catch(() => undefined);
      expect(gw.chatRequests.length).toBeGreaterThan(0);
      for (const req of gw.chatRequests) expectGpt(req);
      expect(q.storeObservations).not.toHaveBeenCalled();
    } finally {
      await gw.close();
    }
  });

  for (const failure of ['output failure', 'ambiguous 500'] as const) {
    it(`${failure} does not gain a fallback send, paired with 429 fallback`, async () => {
      await fallbackControl(429);
      const gw = await writerGateway(() => failure === 'output failure'
        ? { body: { error: { code: 200, message: 'Unable to get json response - Expecting value' } } }
        : { status: 500, body: { error: { message: 'upstream hiccup' } } });
      try {
        writeFileSync(bgModePath, JSON.stringify(gptState));
        const q = makeQueueAndDb([makeObs(1, { tool_name: 'Edit', tool_input: {}, tool_response: 'updated' })]);
        const session = makeSession();
        const run = new OpenAICompatProvider(q.dbManager, q.sessionManager).startSession(session);
        if (failure === 'output failure') await run;
        else await expect(run).rejects.toThrow();
        expect(gw.chatRequests).toHaveLength(1);
        expectGpt(gw.chatRequests[0]);
        expect(q.storeObservations).not.toHaveBeenCalled();
        expect(session.paidSendBudget?.spentPaidSends).toBe(1);
      } finally {
        await gw.close();
      }
    });
  }

  it('a failure inside a 200 reply that nothing classifies gains no fallback send, paired with a declared 400 fallback', async () => {
    await fallbackControl(400, { status: 400, body: { error: { message: 'model is not supported' } } });
    const gw = await writerGateway(() => ({ body: { error: { message: 'the stream broke upstream' } } }));
    try {
      writeFileSync(bgModePath, JSON.stringify(gptState));
      const q = makeQueueAndDb([makeObs(1, { tool_name: 'Edit', tool_input: {}, tool_response: 'updated' })]);
      await new OpenAICompatProvider(q.dbManager, q.sessionManager).startSession(makeSession()).catch(() => undefined);
      expect(gw.chatRequests.length).toBeGreaterThan(0);
      for (const req of gw.chatRequests) expectGpt(req);
      expect(q.storeObservations).not.toHaveBeenCalled();
    } finally {
      await gw.close();
    }
  });

  it('a refusal that names the pinned account never carries it into the logs or the error', async () => {
    const named = { status: 429, headers: { 'Retry-After': '0' }, body: { error: { message: 'codex someone@example.test rests until 2026-11-17T03:00:00Z (rate limit)' } } };
    await fallbackControl(429, named);
    // Positive control: the warn line for the refused send was captured, so its absence below means something.
    expect(loudLogs.some(line => line.includes('rests until'))).toBe(true);
    for (const line of [...loudLogs, ...infoLogs.map(l => logText([l.message, l.context]))]) {
      expect(line.includes('someone@example.test') || line.includes('someone%40example.test')).toBe(false);
    }

    const options = { headers: { 'X-Magpie-Account': 'someone%40example.test' } };
    expect(withoutPinnedAccount('codex someone@example.test rests; pinned someone%40example.test', options)).toBe('codex [pinned account] rests; pinned [pinned account]');
    expect(withoutPinnedAccount('codex someone@example.test rests', undefined)).toBe('codex someone@example.test rests');
    expect(withoutPinnedAccount('100% of %zz', { headers: { 'X-Magpie-Account': '%zz' } })).toBe('100% of [pinned account]');
  });

  it('eco GPT refusal returns to bg-mem-eco then the existing Hydrogen tail without leaking options', async () => {
    writeFileSync(bgModePath, JSON.stringify({ ...gptState, mode: 'eco' }));
    const gw = await writerGateway(req => {
      if (req.body.model === 'codex/gpt-6.1-sol') return refusal(429);
      if (req.body.model === 'bg-mem-eco') return refusal(402);
      return writerReply();
    });
    try {
      const { q, session } = await observe();
      expectGpt(gw.chatRequests[0]);
      const defaults = gw.chatRequests.filter(r => r.body.model !== 'codex/gpt-6.1-sol');
      expect(defaults.map(recipe)).toEqual([
        { model: 'bg-mem-eco', effort: undefined, account: undefined },
        { model: 'bg-audit-primary', effort: undefined, account: undefined },
      ]);
      expect(session.paidSendBudget?.spentPaidSends).toBe(1);
      expectStored(q);
      expect(infoLogs.some(l => l.component === 'OBS_GATE' && l.message.includes('writer=bg-audit-primary'))).toBe(true);
    } finally {
      await gw.close();
    }
  });

  it('switching GPT back to current restores the full original body and headers', async () => {
    const gw = await writerGateway();
    try {
      writeFileSync(bgModePath, JSON.stringify({ ...gptState, profile: 'current' }));
      await observe();
      await positiveControl(gw);
      writeFileSync(bgModePath, JSON.stringify({ ...gptState, profile: 'current' }));
      const { q } = await observe();
      expect(gw.chatRequests).toHaveLength(3);
      expect(recipe(gw.chatRequests[0])).toEqual({ model: 'bg-mem-quality', effort: undefined, account: undefined });
      expect(gw.chatRequests[2].body).toEqual(gw.chatRequests[0].body);
      const headersWithoutTrace = (req: RecordedChatRequest) => Object.fromEntries(
        Object.entries(req.headers).filter(([name]) => name !== 'x-client-request-id'),
      );
      expect(headersWithoutTrace(gw.chatRequests[2])).toEqual(headersWithoutTrace(gw.chatRequests[0]));
      expectStored(q);
    } finally {
      await gw.close();
    }
  });

  it('GPT context overflow recycles without falling back, paired with 429 fallback', async () => {
    await fallbackControl(429);
    const gw = await writerGateway(() => ({ status: 400, body: { error: { message: 'context_length_exceeded' } } }));
    try {
      const { q, session } = await observe();
      expect(gw.chatRequests).toHaveLength(1);
      expectGpt(gw.chatRequests[0]);
      expect(session.abortReason).toBe('overflow:recycle');
      expect(session.consecutiveContextOverflows).toBe(1);
      expect(session.paidSendBudget?.spentPaidSends).toBe(0);
      expect(q.storeObservations).not.toHaveBeenCalled();
      expect(q.confirmCalls).toBe(0);
    } finally {
      await gw.close();
    }
  });

  it('GPT cannot replace an exhausted claimed-batch budget, paired with a fresh GPT budget', async () => {
    const gw = await writerGateway();
    try {
      await positiveControl(gw);
      const q = makeQueueAndDb([makeObs(1, { tool_name: 'Edit', tool_input: {}, tool_response: 'updated' })]);
      const budget = new PaidSendBudget(1);
      budget.recordPaidSend();
      budget.recordPaidSend();
      const session = makeSession({ paidSendBudget: budget });
      await expect(new OpenAICompatProvider(q.dbManager, q.sessionManager).startSession(session)).rejects.toThrow('already made 2 paid sends');
      expect(gw.chatRequests).toHaveLength(1);
      expect(session.paidSendBudget).toBe(budget);
      expect(budget.spentPaidSends).toBe(2);
      expect(q.storeObservations).not.toHaveBeenCalled();
    } finally {
      await gw.close();
    }
  });

  it('pure state parser preserves eco and rejects malformed or unqualified GPT state', () => {
    expect(parseBgWriterState(JSON.stringify(gptState))).toEqual({
      mode: 'quality',
      gptWriter: { model: 'codex/gpt-6.1-sol', requestOptions: { reasoningEffort: 'high', headers: { 'X-Magpie-Account': 'someone%40example.test' } } },
    });
    expect(parseBgWriterState('{')).toEqual({ mode: 'quality' });
    expect(parseBgWriterState(JSON.stringify({ ...gptState, profile: 'current', mode: 'eco' }))).toEqual({ mode: 'eco' });
  });

  it('routes observation writer model to bg-mem-quality or bg-mem-eco from bg-mode.json (missing/corrupt falls back to quality)', async () => {
    writeFileSync(bgModePath, JSON.stringify({ mode: 'eco' }));

    const gw = await startFakeGateway({
      onChat: (req) => ({
        body: {
          model: req.body.model === 'bg-mem-eco' ? 'grok/grok-4.7' : 'codex/gpt-6-luna:high',
          choices: [{ message: { role: 'assistant', content: '<skip_summary reason="noise" />' }, finish_reason: 'stop' }],
          usage: { prompt_tokens: 100, completion_tokens: 10, total_tokens: 110 },
        },
      }),
      onSystemOne: (_req, res) => {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ answers: { keep_observation: { noul: 0.9 } } }));
      },
    });

    try {
      process.env.CLAUDE_MEM_OPENAI_COMPAT_BASE_URL = gw.baseUrl;

      // 1) eco mode
      const q1 = makeQueueAndDb([makeObs(1, { tool_name: 'Read', tool_input: { file_path: 'a.ts' }, tool_response: 'ok' })]);
      await new OpenAICompatProvider(q1.dbManager, q1.sessionManager).startSession(makeSession());
      expect(gw.chatRequests[0].body.model).toBe('bg-mem-eco');

      // 2) quality mode
      writeFileSync(bgModePath, JSON.stringify({ mode: 'quality' }));
      const q2 = makeQueueAndDb([makeObs(2, { tool_name: 'Read', tool_input: { file_path: 'b.ts' }, tool_response: 'ok' })]);
      await new OpenAICompatProvider(q2.dbManager, q2.sessionManager).startSession(makeSession());
      expect(gw.chatRequests[1].body.model).toBe('bg-mem-quality');

      // 3) corrupt bg-mode.json falls back to quality
      writeFileSync(bgModePath, '{corrupt json');
      const q3 = makeQueueAndDb([makeObs(3, { tool_name: 'Read', tool_input: { file_path: 'c.ts' }, tool_response: 'ok' })]);
      await new OpenAICompatProvider(q3.dbManager, q3.sessionManager).startSession(makeSession());
      expect(gw.chatRequests[2].body.model).toBe('bg-mem-quality');

      // 4) missing bg-mode.json falls back to quality
      rmSync(bgModePath, { force: true });
      const q4 = makeQueueAndDb([makeObs(4, { tool_name: 'Read', tool_input: { file_path: 'd.ts' }, tool_response: 'ok' })]);
      await new OpenAICompatProvider(q4.dbManager, q4.sessionManager).startSession(makeSession());
      expect(gw.chatRequests[3].body.model).toBe('bg-mem-quality');

      // All 4 were <skip_summary /> so 0 System One calls were made.
      expect(gw.systemOneRequests).toHaveLength(0);
    } finally {
      await gw.close();
    }
  });

  it('keeps >=0.20 observations, drops <0.20 while preserving history & usage, and skips gate on <skip_summary />', async () => {
    writeFileSync(bgModePath, JSON.stringify({ mode: 'quality' }));
    let turn = 0;

    const gw = await startFakeGateway({
      onChat: () => {
        turn += 1;
        if (turn === 1) {
          return {
            body: {
              model: 'codex/gpt-6-luna:high',
              choices: [{ message: { role: 'assistant', content: OBSERVATION_XML }, finish_reason: 'stop' }],
              usage: { prompt_tokens: 500, completion_tokens: 80, total_tokens: 580 },
            },
          };
        }
        if (turn === 2) {
          return {
            body: {
              model: 'codex/gpt-6-luna:high',
              choices: [{ message: { role: 'assistant', content: OBSERVATION_XML }, finish_reason: 'stop' }],
              usage: { prompt_tokens: 750, completion_tokens: 60, total_tokens: 810 },
            },
          };
        }
        return {
          body: {
            model: 'codex/gpt-6-luna:high',
            choices: [{ message: { role: 'assistant', content: '<skip_summary reason="routine" />' }, finish_reason: 'stop' }],
            usage: { prompt_tokens: 900, completion_tokens: 15, total_tokens: 915 },
          },
        };
      },
      onSystemOne: (req, res) => {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        const score = gw.systemOneRequests.length === 1 ? 0.20 : 0.19;
      const questions = req.body.questions as { keep_observation?: { type?: string } } | undefined;
        expect(req.body.model).toBe('group/obs-gate');
        expect(req.headers.authorization).toBe('Bearer synthetic-gate-key');
        expect(questions?.keep_observation?.type).toBe('noul');
        res.end(JSON.stringify({ answers: { keep_observation: { noul: score } } }));
      },
    });

    try {
      // Use rootUrl (without /v1) here to verify resolveSystemOneUrl still resolves /v1/systemone!
      process.env.CLAUDE_MEM_OPENAI_COMPAT_BASE_URL = gw.rootUrl;

      // Turn 1: prob = 0.20 -> KEEP & store (serialized JSON string inputs must NOT be double-escaped)
      const qKeep = makeQueueAndDb([
        makeObs(10, {
          tool_name: 'Edit',
          tool_input: { file_path: 'src/a.ts', longParam: 'p'.repeat(1500) },
          tool_response: 'r'.repeat(2500),
        }),
      ]);
      const keepSession = makeSession({ userPrompt: 'u'.repeat(800) });
      await new OpenAICompatProvider(qKeep.dbManager, qKeep.sessionManager).startSession(keepSession);

      expect(gw.systemOneRequests).toHaveLength(1);
      const sentState = gw.systemOneRequests[0].body.state as Record<string, string>;
      expect(sentState.user_request).toBe('u'.repeat(800));
      expect(sentState.tool_name).toBe('Edit');
      expect(sentState.parameters).toBe(JSON.stringify({ file_path: 'src/a.ts', longParam: 'p'.repeat(1500) }));
      expect(sentState.outcome).toBe('r'.repeat(2500));
      expect(sentState.candidate_draft).toBe(OBSERVATION_XML);
      expect(qKeep.storeObservations).toHaveBeenCalledTimes(1);

      // Turn 2 & 3 in one session: Turn 2 prob = 0.19 -> DROP (no DB write, history + usage kept, overflow debt cleared),
      // Turn 3 <skip_summary /> -> 0 extra System One calls and multi-turn history includes Turn 2 draft!
      const qDropThenSkip = makeQueueAndDb([
        makeObs(11, { tool_name: 'Bash', tool_input: { command: 'ls' }, tool_response: 'a\nb' }),
        makeObs(12, { tool_name: 'Read', tool_input: { file_path: 'README.md' }, tool_response: 'hi' }),
      ]);
      const dropSession = makeSession({ consecutiveContextOverflows: 2 });
      await new OpenAICompatProvider(qDropThenSkip.dbManager, qDropThenSkip.sessionManager).startSession(dropSession);

      // Only 1 more System One call (for Turn 2); Turn 3 (<skip_summary />) made 0 System One calls.
      expect(gw.systemOneRequests).toHaveLength(2);
      const storedObsCount = qDropThenSkip.storeObservations.mock.calls.reduce(
        (sum: number, call: unknown[]) => sum + ((call[2] as unknown[])?.length ?? 0),
        0,
      );
      expect(storedObsCount).toBe(0);
      expect(qDropThenSkip.confirmCalls).toBe(2);
      expect(dropSession.consecutiveContextOverflows).toBe(0);
      expect(dropSession.lastContextTokens).toBe(900);
      expect(dropSession.cumulativeInputTokens).toBe(750 + 900);
      expect(dropSession.cumulativeOutputTokens).toBe(60 + 15);
      expect(dropSession.conversationHistory.some(m => m.role === 'assistant' && m.content === OBSERVATION_XML)).toBe(true);

      // Separate single-turn drop check so lastUsage, lastContextTokens & debt reset right after DROP are asserted directly.
      turn = 1; // next chat returns Turn 2 response
      const qSingleDrop = makeQueueAndDb([
        makeObs(13, { tool_name: 'Bash', tool_input: { command: 'pwd' }, tool_response: '/tmp' }),
      ]);
      const singleDropSession = makeSession({ consecutiveContextOverflows: 1 });
      await new OpenAICompatProvider(qSingleDrop.dbManager, qSingleDrop.sessionManager).startSession(singleDropSession);
      expect(qSingleDrop.storeObservations).not.toHaveBeenCalled();
      expect(singleDropSession.consecutiveContextOverflows).toBe(0);
      expect(singleDropSession.lastContextTokens).toBe(750);
      expect(singleDropSession.lastUsage).toEqual({ input: 750, output: 60 });
      expect(singleDropSession.conversationHistory.map(m => m.role)).toEqual(['user', 'user', 'assistant']);

      // Receipt check: single-line OBS_GATE logs for KEEP, DROP, WRITER_SKIP
      const gateReceipts = infoLogs.filter(l => l.component === 'OBS_GATE');
      expect(gateReceipts.length).toBeGreaterThanOrEqual(4);
      for (const r of gateReceipts) {
        expect(r.message).not.toContain('\n');
        expect(r.message).toContain('mode=quality');
        expect(r.message).toContain('writer=codex/gpt-6-luna:high');
        expect(r.message).toContain('gate_verdict=');
        expect(r.message).toContain('gate_model=');
        expect(r.message).toContain('prob=');
      }
      expect(gateReceipts.some(r => r.message.includes('gate_verdict=KEEP') && r.message.includes('gate_model=group/obs-gate'))).toBe(true);
      expect(gateReceipts.some(r => r.message.includes('gate_verdict=DROP'))).toBe(true);
      expect(gateReceipts.some(r => r.message.includes('gate_verdict=WRITER_SKIP') && r.message.includes('gate_model=none') && r.message.includes('prob=n/a'))).toBe(true);
    } finally {
      await gw.close();
    }
  });

  it('fails open with GATE_ERROR and stores draft when /v1/systemone omits noul or returns non-finite value (F2)', async () => {
    writeFileSync(bgModePath, JSON.stringify({ mode: 'quality' }));
    const badPayloads = [
      {},
      { answers: { keep_observation: { noul: '0.85' } } },
      { answers: { keep_observation: { noul: null } } },
    ];
    let idx = 0;

    const gw = await startFakeGateway({
      onChat: () => ({
        body: {
          model: 'codex/gpt-6-luna:high',
          choices: [{ message: { role: 'assistant', content: OBSERVATION_XML }, finish_reason: 'stop' }],
          usage: { prompt_tokens: 200, completion_tokens: 40, total_tokens: 240 },
        },
      }),
      onSystemOne: (_req, res) => {
        const payload = badPayloads[idx++] ?? {};
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(payload));
      },
    });

    try {
      process.env.CLAUDE_MEM_OPENAI_COMPAT_BASE_URL = gw.baseUrl;

      for (let i = 0; i < badPayloads.length; i++) {
        const q = makeQueueAndDb([
          makeObs(20 + i, { tool_name: 'Edit', tool_input: { file_path: 'x.ts' }, tool_response: 'ok' }),
        ]);
        await new OpenAICompatProvider(q.dbManager, q.sessionManager).startSession(makeSession());
        expect(q.storeObservations).toHaveBeenCalledTimes(1);
        expect(q.confirmCalls).toBe(1);
      }

      const errorReceipts = infoLogs.filter(l => l.component === 'OBS_GATE' && l.message.includes('gate_verdict=GATE_ERROR'));
      expect(errorReceipts).toHaveLength(badPayloads.length);
      for (const r of errorReceipts) {
        expect(r.message).toContain('prob=n/a');
      }
    } finally {
      await gw.close();
    }
  });

  it('single-leg fail-open stores draft on 404, 429, 500, and timeout without blocking the queue (F4)', async () => {
    writeFileSync(bgModePath, JSON.stringify({ mode: 'eco' }));
    let mode: '404' | '429' | '500' | 'timeout' = '404';

    const gw = await startFakeGateway({
      onChat: () => ({
        body: {
          model: 'grok/grok-4.7',
          choices: [{ message: { role: 'assistant', content: OBSERVATION_XML }, finish_reason: 'stop' }],
          usage: { prompt_tokens: 300, completion_tokens: 50, total_tokens: 350 },
        },
      }),
      onSystemOne: (_req, res) => {
        if (mode === 'timeout') {
          // Hang until the 5000ms gate timeout aborts the request.
          return;
        }
        const status = Number(mode);
        res.writeHead(status, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: { message: `status ${status}` } }));
      },
    });

    try {
      process.env.CLAUDE_MEM_OPENAI_COMPAT_BASE_URL = gw.baseUrl;

      for (const nextMode of ['404', '429', '500', 'timeout'] as const) {
        mode = nextMode;
        const beforeCalls = gw.systemOneRequests.length;
        const q = makeQueueAndDb([
          makeObs(30, { tool_name: 'Edit', tool_input: { file_path: 'y.ts' }, tool_response: 'ok' }),
        ]);
        await new OpenAICompatProvider(q.dbManager, q.sessionManager).startSession(makeSession());
        expect(gw.systemOneRequests.length - beforeCalls).toBe(1);
        expect(q.storeObservations).toHaveBeenCalledTimes(1);
        expect(q.confirmCalls).toBe(1);
      }

      const errorReceipts = infoLogs.filter(l => l.component === 'OBS_GATE' && l.message.includes('gate_verdict=GATE_ERROR'));
      expect(errorReceipts).toHaveLength(4);
      expect(errorReceipts.every(r => r.message.includes('mode=eco'))).toBe(true);
    } finally {
      await gw.close();
    }
  }, 10_000);

  it('GPT profile leaves Stop summary and field compression on the original recipe (F5)', async () => {
    writeFileSync(bgModePath, JSON.stringify(gptState));

    const gw = await startFakeGateway({
      onChat: (req) => {
        const msgs = (req.body.messages as Array<{ content?: string }> | undefined) ?? [];
        const lastUser = msgs[msgs.length - 1]?.content ?? '';
        const isCondense = typeof lastUser === 'string' && lastUser.includes('Condense the tool payload below');
        if (isCondense) {
          return {
            body: {
              model: req.body.model,
              choices: [{ message: { role: 'assistant', content: 'condensed output summary' }, finish_reason: 'stop' }],
              usage: { prompt_tokens: 100, completion_tokens: 10, total_tokens: 110 },
            },
          };
        }
        if (req.body.model === 'mem-observer') {
          return {
            body: {
              model: req.body.model,
              choices: [{ message: { role: 'assistant', content: SUMMARY_XML }, finish_reason: 'stop' }],
              usage: { prompt_tokens: 400, completion_tokens: 70, total_tokens: 470 },
            },
          };
        }
        return {
          body: {
            model: 'codex/gpt-6-luna:high',
            choices: [{ message: { role: 'assistant', content: OBSERVATION_XML }, finish_reason: 'stop' }],
            usage: { prompt_tokens: 500, completion_tokens: 80, total_tokens: 580 },
          },
        };
      },
      onSystemOne: (_req, res) => {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ answers: { keep_observation: { noul: 0.9 } } }));
      },
    });

    try {
      process.env.CLAUDE_MEM_OPENAI_COMPAT_BASE_URL = gw.baseUrl;

      // 1) Stop <summary> alone -> 0 /v1/systemone calls, keeps mem-observer channel, stores summary
      const qSummary = makeQueueAndDb([makeSummary(40, 'finished work')]);
      const summarySession = makeSession();
      await new OpenAICompatProvider(qSummary.dbManager, qSummary.sessionManager).startSession(summarySession);

      expect(gw.chatRequests).toHaveLength(1);
      expect(recipe(gw.chatRequests[0])).toEqual({ model: 'mem-observer', effort: undefined, account: undefined });
      expect(gw.systemOneRequests).toHaveLength(0);
      expect(qSummary.storeObservations).toHaveBeenCalledTimes(1);
      expect(summarySession.lastContextTokens).toBe(400);

      // 2) Observation with oversized tool_response -> compressField stays on mem-observer & does not call /v1/systemone;
      //    only the observation draft calls /v1/systemone once.
      const qObsWithCompress = makeQueueAndDb([
        makeObs(41, {
          tool_name: 'Read',
          tool_input: { file_path: 'huge.log' },
          tool_response: 'line\n'.repeat(6000),
        }),
      ]);
      await new OpenAICompatProvider(qObsWithCompress.dbManager, qObsWithCompress.sessionManager).startSession(makeSession());

      expect(gw.chatRequests).toHaveLength(3); // summary + compressField + observation
      expect(recipe(gw.chatRequests[1])).toEqual({ model: 'mem-observer', effort: undefined, account: undefined });
      expectGpt(gw.chatRequests[2]);
      expect(gw.systemOneRequests).toHaveLength(1);
    } finally {
      await gw.close();
    }
  });

  it('eco Hydrogen fallback yields when >=8 foreground Google calls are active, and runs below 8', async () => {
    writeFileSync(bgModePath, JSON.stringify({ mode: 'eco' }));
    let fgCount = 8;

    const gw = await startFakeGateway({
      onConcurrency: () => ({
        status: 200,
        body: {
          active: Array.from({ length: fgCount }, (_, i) => ({
            id: `fg-${i}`,
            agent: 'claude-code',
            provider: 'google',
            model: 'google/gemini-4-carbon',
          })),
        },
      }),
      onChat: (req) => {
        if (req.body.model === 'bg-mem-eco') {
          return { status: 402, body: { error: { message: 'swe-2 expired and grok gated' } } };
        }
        return {
          body: {
            model: 'google/gemini-4-hydrogen:high',
            choices: [{ message: { role: 'assistant', content: OBSERVATION_XML }, finish_reason: 'stop' }],
            usage: { prompt_tokens: 220, completion_tokens: 45, total_tokens: 265 },
          },
        };
      },
      onSystemOne: (_req, res) => {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ answers: { keep_observation: { noul: 0.85 } } }));
      },
    });

    try {
      process.env.CLAUDE_MEM_OPENAI_COMPAT_BASE_URL = gw.baseUrl;
      process.env.CLAUDE_MEM_MAGPIE_CONCURRENCY_URL = gw.concurrencyUrl;

      // 1) >=8 Foreground Google active -> yields (pauses with rate_limit:rate_limit, 0 Hydrogen calls, batch unconfirmed)
      const qBusy = makeQueueAndDb([makeObs(50, { tool_name: 'Edit', tool_input: { file_path: 'fg.ts' }, tool_response: 'ok' })]);
      const busySession = makeSession();
      await expect(
        new OpenAICompatProvider(qBusy.dbManager, qBusy.sessionManager).startSession(busySession),
      ).rejects.toThrow();
      expect(busySession.abortReason).toBe('rate_limit:rate_limit');
      expect(qBusy.confirmCalls).toBe(0);
      expect(gw.chatRequests.map(r => r.body.model)).toEqual(['bg-mem-eco']);

      // 2) <8 Foreground Google active (7 active) -> falls back cleanly to bg-audit-primary (Hydrogen) and stores observation
      fgCount = 7;
      const qModerate = makeQueueAndDb([makeObs(51, { tool_name: 'Edit', tool_input: { file_path: 'fg.ts' }, tool_response: 'ok' })]);
      await new OpenAICompatProvider(qModerate.dbManager, qModerate.sessionManager).startSession(makeSession());
      expect(gw.chatRequests.slice(1).map(r => r.body.model)).toEqual(['bg-mem-eco', 'bg-audit-primary']);
      expect(qModerate.confirmCalls).toBe(1);
      expect(qModerate.storeObservations).toHaveBeenCalledTimes(1);
    } finally {
      await gw.close();
    }
  });
});
