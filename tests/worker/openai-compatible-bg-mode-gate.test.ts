import { afterEach, beforeEach, describe, expect, it, mock, spyOn } from 'bun:test';
import { Database } from 'bun:sqlite';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'fs';
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { tmpdir } from 'os';
import { join } from 'path';
import { ModeManager } from '../../src/services/domain/ModeManager.js';
import { SessionStore } from '../../src/services/sqlite/SessionStore.js';
import { setIngestContext, ingestObservation } from '../../src/services/worker/http/shared.js';
import { OpenAICompatProvider } from '../../src/services/worker/OpenAICompatProvider.js';
import type { DatabaseManager } from '../../src/services/worker/DatabaseManager.js';
import type { SessionManager } from '../../src/services/worker/SessionManager.js';
import type { ActiveSession, PendingMessageWithId } from '../../src/services/worker-types.js';
import { SettingsDefaultsManager } from '../../src/shared/SettingsDefaultsManager.js';
import { USER_SETTINGS_PATH } from '../../src/shared/paths.js';
import { logger } from '../../src/utils/logger.js';

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
    lastGeneratorActivity: Date.now(),
    ...overrides,
  };
}

interface RecordedChatRequest {
  url: string;
  headers: Record<string, string | string[] | undefined>;
  body: Record<string, any>;
}

interface RecordedSystemOneRequest {
  url: string;
  headers: Record<string, string | string[] | undefined>;
  body: Record<string, any>;
}

async function startFakeGateway(handlers: {
  onChat: (req: RecordedChatRequest) => { status?: number; body: Record<string, any> } | Promise<{ status?: number; body: Record<string, any> }>;
  onSystemOne: (req: RecordedSystemOneRequest, res: ServerResponse) => void | Promise<void>;
}) {
  const chatRequests: RecordedChatRequest[] = [];
  const systemOneRequests: RecordedSystemOneRequest[] = [];

  const readJsonBody = (req: IncomingMessage): Promise<Record<string, any>> =>
    new Promise((resolve, reject) => {
      const chunks: Buffer[] = [];
      req.on('data', (chunk: Buffer) => chunks.push(chunk));
      req.on('end', () => {
        const text = Buffer.concat(chunks).toString('utf-8');
        try {
          resolve(text ? JSON.parse(text) : {});
        } catch (err) {
          reject(err);
        }
      });
      req.on('error', reject);
    });

  const server = createServer(async (req, res) => {
    const url = req.url ?? '/';
    const body = await readJsonBody(req);
    if (url.endsWith('/chat/completions')) {
      const record: RecordedChatRequest = { url, headers: req.headers, body };
      chatRequests.push(record);
      const reply = await handlers.onChat(record);
      res.writeHead(reply.status ?? 200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(reply.body));
      return;
    }
    if (url.endsWith('/systemone')) {
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
    baseUrl: `http://127.0.0.1:${port}/v1`,
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
  const storeObservations = mock(() => ({
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
  'CLAUDE_MEM_SYSTEMONE_URL',
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
      } as any,
      dbManager: { getSessionStore: () => store } as any,
      eventBroadcaster: { broadcastObservationQueued: mock(() => {}) } as any,
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
    spies = [
      spyOn(ModeManager, 'getInstance').mockImplementation(() => ({
        getActiveMode: () => mockMode,
        loadMode: () => {},
      }) as any),
      spyOn(logger, 'debug').mockImplementation(() => {}),
      spyOn(logger, 'info').mockImplementation((component: any, message: string, context?: unknown) => {
        infoLogs.push({ component, message, context });
      }),
      spyOn(logger, 'warn').mockImplementation(() => {}),
      spyOn(logger, 'error').mockImplementation(() => {}),
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
      const q1 = makeQueueAndDb([
        { id: 1, type: 'observation', tool_name: 'Read', tool_input: { file_path: 'a.ts' }, tool_response: 'ok', prompt_number: 2 } as PendingMessageWithId,
      ]);
      await new OpenAICompatProvider(q1.dbManager, q1.sessionManager).startSession(makeSession());
      expect(gw.chatRequests[0].body.model).toBe('bg-mem-eco');

      // 2) quality mode
      writeFileSync(bgModePath, JSON.stringify({ mode: 'quality' }));
      const q2 = makeQueueAndDb([
        { id: 2, type: 'observation', tool_name: 'Read', tool_input: { file_path: 'b.ts' }, tool_response: 'ok', prompt_number: 2 } as PendingMessageWithId,
      ]);
      await new OpenAICompatProvider(q2.dbManager, q2.sessionManager).startSession(makeSession());
      expect(gw.chatRequests[1].body.model).toBe('bg-mem-quality');

      // 3) corrupt bg-mode.json falls back to quality
      writeFileSync(bgModePath, '{corrupt json');
      const q3 = makeQueueAndDb([
        { id: 3, type: 'observation', tool_name: 'Read', tool_input: { file_path: 'c.ts' }, tool_response: 'ok', prompt_number: 2 } as PendingMessageWithId,
      ]);
      await new OpenAICompatProvider(q3.dbManager, q3.sessionManager).startSession(makeSession());
      expect(gw.chatRequests[2].body.model).toBe('bg-mem-quality');

      // 4) missing bg-mode.json falls back to quality
      rmSync(bgModePath, { force: true });
      const q4 = makeQueueAndDb([
        { id: 4, type: 'observation', tool_name: 'Read', tool_input: { file_path: 'd.ts' }, tool_response: 'ok', prompt_number: 2 } as PendingMessageWithId,
      ]);
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
        expect(req.body.model).toBe('openrouter/perplexity/pplx-decider-v1.3:free');
        expect(req.headers.authorization).toBe('Bearer synthetic-gate-key');
        expect(req.body.questions?.keep_observation?.type).toBe('noul');
        res.end(JSON.stringify({ answers: { keep_observation: { noul: score } } }));
      },
    });

    try {
      process.env.CLAUDE_MEM_OPENAI_COMPAT_BASE_URL = gw.baseUrl;

      // Turn 1: prob = 0.20 -> KEEP & store
      const qKeep = makeQueueAndDb([
        {
          id: 10,
          type: 'observation',
          tool_name: 'Edit',
          tool_input: { file_path: 'src/a.ts', longParam: 'p'.repeat(1500) },
          tool_response: 'r'.repeat(2500),
          prompt_number: 2,
        } as PendingMessageWithId,
      ]);
      const keepSession = makeSession({ userPrompt: 'u'.repeat(800) });
      await new OpenAICompatProvider(qKeep.dbManager, qKeep.sessionManager).startSession(keepSession);

      expect(gw.systemOneRequests).toHaveLength(1);
      const sentState = gw.systemOneRequests[0].body.state;
      expect(sentState.user_request).toBe('u'.repeat(800));
      expect(sentState.tool_name).toBe('Edit');
      expect(sentState.parameters).toBe(JSON.stringify({ file_path: 'src/a.ts', longParam: 'p'.repeat(1500) }));
      expect(sentState.outcome).toBe(JSON.stringify('r'.repeat(2500)));
      expect(sentState.candidate_draft).toBe(OBSERVATION_XML);
      expect(qKeep.storeObservations).toHaveBeenCalledTimes(1);

      // Turn 2 & 3 in one session: Turn 2 prob = 0.19 -> DROP (no DB write, history + usage kept),
      // Turn 3 <skip_summary /> -> 0 extra System One calls and multi-turn history includes Turn 2 draft!
      const qDropThenSkip = makeQueueAndDb([
        { id: 11, type: 'observation', tool_name: 'Bash', tool_input: { command: 'ls' }, tool_response: 'a\nb', prompt_number: 2 } as PendingMessageWithId,
        { id: 12, type: 'observation', tool_name: 'Read', tool_input: { file_path: 'README.md' }, tool_response: 'hi', prompt_number: 2 } as PendingMessageWithId,
      ]);
      const dropSession = makeSession();
      await new OpenAICompatProvider(qDropThenSkip.dbManager, qDropThenSkip.sessionManager).startSession(dropSession);

      // Only 1 more System One call (for Turn 2); Turn 3 (<skip_summary />) made 0 System One calls.
      expect(gw.systemOneRequests).toHaveLength(2);
      const storedObsCount = qDropThenSkip.storeObservations.mock.calls.reduce(
        (sum, call) => sum + ((call[2] as unknown[])?.length ?? 0),
        0,
      );
      expect(storedObsCount).toBe(0);
      expect(qDropThenSkip.confirmCalls).toBe(2);
      expect(dropSession.lastContextTokens).toBe(900);
      expect(dropSession.cumulativeInputTokens).toBe(750 + 900);
      expect(dropSession.cumulativeOutputTokens).toBe(60 + 15);
      expect(dropSession.conversationHistory.some(m => m.role === 'assistant' && m.content === OBSERVATION_XML)).toBe(true);

      // Separate single-turn drop check so lastUsage & lastContextTokens right after DROP are asserted directly.
      turn = 1; // next chat returns Turn 2 response
      const qSingleDrop = makeQueueAndDb([
        { id: 13, type: 'observation', tool_name: 'Bash', tool_input: { command: 'pwd' }, tool_response: '/tmp', prompt_number: 2 } as PendingMessageWithId,
      ]);
      const singleDropSession = makeSession();
      await new OpenAICompatProvider(qSingleDrop.dbManager, qSingleDrop.sessionManager).startSession(singleDropSession);
      expect(qSingleDrop.storeObservations).not.toHaveBeenCalled();
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
      expect(gateReceipts.some(r => r.message.includes('gate_verdict=KEEP') && r.message.includes('gate_model=openrouter/perplexity/pplx-decider-v1.3:free'))).toBe(true);
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
          { id: 20 + i, type: 'observation', tool_name: 'Edit', tool_input: { file_path: 'x.ts' }, tool_response: 'ok', prompt_number: 2 } as PendingMessageWithId,
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
          { id: 30, type: 'observation', tool_name: 'Edit', tool_input: { file_path: 'y.ts' }, tool_response: 'ok', prompt_number: 2 } as PendingMessageWithId,
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

  it('passes Stop <summary> and compressField straight through with 0 /v1/systemone calls (F5)', async () => {
    writeFileSync(bgModePath, JSON.stringify({ mode: 'quality' }));

    const gw = await startFakeGateway({
      onChat: (req) => {
        const lastUser = req.body.messages[req.body.messages.length - 1]?.content ?? '';
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
      const qSummary = makeQueueAndDb([
        { id: 40, type: 'summarize', last_assistant_message: 'finished work', prompt_number: 2 } as unknown as PendingMessageWithId,
      ]);
      const summarySession = makeSession();
      await new OpenAICompatProvider(qSummary.dbManager, qSummary.sessionManager).startSession(summarySession);

      expect(gw.chatRequests).toHaveLength(1);
      expect(gw.chatRequests[0].body.model).toBe('mem-observer');
      expect(gw.systemOneRequests).toHaveLength(0);
      expect(qSummary.storeObservations).toHaveBeenCalledTimes(1);
      expect(summarySession.lastContextTokens).toBe(400);

      // 2) Observation with oversized tool_response -> compressField stays on mem-observer & does not call /v1/systemone;
      //    only the observation draft calls /v1/systemone once.
      const qObsWithCompress = makeQueueAndDb([
        {
          id: 41,
          type: 'observation',
          tool_name: 'Read',
          tool_input: { file_path: 'huge.log' },
          tool_response: 'line\n'.repeat(6000),
          prompt_number: 2,
        } as PendingMessageWithId,
      ]);
      await new OpenAICompatProvider(qObsWithCompress.dbManager, qObsWithCompress.sessionManager).startSession(makeSession());

      expect(gw.chatRequests).toHaveLength(3); // summary + compressField + observation
      expect(gw.chatRequests[1].body.model).toBe('mem-observer');
      expect(gw.chatRequests[2].body.model).toBe('bg-mem-quality');
      expect(gw.systemOneRequests).toHaveLength(1);
    } finally {
      await gw.close();
    }
  });
});
