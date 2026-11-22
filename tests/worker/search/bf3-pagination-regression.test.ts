import { afterEach, beforeEach, describe, expect, it } from 'bun:test';
import { SessionStore } from '../../../src/services/sqlite/SessionStore.js';
import { SessionSearch } from '../../../src/services/sqlite/SessionSearch.js';
import { SearchOrchestrator } from '../../../src/services/worker/search/SearchOrchestrator.js';
import { SearchManager } from '../../../src/services/worker/SearchManager.js';
import { FormattingService } from '../../../src/services/worker/FormattingService.js';
import { TimelineService } from '../../../src/services/worker/TimelineService.js';

interface Seeded {
  observation: number;
  summary: number;
  prompt: number;
  epoch: number;
}

let store: SessionStore;
let search: SessionSearch;
const now = Date.now() - 100_000;

function seed(label: string, epoch: number, project = 'repo', file = 'src/file.ts', mergedInto?: string): Seeded {
  const contentId = `content-${label}`;
  const memoryId = `memory-${label}`;
  const sessionId = store.createSDKSession(contentId, project, `needle ${label}`);
  store.ensureMemorySessionIdRegistered(sessionId, memoryId);
  const observation = store.storeObservation(memoryId, project, {
    type: 'discovery', title: `needle ${label}`, subtitle: null, narrative: `needle ${label}`,
    facts: [], concepts: [], files_read: [file], files_modified: [],
  }, 1, 0, epoch).id;
  const summary = store.importSessionSummary({
    memory_session_id: memoryId, project, request: `needle ${label}`,
    investigated: null, learned: null, completed: null, next_steps: null, notes: null,
    files_read: JSON.stringify([file]), files_edited: null, prompt_number: 1,
    discovery_tokens: 0, created_at: new Date(epoch).toISOString(), created_at_epoch: epoch,
  }).id;
  store.saveUserPrompt(contentId, 1, `needle ${label}`, sessionId);
  store.db.prepare('UPDATE user_prompts SET created_at = ?, created_at_epoch = ? WHERE session_db_id = ?')
    .run(new Date(epoch).toISOString(), epoch, sessionId);
  const prompt = (store.db.prepare('SELECT id FROM user_prompts WHERE session_db_id = ?').get(sessionId) as { id: number }).id;
  if (mergedInto) {
    store.db.prepare('UPDATE observations SET merged_into_project = ? WHERE id = ?').run(mergedInto, observation);
    store.db.prepare('UPDATE session_summaries SET merged_into_project = ? WHERE id = ?').run(mergedInto, summary);
  }
  return { observation, summary, prompt, epoch };
}

function fakeChroma(rows: Array<{ id: number; kind: 'observation' | 'session_summary' | 'user_prompt'; epoch: number }>) {
  return {
    queryChroma: async (_query: string, limit: number) => {
      const selected = rows.slice(0, limit);
      return {
        ids: selected.map(row => row.id),
        distances: selected.map(() => 0.1),
        metadatas: selected.map(row => ({ sqlite_id: row.id, doc_type: row.kind, created_at_epoch: row.epoch, project: 'repo', memory_session_id: 'fixture' })),
      };
    },
  };
}

function vectors(rows: Seeded[]) {
  return rows.flatMap(row => [
    { id: row.observation, kind: 'observation' as const, epoch: row.epoch },
    { id: row.summary, kind: 'session_summary' as const, epoch: row.epoch },
    { id: row.prompt, kind: 'user_prompt' as const, epoch: row.epoch },
  ]);
}

function manager(chroma: unknown): SearchManager {
  return new SearchManager(search, store, chroma as never, new FormattingService(), new TimelineService());
}

beforeEach(() => {
  store = new SessionStore(':memory:');
  search = new SessionSearch(store.db);
});
afterEach(() => store.close());

describe('file, prompt and cross-backend pagination', () => {
  it('uses descending id to break prompt timestamp ties with and without query text', () => {
    const rows = [seed('a', now), seed('b', now), seed('c', now)];
    for (const query of [undefined, 'needle']) {
      expect(search.searchUserPrompts(query, { project: 'repo', orderBy: 'date_desc', limit: 2, offset: 1 }).map(row => row.id))
        .toEqual([rows[1].prompt, rows[0].prompt]);
    }
  });

  for (const category of ['observations', 'sessions', 'prompts'] as const) {
    it(`preserves semantic ${category} ranking when applying offset`, async () => {
      const rows = [seed('a', now), seed('b', now + 1), seed('c', now + 2), seed('d', now + 3)];
      const ranked = [rows[2], rows[0], rows[3], rows[1]];
      const result = await manager(fakeChroma(vectors(ranked))).search({
        query: 'needle', project: 'repo', format: 'json', limit: 2, offset: 1, orderBy: 'relevance',
      });
      const key = category === 'observations' ? 'observation' : category === 'sessions' ? 'summary' : 'prompt';
      expect(result[category].map((row: { id: number }) => row.id)).toEqual(ranked.slice(1, 3).map(row => row[key]));
    });
  }

  it('applies the same Chroma offset through the orchestrator', async () => {
    const rows = [seed('a', now), seed('b', now + 1), seed('c', now + 2)];
    const result = await new SearchOrchestrator(search, store, fakeChroma(vectors(rows)) as never).search({
      query: 'needle', project: 'repo', limit: 1, offset: 1, orderBy: 'date_asc',
    });
    expect(result.results.observations.map(row => row.id)).toEqual([rows[1].observation]);
    expect(result.results.sessions.map(row => row.id)).toEqual([rows[1].summary]);
    expect(result.results.prompts.map(row => row.id)).toEqual([rows[1].prompt]);
  });

  for (const orderBy of ['date_asc', 'date_desc'] as const) {
    it(`uses id to break timestamp ties for file summaries and prompts (${orderBy})`, () => {
      const rows = [seed('a', now), seed('b', now), seed('c', now)];
      const expected = orderBy === 'date_asc' ? rows : [...rows].reverse();
      for (let offset = 0; offset < 3; offset++) {
        expect(search.findByFile('src/file.ts', { project: 'repo', orderBy, limit: 1, offset }).sessions.map(row => row.id))
          .toEqual([expected[offset].summary]);
        for (const query of [undefined, 'needle']) {
          expect(search.searchUserPrompts(query, { project: 'repo', orderBy, limit: 1, offset }).map(row => row.id))
            .toEqual([expected[offset].prompt]);
        }
      }
    });
  }

  for (const folder of [false, true]) {
    it(`paginates after Chroma ranking and SQLite backfill (${folder ? 'folder' : 'file'})`, async () => {
      const rows = [seed('a', now), seed('b', now + 1), seed('c', now + 2), seed('d', now + 3)];
      const chroma = fakeChroma([
        { id: rows[0].observation, kind: 'observation', epoch: rows[0].epoch },
        { id: rows[3].observation, kind: 'observation', epoch: rows[3].epoch },
      ]);
      const orchestrator = new SearchOrchestrator(search, store, chroma as never);
      const options = { project: 'repo', isFolder: folder, limit: 2, orderBy: 'relevance' };
      const first = await orchestrator.findByFile(folder ? 'src' : 'src/file.ts', options);
      const second = await orchestrator.findByFile(folder ? 'src' : 'src/file.ts', { ...options, offset: 2 });
      expect(first.observations.map(row => row.id)).toEqual([rows[0].observation, rows[3].observation]);
      expect(second.observations.map(row => row.id)).toEqual([rows[2].observation, rows[1].observation]);
      expect(second.sessions.map(row => row.id)).toEqual([rows[1].summary, rows[0].summary]);
    });
  }

  for (const orderBy of ['date_asc', 'date_desc', 'relevance'] as const) {
    it(`applies offset once per category on SQLite and Chroma (${orderBy})`, async () => {
      const rows = [seed('a', now), seed('b', now + 1), seed('c', now + 2), seed('d', now + 3)];
      const ranked = [...rows].reverse();
      const args = { query: 'needle', project: 'repo', format: 'json', limit: 2, offset: 1, orderBy };
      for (const chroma of [null, fakeChroma(vectors(ranked))]) {
        const service = manager(chroma);
        const all = await service.search({ ...args, limit: 4, offset: 0 });
        const result = await service.search(args);
        for (const category of ['observations', 'sessions', 'prompts'] as const) {
          expect(all[category]).toHaveLength(4);
          expect(result[category].map((row: { id: number }) => row.id))
            .toEqual(all[category].slice(1, 3).map((row: { id: number }) => row.id));
        }
        if (orderBy !== 'relevance') {
          const expected = orderBy === 'date_asc' ? rows : ranked;
          expect(result.observations.map((row: { id: number }) => row.id)).toEqual(expected.slice(1, 3).map(row => row.observation));
        }
      }
    });
  }

  it('keeps category offsets when semantic hits and SQLite backfill are mixed', async () => {
    const rows = [seed('a', now), seed('b', now + 1), seed('c', now + 2), seed('d', now + 3)];
    const chroma = fakeChroma(rows.map(row => ({ id: row.observation, kind: 'observation', epoch: row.epoch })));
    const args = { query: 'needle', project: 'repo', format: 'json', limit: 2, offset: 1, orderBy: 'date_asc' };
    const result = await manager(chroma).search(args);
    expect(result.observations.map((row: { id: number }) => row.id)).toEqual(rows.slice(1, 3).map(row => row.observation));
    expect(result.sessions.map((row: { id: number }) => row.id)).toEqual(rows.slice(1, 3).map(row => row.summary));
    expect(result.prompts.map((row: { id: number }) => row.id)).toEqual(rows.slice(1, 3).map(row => row.prompt));
  });

  it('finds both path separator styles before direct-child filtering', () => {
    const windows = seed('windows', now, 'repo', 'src\\folder\\file.ts');
    const posix = seed('posix', now + 1, 'repo', 'src/folder/other.ts');
    const nested = seed('nested', now + 2, 'repo', 'src/folder/deep/no.ts');
    for (const path of ['/repo/src/folder', 'C:\\repo\\src\\folder', 'src/folder', 'src\\folder']) {
      const result = search.findByFile(path, { project: 'repo', isFolder: true, limit: 10 });
      expect(result.observations.map(row => row.id).sort()).toEqual([windows.observation, posix.observation].sort());
      expect(result.sessions.map(row => row.id).sort()).toEqual([windows.summary, posix.summary].sort());
      expect(result.observations.some(row => row.id === nested.observation)).toBe(false);
    }
  });

  for (const category of ['observations', 'sessions', 'prompts'] as const) {
    it(`uses the same timestamp tie order for ${category} on each backend`, async () => {
      const rows = [seed('a', now), seed('b', now), seed('c', now)];
      const key = category === 'observations' ? 'observation' : category === 'sessions' ? 'summary' : 'prompt';
      for (const chroma of [null, fakeChroma(vectors(rows))]) {
        const result = await manager(chroma).search({
          query: 'needle', project: 'repo', format: 'json', limit: 2, offset: 0, orderBy: 'date_desc',
        });
        expect(result[category].map((row: { id: number }) => row.id)).toEqual([rows[2][key], rows[1][key]]);
      }
    });
  }

  it('normalizes each stored separator before selecting folder candidates', () => {
    const match = seed('mixed', now, 'repo', 'src\\area/folder\\file.ts');
    seed('nested', now + 1, 'repo', 'src\\area/folder/deep/file.ts');
    const result = search.findByFile('src/area/folder', { project: 'repo', isFolder: true });
    expect(result.observations.map(row => row.id)).toEqual([match.observation]);
    expect(result.sessions.map(row => row.id)).toEqual([match.summary]);
  });

  it('continues semantic pages beyond the first hundred candidates', async () => {
    const rows = Array.from({ length: 120 }, (_, i) => seed(`deep-${i}`, now + i));
    const chroma = fakeChroma(rows.map(row => ({ id: row.observation, kind: 'observation' as const, epoch: row.epoch })));
    const args = { query: 'needle', project: 'repo', format: 'json', type: 'observations', orderBy: 'relevance', limit: 10, offset: 100 };
    const results = [
      await manager(chroma).search(args),
      (await new SearchOrchestrator(search, store, chroma as never).search(args)).results,
    ];
    for (const result of results) {
      expect(result.observations.map((row: { id: number }) => row.id)).toEqual(rows.slice(100, 110).map(row => row.observation));
    }
  });

  it('keeps the same backfill source on successive pages', async () => {
    const rows = Array.from({ length: 100 }, (_, i) => seed(`row-${i}`, now + i));
    const chroma = fakeChroma([
      ...rows.map(row => ({ id: row.prompt, kind: 'user_prompt' as const, epoch: row.epoch })),
      { id: rows[99].observation, kind: 'observation' as const, epoch: rows[99].epoch },
    ]);
    const args = { query: 'needle', project: 'repo', format: 'json', orderBy: 'relevance', limit: 20 };
    const services = [
      (offset: number, limit: number) => manager(chroma).search({ ...args, offset, limit }),
      async (offset: number, limit: number) => (await new SearchOrchestrator(search, store, chroma as never)
        .search({ ...args, offset, limit })).results,
    ];
    for (const run of services) {
      const first = await run(0, 20);
      const second = await run(20, 20);
      const whole = await run(0, 40);
      expect(whole.observations.length).toBeGreaterThan(0);
      expect([...first.observations, ...second.observations].map((row: { id: number }) => row.id))
        .toEqual(whole.observations.map((row: { id: number }) => row.id));
    }
  });

  it('accepts a projects-only filter for all categories in the orchestrator', async () => {
    const native = seed('native', now, 'repo');
    const adopted = seed('adopted', now + 1, 'repo/worktree', 'src/file.ts', 'repo');
    seed('other', now + 2, 'unrelated');
    const result = await new SearchOrchestrator(search, store, null).search({ projects: ['repo'], orderBy: 'date_desc', limit: 10 });
    expect(result.results.observations.map(row => row.id)).toEqual([adopted.observation, native.observation]);
    expect(result.results.sessions.map(row => row.id)).toEqual([adopted.summary, native.summary]);
    expect(result.results.prompts.map(row => row.id)).toEqual([adopted.prompt, native.prompt]);
  });

  it('searches merged project observations, summaries and prompts in both entry points', async () => {
    const native = seed('native', now, 'repo');
    const adopted = seed('adopted', now + 1, 'repo/worktree', 'src/file.ts', 'repo');
    seed('other', now + 2, 'unrelated');
    const expected = [adopted, native];
    const args = { query: 'needle', project: 'repo', orderBy: 'date_desc', limit: 10 };
    const direct = await new SearchOrchestrator(search, store, null).search(args);
    const managed = await manager(null).search({ ...args, format: 'json' });
    for (const result of [direct.results, managed]) {
      expect(result.observations.map((row: { id: number }) => row.id)).toEqual(expected.map(row => row.observation));
      expect(result.sessions.map((row: { id: number }) => row.id)).toEqual(expected.map(row => row.summary));
      expect(result.prompts.map((row: { id: number }) => row.id)).toEqual(expected.map(row => row.prompt));
    }
  });
});
