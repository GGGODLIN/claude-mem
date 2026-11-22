import { afterEach, beforeEach, describe, expect, it, mock } from 'bun:test';
import type { Request, Response } from 'express';
import { SessionStore } from '../../../src/services/sqlite/SessionStore.js';
import { SessionSearch } from '../../../src/services/sqlite/SessionSearch.js';
import { ChromaSearchStrategy } from '../../../src/services/worker/search/strategies/ChromaSearchStrategy.js';
import { SearchOrchestrator } from '../../../src/services/worker/search/SearchOrchestrator.js';
import { SearchManager } from '../../../src/services/worker/SearchManager.js';
import { SearchRoutes } from '../../../src/services/worker/http/routes/SearchRoutes.js';
import { TimelineService } from '../../../src/services/worker/TimelineService.js';

describe('Wave10 BF3: cross-backend offset, supplement consistency, and SearchRoutes', () => {
  let store: SessionStore;
  let search: SessionSearch;

  beforeEach(() => {
    store = new SessionStore(':memory:');
    search = new SessionSearch(store.db);
  });

  afterEach(() => {
    store.close();
  });

  function seedObs(memoryId: string, project: string, title: string, epoch: number, filesRead: string[] = ['src/app.ts']): number {
    const sdkId = store.createSDKSession(`content-${memoryId}-${title}`, project, 'prompt');
    store.ensureMemorySessionIdRegistered(sdkId, memoryId);
    return store.storeObservation(memoryId, project, {
      type: 'discovery',
      title,
      subtitle: null,
      facts: [],
      narrative: `${title} narrative`,
      concepts: [],
      files_read: filesRead,
      files_modified: [],
    }, 1, 0, epoch).id;
  }

  function seedSummary(memoryId: string, project: string, request: string, epoch: number, filesRead: string[] = ['src/app.ts']): number {
    const sdkId = store.createSDKSession(`content-sum-${memoryId}-${request}`, project, 'prompt');
    store.ensureMemorySessionIdRegistered(sdkId, memoryId);
    return store.importSessionSummary({
      memory_session_id: memoryId,
      project,
      request,
      investigated: null,
      learned: null,
      completed: null,
      next_steps: null,
      files_read: JSON.stringify(filesRead),
      files_edited: null,
      notes: null,
      prompt_number: 1,
      discovery_tokens: 0,
      created_at: new Date(epoch).toISOString(),
      created_at_epoch: epoch,
    }).id;
  }

  function seedPrompt(contentId: string, project: string, text: string, epoch: number): number {
    const sdkId = store.createSDKSession(contentId, project, text);
    const row = store.db.prepare(`
      INSERT INTO user_prompts (session_db_id, content_session_id, prompt_number, prompt_text, created_at, created_at_epoch)
      VALUES (?, ?, 1, ?, ?, ?) RETURNING id
    `).get(sdkId, contentId, text, new Date(epoch).toISOString(), epoch) as { id: number };
    return row.id;
  }

  describe('ChromaSearchStrategy offset & batch size', () => {
    it('expands Chroma batch size to cover offset + limit and pages hydrated results', async () => {
      const now = Date.now();
      const sdkId = store.createSDKSession('content-chroma-batch', 'p', 'prompt');
      store.ensureMemorySessionIdRegistered(sdkId, 'm-chroma-batch');
      const obsIds: number[] = [];
      for (let i = 0; i < 120; i++) {
        obsIds.push(
          store.storeObservation('m-chroma-batch', 'p', {
            type: 'discovery',
            title: `obs-${i}`,
            subtitle: null,
            facts: [],
            narrative: `obs-${i} narrative`,
            concepts: [],
            files_read: ['src/app.ts'],
            files_modified: [],
          }, 1, 0, now - (120 - i) * 1000).id,
        );
      }

      const queryChroma = mock(async (_query: string, batchSize: number) => {
        const sliced = obsIds.slice(0, batchSize);
        return {
          ids: sliced,
          distances: sliced.map((_, idx) => idx * 0.01),
          metadatas: sliced.map(id => ({ sqlite_id: id, doc_type: 'observation', created_at_epoch: now })),
        };
      });

      const strategy = new ChromaSearchStrategy({ queryChroma } as any, store);
      const page2 = await strategy.search({
        query: 'semantic',
        searchType: 'observations',
        project: 'p',
        orderBy: 'relevance',
        limit: 2,
        offset: 2,
      });

      expect(page2.results.observations.map(r => r.id)).toEqual(obsIds.slice(2, 4));
      const firstCall = queryChroma.mock.calls.at(-1)!;
      expect(firstCall[0]).toBe('semantic');
      expect(firstCall[1]).toBeGreaterThanOrEqual(4);

      const page95 = await strategy.search({
        query: 'semantic',
        searchType: 'observations',
        project: 'p',
        orderBy: 'relevance',
        limit: 20,
        offset: 95,
      });
      const lastCall = queryChroma.mock.calls.at(-1)!;
      expect(lastCall[0]).toBe('semantic');
      expect(lastCall[1]).toBeGreaterThanOrEqual(115);
      expect(page95.results.observations.map(r => r.id)).toEqual(obsIds.slice(95, 115));
    });
  });

  describe('SearchOrchestrator supplement & SQLite projectReadKeys consistency', () => {
    it('does not fall back to SQLite when offset exceeds non-empty Chroma hits', async () => {
      const now = Date.now();
      const chromaId1 = seedObs('m-c1', 'p', 'semantic alpha 1', now - 2000);
      const chromaId2 = seedObs('m-c2', 'p', 'semantic alpha 2', now - 1000);
      // Extra SQLite-only rows matching keyword "alpha" that Chroma did not return
      seedObs('m-s1', 'p', 'alpha sqlite 3', now - 500);
      seedObs('m-s2', 'p', 'alpha sqlite 4', now - 250);
      seedObs('m-s3', 'p', 'alpha sqlite 5', now - 100);

      const chroma = {
        queryChroma: mock(async () => ({
          ids: [chromaId1, chromaId2],
          distances: [0.1, 0.2],
          metadatas: [
            { sqlite_id: chromaId1, doc_type: 'observation', created_at_epoch: now - 2000 },
            { sqlite_id: chromaId2, doc_type: 'observation', created_at_epoch: now - 1000 },
          ],
        })),
      };

      const orchestrator = new SearchOrchestrator(search, store, chroma as any);
      const page2 = await orchestrator.search({
        query: 'alpha',
        searchType: 'observations',
        project: 'p',
        orderBy: 'relevance',
        limit: 2,
        offset: 2,
      });

      expect(page2.usedChroma).toBe(true);
      expect(page2.strategy).toBe('chroma');
      expect(page2.results.observations).toEqual([]);
    });

    it('supplements truly empty categories while leaving paged-past-end categories empty', async () => {
      const now = Date.now();
      const obs1 = seedObs('m-o1', 'p', '队列 obs', now - 4000);
      // Extra SQLite-only matching observations that Chroma did not return: if
      // paged-past-end protection breaks, SQLite offset=2 would return obs3/obs4.
      seedObs('m-o2', 'p', '队列 obs extra 2', now - 3000);
      seedObs('m-o3', 'p', '队列 obs extra 3', now - 2000);
      seedObs('m-o4', 'p', '队列 obs extra 4', now - 1000);
      seedSummary('m-su1', 'p', '队列 summary 1', now - 3000);
      seedSummary('m-su2', 'p', '队列 summary 2', now - 2000);
      const sum3 = seedSummary('m-su3', 'p', '队列 summary 3', now - 1000);
      const p1 = seedPrompt('c-p1', 'p', '队列 prompt 1', now - 3000);
      const p2 = seedPrompt('c-p2', 'p', '队列 prompt 2', now - 2000);
      const p3 = seedPrompt('c-p3', 'p', '队列 prompt 3', now - 1000);

      // Chroma has 1 obs and 3 prompts, but 0 summaries
      const chroma = {
        queryChroma: mock(async () => ({
          ids: [obs1, p1, p2, p3],
          distances: [0.1, 0.2, 0.3, 0.4],
          metadatas: [
            { sqlite_id: obs1, doc_type: 'observation', created_at_epoch: now - 4000 },
            { sqlite_id: p1, doc_type: 'user_prompt', created_at_epoch: now - 3000 },
            { sqlite_id: p2, doc_type: 'user_prompt', created_at_epoch: now - 2000 },
            { sqlite_id: p3, doc_type: 'user_prompt', created_at_epoch: now - 1000 },
          ],
        })),
      };

      const orchestrator = new SearchOrchestrator(search, store, chroma as any);
      const result = await orchestrator.search({
        query: '队列',
        searchType: 'all',
        project: 'p',
        orderBy: 'date_asc',
        limit: 2,
        offset: 2,
      });

      expect(result.results.observations).toEqual([]);
      expect(result.results.prompts.map(r => r.id)).toEqual([p3]);
      expect(result.results.sessions.map(r => r.id)).toEqual([sum3]);
      expect(result.strategy).toBe('hybrid');
    });

    it('expands projectReadKeysFor across SQLite paths and accepts projects-only searches', async () => {
      const now = Date.now();
      const mergedObs = seedObs('m-merged', 'repo/wt', 'adopted finding', now - 2000);
      store.db.prepare('UPDATE observations SET merged_into_project = ? WHERE id = ?').run('repo', mergedObs);
      const mergedSum = seedSummary('m-merged-sum', 'repo/wt', 'adopted summary', now - 1000);
      store.db.prepare('UPDATE session_summaries SET merged_into_project = ? WHERE id = ?').run('repo', mergedSum);
      const mergedPrompt = seedPrompt('content-sum-m-merged-sum-adopted summary', 'repo/wt', 'adopted prompt', now - 500);

      const orchestrator = new SearchOrchestrator(search, store, null);
      const result = await orchestrator.search({
        projects: ['repo'],
        searchType: 'all',
      });

      expect(result.results.observations.map(r => r.id)).toEqual([mergedObs]);
      expect(result.results.sessions.map(r => r.id)).toEqual([mergedSum]);
      expect(result.results.prompts.map(r => r.id)).toEqual([mergedPrompt]);
    });
  });

  describe('SearchManager cross-backend date pagination & tie-breaker', () => {
    it('paginates date_desc and date_asc without duplicates or gaps across Chroma + SQLite', async () => {
      const now = Date.now();
      // Interleaved timestamps: t4 (SQLite-only), t3 (Chroma-only), t2 (SQLite-only), t1 (Chroma-only)
      const t1 = now - 4000;
      const t2 = now - 3000;
      const t3 = now - 2000;
      const t4 = now - 1000;

      const c1 = seedObs('mc1', 'pag-proj', 'semantic concept one', t1);
      const s2 = seedObs('ms2', 'pag-proj', 'exactkey two', t2);
      const c3 = seedObs('mc3', 'pag-proj', 'semantic concept three', t3);
      const s4 = seedObs('ms4', 'pag-proj', 'exactkey four', t4);

      const chroma = {
        queryChroma: mock(async () => ({
          ids: [c1, c3],
          distances: [0.1, 0.2],
          metadatas: [
            { sqlite_id: c1, doc_type: 'observation', created_at_epoch: t1 },
            { sqlite_id: c3, doc_type: 'observation', created_at_epoch: t3 },
          ],
        })),
      };

      const manager = new SearchManager(search, store, chroma as any, {} as any, new TimelineService());

      const descP1 = await manager.search({
        query: 'exactkey',
        type: 'observations',
        project: 'pag-proj',
        orderBy: 'date_desc',
        limit: 2,
        offset: 0,
        format: 'json',
      });
      const descP2 = await manager.search({
        query: 'exactkey',
        type: 'observations',
        project: 'pag-proj',
        orderBy: 'date_desc',
        limit: 2,
        offset: 2,
        format: 'json',
      });

      expect(descP1.observations.map((r: any) => r.id)).toEqual([s4, c3]);
      expect(descP2.observations.map((r: any) => r.id)).toEqual([s2, c1]);

      const ascP1 = await manager.search({
        query: 'exactkey',
        type: 'observations',
        project: 'pag-proj',
        orderBy: 'date_asc',
        limit: 2,
        offset: 0,
        format: 'json',
      });
      const ascP2 = await manager.search({
        query: 'exactkey',
        type: 'observations',
        project: 'pag-proj',
        orderBy: 'date_asc',
        limit: 2,
        offset: 2,
        format: 'json',
      });

      expect(ascP1.observations.map((r: any) => r.id)).toEqual([c1, s2]);
      expect(ascP2.observations.map((r: any) => r.id)).toEqual([c3, s4]);
    });

    it('breaks same-epoch ties by id in mergeKeywordMatchesByDate and text output', async () => {
      const epoch = Date.now() - 1000;
      const id1 = seedObs('mt1', 'tie-m', 'semantic only one', epoch);
      const id2 = seedObs('mt2', 'tie-m', 'tiekey two', epoch);

      // Chroma returns id1 (not id2), SQLite FTS returns id2 (not id1) at identical epoch.
      const chroma = {
        queryChroma: mock(async () => ({
          ids: [id1],
          distances: [0.1],
          metadatas: [
            { sqlite_id: id1, doc_type: 'observation', created_at_epoch: epoch },
          ],
        })),
      };

      const formatter = {
        formatSearchTableHeader: () => '| ID |',
        formatObservationSearchRow: (obs: any, lastTime: string) => ({ row: `| #${obs.id} |`, time: lastTime }),
        formatSessionSearchRow: (s: any, lastTime: string) => ({ row: `| #S${s.id} |`, time: lastTime }),
        formatUserPromptSearchRow: (p: any, lastTime: string) => ({ row: `| #P${p.id} |`, time: lastTime }),
      };

      const manager = new SearchManager(search, store, chroma as any, formatter as any, new TimelineService());

      const descText = await manager.search({
        query: 'tiekey',
        type: 'observations',
        project: 'tie-m',
        orderBy: 'date_desc',
        limit: 2,
        format: 'text',
      });
      const descTextStr = descText.content[0].text;
      expect(descTextStr).toContain(`#${id2}`);
      expect(descTextStr).toContain(`#${id1}`);
      expect(descTextStr.indexOf(`#${id2}`)).toBeLessThan(descTextStr.indexOf(`#${id1}`));

      // Verify mergeKeywordMatchesByDate tie-break directly at limit: 1, format: 'json'
      const descLimit1 = await manager.search({
        query: 'tiekey',
        type: 'observations',
        project: 'tie-m',
        orderBy: 'date_desc',
        limit: 1,
        format: 'json',
      });
      expect(descLimit1.observations.map((r: any) => r.id)).toEqual([id2]);

      // Reverse: Chroma returns id4, SQLite FTS returns id3 at identical epoch -> date_asc puts id3 first
      const id3 = seedObs('mt3', 'tie-m2', 'tiekey asc one', epoch);
      const id4 = seedObs('mt4', 'tie-m2', 'semantic asc two', epoch);
      const chromaAsc = {
        queryChroma: mock(async () => ({
          ids: [id4],
          distances: [0.1],
          metadatas: [
            { sqlite_id: id4, doc_type: 'observation', created_at_epoch: epoch },
          ],
        })),
      };
      const managerAsc = new SearchManager(search, store, chromaAsc as any, formatter as any, new TimelineService());
      const ascText = await managerAsc.search({
        query: 'tiekey',
        type: 'observations',
        project: 'tie-m2',
        orderBy: 'date_asc',
        limit: 2,
        format: 'text',
      });
      const ascTextStr = ascText.content[0].text;
      expect(ascTextStr).toContain(`#${id3}`);
      expect(ascTextStr).toContain(`#${id4}`);
      expect(ascTextStr.indexOf(`#${id3}`)).toBeLessThan(ascTextStr.indexOf(`#${id4}`));

      const ascLimit1 = await managerAsc.search({
        query: 'tiekey',
        type: 'observations',
        project: 'tie-m2',
        orderBy: 'date_asc',
        limit: 1,
        format: 'json',
      });
      expect(ascLimit1.observations.map((r: any) => r.id)).toEqual([id3]);
    });

    it('does not fall back to SQLite via SearchManager.search when relevance offset exceeds Chroma hits', async () => {
      const now = Date.now();
      const c1 = seedObs('sm-c1', 'sm-off', 'semantic alpha 1', now - 2000);
      const c2 = seedObs('sm-c2', 'sm-off', 'semantic alpha 2', now - 1000);
      seedObs('sm-s1', 'sm-off', 'alpha sqlite 3', now - 500);
      seedObs('sm-s2', 'sm-off', 'alpha sqlite 4', now - 250);
      seedObs('sm-s3', 'sm-off', 'alpha sqlite 5', now - 100);

      const chroma = {
        queryChroma: mock(async () => ({
          ids: [c1, c2],
          distances: [0.1, 0.2],
          metadatas: [
            { sqlite_id: c1, doc_type: 'observation', created_at_epoch: now - 2000 },
            { sqlite_id: c2, doc_type: 'observation', created_at_epoch: now - 1000 },
          ],
        })),
      };

      const manager = new SearchManager(search, store, chroma as any, {} as any, new TimelineService());
      const telemetry: any = {};
      const page2 = await manager.search({
        query: 'alpha',
        type: 'observations',
        project: 'sm-off',
        orderBy: 'relevance',
        offset: 2,
        limit: 2,
        format: 'json',
      }, telemetry);

      expect(page2.observations).toEqual([]);
      expect(page2.totalResults).toBe(0);
      expect(telemetry.search_strategy).toBe('chroma');
    });

    it('caps invalid limits to safeLimit in performChromaSemanticSearch and text output', async () => {
      const now = Date.now();
      const obsIds: number[] = [];
      const sumIds: number[] = [];
      const promptIds: number[] = [];
      for (let i = 0; i < 22; i++) {
        const ts = now - (22 - i) * 1000;
        obsIds.push(seedObs(`lim-o-${i}`, 'lim-proj', `limitem ${i}`, ts));
        sumIds.push(seedSummary(`lim-s-${i}`, 'lim-proj', `limitem sum ${i}`, ts));
        promptIds.push(seedPrompt(`lim-p-${i}`, 'lim-proj', `limitem prompt ${i}`, ts));
      }

      const chroma = {
        queryChroma: mock(async () => ({
          ids: [...obsIds, ...sumIds, ...promptIds],
          distances: new Array(66).fill(0.1),
          metadatas: [
            ...obsIds.map(id => ({ sqlite_id: id, doc_type: 'observation', created_at_epoch: now })),
            ...sumIds.map(id => ({ sqlite_id: id, doc_type: 'session_summary', created_at_epoch: now })),
            ...promptIds.map(id => ({ sqlite_id: id, doc_type: 'user_prompt', created_at_epoch: now })),
          ],
        })),
      };

      const formatter = {
        formatSearchTableHeader: () => '| ID |',
        formatObservationSearchRow: (obs: any, lastTime: string) => ({ row: `| OBS #${obs.id} |`, time: lastTime }),
        formatSessionSearchRow: (s: any, lastTime: string) => ({ row: `| SUM #S${s.id} |`, time: lastTime }),
        formatUserPromptSearchRow: (p: any, lastTime: string) => ({ row: `| PRM #P${p.id} |`, time: lastTime }),
      };

      const manager = new SearchManager(search, store, chroma as any, formatter as any, new TimelineService());

      for (const badLimit of [-1, 'bad']) {
        const jsonRes = await manager.search({
          query: 'limitem',
          project: 'lim-proj',
          orderBy: 'relevance',
          limit: badLimit,
          format: 'json',
        });
        expect(jsonRes.observations).toHaveLength(20);
        expect(jsonRes.sessions).toHaveLength(20);
        expect(jsonRes.prompts).toHaveLength(20);

        const textRes = await manager.search({
          query: 'limitem',
          type: 'observations',
          project: 'lim-proj',
          orderBy: 'relevance',
          limit: badLimit,
          format: 'text',
        });
        const text = textRes.content[0].text as string;
        expect(text).toContain(`OBS #${obsIds[0]}`);
        expect(text).toContain(`OBS #${obsIds[19]}`);
      }
    });

    it('expands projectReadKeysFor only once so 2-hop merged projects do not leak during supplement', async () => {
      const now = Date.now();
      // Merge chain: proj-3hop -> proj-2hop -> proj-1hop -> proj-root
      const rootObs = seedObs('hop-root-o', 'proj-root', 'hopkey root obs', now - 4000);
      const hop1Sum = seedSummary('hop-1-s', 'proj-1hop', 'hopkey one', now - 3000);
      store.db.prepare('UPDATE session_summaries SET merged_into_project = ? WHERE id = ?').run('proj-root', hop1Sum);
      const hop2Sum = seedSummary('hop-2-s', 'proj-2hop', 'hopkey two', now - 2000);
      store.db.prepare('UPDATE session_summaries SET merged_into_project = ? WHERE id = ?').run('proj-1hop', hop2Sum);
      const hop3Sum = seedSummary('hop-3-s', 'proj-3hop', 'hopkey three', now - 1000);
      store.db.prepare('UPDATE session_summaries SET merged_into_project = ? WHERE id = ?').run('proj-2hop', hop3Sum);

      const rootPrompt = seedPrompt('content-hop-root-p', 'proj-root', 'hopkey root prompt', now - 3000);
      const hop1Prompt = seedPrompt('content-hop-1-p', 'proj-1hop', 'hopkey 1hop prompt', now - 2000);
      const hop2Prompt = seedPrompt('content-hop-2-p', 'proj-2hop', 'hopkey 2hop prompt', now - 1000);

      // Chroma returns only rootObs; sessions and prompts are supplemented via orchestrator.supplementEmptyCategories.
      // If SearchManager passes already-expanded projects to supplementEmptyCategories, projectReadKeysFor runs a 2nd hop
      // and leaks hop2Prompt (prompts use includeMerged: false) and hop3Sum (sessions use includeMerged: true).
      const chroma = {
        queryChroma: mock(async () => ({
          ids: [rootObs],
          distances: [0.1],
          metadatas: [{ sqlite_id: rootObs, doc_type: 'observation', created_at_epoch: now - 4000 }],
        })),
      };

      const manager = new SearchManager(search, store, chroma as any, {} as any, new TimelineService());
      const res = await manager.search({
        query: 'hopkey',
        project: 'proj-root',
        format: 'json',
      });

      expect(res.observations.map((r: any) => r.id)).toEqual([rootObs]);
      expect(res.prompts.map((r: any) => r.id).sort()).toEqual([rootPrompt, hop1Prompt].sort());
      expect(res.prompts.map((r: any) => r.id)).not.toContain(hop2Prompt);
      expect(res.sessions.map((r: any) => r.id).sort()).toEqual([hop1Sum, hop2Sum].sort());
      expect(res.sessions.map((r: any) => r.id)).not.toContain(hop3Sum);
    });
  });

  describe('SearchRoutes GET /api/search/by-file', () => {
    type Handler = (req: Request, res: Response) => void;

    function byFileHandler(): Handler {
      const formatter = {
        formatTableHeader: () => '| ID |',
        formatObservationIndex: (obs: { id: number; title: string }) => `OBS #${obs.id} ${obs.title}`,
        formatSessionIndex: (sess: { id: number; request: string }) => `SESSION #S${sess.id} ${sess.request}`,
      };
      const manager = new SearchManager(search, store, null, formatter as any, new TimelineService());
      const handlers = new Map<string, Handler>();
      new SearchRoutes(manager).setupRoutes({
        use: () => {},
        get: (path: string, handler: Handler) => { handlers.set(path, handler); },
        post: () => {},
      } as any);
      return handlers.get('/api/search/by-file')!;
    }

    async function callByFile(handler: Handler, query: Record<string, unknown>): Promise<{ status: number; body: any }> {
      let status = 200;
      let body: any;
      let done!: () => void;
      const promise = new Promise<void>(resolve => { done = resolve; });
      const res = {
        headersSent: false,
        locals: {},
        status(code: number) { status = code; return res; },
        json(payload: unknown) { body = payload; done(); return res; },
      };
      handler({ path: '/api/search/by-file', query, body: {}, get: () => undefined } as any, res as any);
      await promise;
      return { status, body };
    }

    it('renders date_asc ascending across days and breaks same-epoch ties by id', async () => {
      const day1 = Date.UTC(2025, 0, 1, 12, 0, 0);
      const day2 = Date.UTC(2025, 0, 2, 12, 0, 0);
      const obs1 = seedObs('r1', 'route-proj', 'DAY1_FIRST', day1, ['src/route.ts']);
      const obs2 = seedObs('r2', 'route-proj', 'DAY1_SECOND', day1, ['src/route.ts']);
      const obs3 = seedObs('r3', 'route-proj', 'DAY2_THIRD', day2, ['src/route.ts']);

      const handler = byFileHandler();
      const asc = await callByFile(handler, { filePath: 'src/route.ts', project: 'route-proj', orderBy: 'date_asc' });
      expect(asc.status).toBe(200);
      const ascText: string = asc.body.content[0].text;
      expect(ascText).toContain('Jan 1, 2025');
      expect(ascText).toContain('Jan 2, 2025');
      expect(ascText).toContain(`OBS #${obs1}`);
      expect(ascText).toContain(`OBS #${obs2}`);
      expect(ascText).toContain(`OBS #${obs3}`);
      expect(ascText.indexOf('Jan 1, 2025')).toBeLessThan(ascText.indexOf('Jan 2, 2025'));
      expect(ascText.indexOf(`OBS #${obs1}`)).toBeLessThan(ascText.indexOf(`OBS #${obs2}`));
      expect(ascText.indexOf(`OBS #${obs2}`)).toBeLessThan(ascText.indexOf(`OBS #${obs3}`));

      const desc = await callByFile(handler, { filePath: 'src/route.ts', project: 'route-proj', orderBy: 'date_desc' });
      expect(desc.status).toBe(200);
      const descText: string = desc.body.content[0].text;
      expect(descText).toContain('Jan 2, 2025');
      expect(descText).toContain('Jan 1, 2025');
      expect(descText).toContain(`OBS #${obs3}`);
      expect(descText).toContain(`OBS #${obs2}`);
      expect(descText).toContain(`OBS #${obs1}`);
      expect(descText.indexOf('Jan 2, 2025')).toBeLessThan(descText.indexOf('Jan 1, 2025'));
      expect(descText.indexOf(`OBS #${obs3}`)).toBeLessThan(descText.indexOf(`OBS #${obs2}`));
      expect(descText.indexOf(`OBS #${obs2}`)).toBeLessThan(descText.indexOf(`OBS #${obs1}`));
    });

    it('returns 400 when filePath is missing or whitespace-only', async () => {
      const handler = byFileHandler();
      for (const query of [{ project: 'route-proj' }, { filePath: '', project: 'route-proj' }, { filePath: '   ', project: 'route-proj' }]) {
        const res = await callByFile(handler, query);
        expect(res.status).toBe(400);
      }
    });

    it('treats isFolder="false" query string as boolean false rather than truthy folder mode', async () => {
      const nestedObs = seedObs('r-nested', 'route-proj', 'NESTED_FILE', Date.UTC(2025, 0, 3), ['src/sub/deep/leaf.ts']);
      const handler = byFileHandler();
      // With isFolder=false, substring search on 'src/sub' matches nested 'src/sub/deep/leaf.ts';
      // with isFolder=true (the bug when 'false' was truthy), isDirectChild rejects nested descendants.
      const res = await callByFile(handler, { filePath: 'src/sub', project: 'route-proj', isFolder: 'false' });
      expect(res.status).toBe(200);
      expect(res.body.content[0].text).toContain(`OBS #${nestedObs}`);
    });
  });
});
