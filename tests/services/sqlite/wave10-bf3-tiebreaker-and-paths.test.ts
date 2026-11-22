import { afterEach, beforeEach, describe, expect, it, mock } from 'bun:test';
import { SessionStore } from '../../../src/services/sqlite/SessionStore.js';
import { SessionSearch } from '../../../src/services/sqlite/SessionSearch.js';
import { scopedProjects } from '../../../src/services/sqlite/project-read-keys.js';
import { ChromaSearchStrategy } from '../../../src/services/worker/search/strategies/ChromaSearchStrategy.js';
import { SearchOrchestrator } from '../../../src/services/worker/search/SearchOrchestrator.js';
import { SearchManager } from '../../../src/services/worker/SearchManager.js';
import { TimelineService } from '../../../src/services/worker/TimelineService.js';

describe('Wave10 BF3: tie-breakers, *ByIds offset, project scoping, path candidates, json_each safety', () => {
  let store: SessionStore;
  let search: SessionSearch;

  beforeEach(() => {
    store = new SessionStore(':memory:');
    search = new SessionSearch(store.db);
  });

  afterEach(() => {
    store.close();
  });

  function registerSession(contentId: string, memoryId: string, project = 'tie-proj'): number {
    const sdkId = store.createSDKSession(contentId, project, 'seed prompt');
    store.ensureMemorySessionIdRegistered(sdkId, memoryId);
    return sdkId;
  }

  describe('1. Same-epoch id tie-breakers', () => {
    it('breaks timestamp ties by id in SessionSearch observations (filter-only, FTS, substring)', () => {
      registerSession('c-obs', 'm-obs');
      const epoch = 1_700_000_000_000;
      const ids: number[] = [];
      for (const label of ['alpha', 'beta', 'gamma']) {
        ids.push(
          store.storeObservation('m-obs', 'tie-proj', {
            type: 'discovery',
            title: `common keyword ${label} 队列`,
            subtitle: null,
            facts: [],
            narrative: `same narrative ${label}`,
            concepts: [],
            files_read: ['src/tie.ts'],
            files_modified: [],
          }, 1, 0, epoch).id,
        );
      }

      expect(search.searchObservations(undefined, { project: 'tie-proj', orderBy: 'date_asc' }).map(r => r.id)).toEqual(ids);
      expect(search.searchObservations(undefined, { project: 'tie-proj', orderBy: 'date_desc' }).map(r => r.id)).toEqual([...ids].reverse());

      expect(search.searchObservations('common', { project: 'tie-proj', orderBy: 'date_asc' }).map(r => r.id)).toEqual(ids);
      expect(search.searchObservations('common', { project: 'tie-proj', orderBy: 'date_desc' }).map(r => r.id)).toEqual([...ids].reverse());
      // FTS relevance with identical text/BM25 score falls back to created_at_epoch DESC, id DESC
      expect(search.searchObservations('common', { project: 'tie-proj', orderBy: 'relevance' }).map(r => r.id)).toEqual([...ids].reverse());

      // CJK substring branch
      expect(search.searchObservations('队列', { project: 'tie-proj', orderBy: 'date_asc' }).map(r => r.id)).toEqual(ids);
      expect(search.searchObservations('队列', { project: 'tie-proj', orderBy: 'date_desc' }).map(r => r.id)).toEqual([...ids].reverse());
    });

    it('breaks timestamp ties by id in SessionSearch sessions (filter-only, FTS, substring) and findByFile', () => {
      const epoch = 1_700_000_100_000;
      const obsIds: number[] = [];
      const sumIds: number[] = [];
      for (const i of [1, 2, 3]) {
        const memId = `m-sum-${i}`;
        registerSession(`c-sum-${i}`, memId);
        obsIds.push(
          store.storeObservation(memId, 'tie-proj', {
            type: 'discovery',
            title: `file obs ${i}`,
            subtitle: null,
            facts: [],
            narrative: 'file obs narrative',
            concepts: [],
            files_read: ['src/shared.ts'],
            files_modified: [],
          }, 1, 0, epoch).id,
        );
        sumIds.push(
          store.importSessionSummary({
            memory_session_id: memId,
            project: 'tie-proj',
            request: `shared session token 队列 ${i}`,
            investigated: 'shared',
            learned: null,
            completed: null,
            next_steps: null,
            files_read: JSON.stringify(['src/shared.ts']),
            files_edited: null,
            notes: null,
            prompt_number: i,
            discovery_tokens: 0,
            created_at: new Date(epoch).toISOString(),
            created_at_epoch: epoch,
          }).id,
        );
      }

      expect(search.searchSessions(undefined, { project: 'tie-proj', orderBy: 'date_asc' }).map(r => r.id)).toEqual(sumIds);
      expect(search.searchSessions(undefined, { project: 'tie-proj', orderBy: 'date_desc' }).map(r => r.id)).toEqual([...sumIds].reverse());

      expect(search.searchSessions('shared', { project: 'tie-proj', orderBy: 'date_asc' }).map(r => r.id)).toEqual(sumIds);
      expect(search.searchSessions('shared', { project: 'tie-proj', orderBy: 'date_desc' }).map(r => r.id)).toEqual([...sumIds].reverse());
      expect(search.searchSessions('shared', { project: 'tie-proj', orderBy: 'relevance' }).map(r => r.id)).toEqual([...sumIds].reverse());

      expect(search.searchSessions('队列', { project: 'tie-proj', orderBy: 'date_asc' }).map(r => r.id)).toEqual(sumIds);
      expect(search.searchSessions('队列', { project: 'tie-proj', orderBy: 'date_desc' }).map(r => r.id)).toEqual([...sumIds].reverse());

      const ascFile = search.findByFile('src/shared.ts', { project: 'tie-proj', orderBy: 'date_asc' });
      expect(ascFile.observations.map(r => r.id)).toEqual(obsIds);
      expect(ascFile.sessions.map(r => r.id)).toEqual(sumIds);

      const descFile = search.findByFile('src/shared.ts', { project: 'tie-proj', orderBy: 'date_desc' });
      expect(descFile.observations.map(r => r.id)).toEqual([...obsIds].reverse());
      expect(descFile.sessions.map(r => r.id)).toEqual([...sumIds].reverse());
    });

    it('breaks timestamp ties by id in SessionSearch.searchUserPrompts and SessionStore.*ByIds', () => {
      const sdkId = registerSession('c-prompts', 'm-prompts');
      const epoch = 1_700_000_200_000;
      const promptIds: number[] = [];
      for (const i of [1, 2, 3]) {
        const row = store.db.prepare(`
          INSERT INTO user_prompts (session_db_id, content_session_id, prompt_number, prompt_text, created_at, created_at_epoch)
          VALUES (?, ?, ?, ?, ?, ?) RETURNING id
        `).get(sdkId, 'c-prompts', i, `tie prompt ${i}`, new Date(epoch).toISOString(), epoch) as { id: number };
        promptIds.push(row.id);
      }

      expect(search.searchUserPrompts(undefined, { project: 'tie-proj', orderBy: 'date_asc' }).map(r => r.id)).toEqual(promptIds);
      expect(search.searchUserPrompts(undefined, { project: 'tie-proj', orderBy: 'date_desc' }).map(r => r.id)).toEqual([...promptIds].reverse());
      expect(search.searchUserPrompts('tie prompt', { project: 'tie-proj', orderBy: 'date_asc' }).map(r => r.id)).toEqual(promptIds);
      expect(search.searchUserPrompts('tie prompt', { project: 'tie-proj', orderBy: 'date_desc' }).map(r => r.id)).toEqual([...promptIds].reverse());

      expect(store.getUserPromptsByIds(promptIds, { orderBy: 'date_asc' }).map(r => r.id)).toEqual(promptIds);
      expect(store.getUserPromptsByIds(promptIds, { orderBy: 'date_desc' }).map(r => r.id)).toEqual([...promptIds].reverse());

      const sumIds: number[] = [];
      for (const i of [1, 2, 3]) {
        const memId = `m-sum-byid-${i}`;
        registerSession(`c-sum-byid-${i}`, memId);
        sumIds.push(
          store.importSessionSummary({
            memory_session_id: memId,
            project: 'tie-proj',
            request: `summary ${i}`,
            investigated: null,
            learned: null,
            completed: null,
            next_steps: null,
            files_read: null,
            files_edited: null,
            notes: null,
            prompt_number: i,
            discovery_tokens: 0,
            created_at: new Date(epoch).toISOString(),
            created_at_epoch: epoch,
          }).id,
        );
      }
      expect(store.getSessionSummariesByIds(sumIds, { orderBy: 'date_asc' }).map(r => r.id)).toEqual(sumIds);
      expect(store.getSessionSummariesByIds(sumIds, { orderBy: 'date_desc' }).map(r => r.id)).toEqual([...sumIds].reverse());
    });
  });

  describe('2. Cross-backend offset/limit normalization & public project merging', () => {
    it('supports numeric, string, and invalid offset/limit across SessionSearch, ChromaSearchStrategy, SearchOrchestrator, and SearchManager', async () => {
      const sdkId = registerSession('c-offset', 'm-offset', 'off-proj');
      const baseTs = Date.now() - 100_000;
      const obsIds: number[] = [];
      const sumIds: number[] = [];
      const promptIds: number[] = [];

      // Seed 52 observations/summaries and 22 prompts so default limits (50 and 20)
      // distinguish fallback-to-default from SQLite LIMIT -1 dumping the entire table.
      for (let i = 0; i < 52; i++) {
        const ts = baseTs + i * 1000;
        const suffix = String(i).padStart(2, '0');
        obsIds.push(
          store.storeObservation('m-offset', 'off-proj', {
            type: 'discovery',
            title: `offobs 队列 ${suffix}`,
            subtitle: null,
            facts: [],
            narrative: `offobs narrative ${suffix}`,
            concepts: ['offconcept'],
            files_read: ['src/off.ts'],
            files_modified: [],
          }, i + 1, 0, ts).id,
        );
        const sMem = `m-sum-off-${i}`;
        registerSession(`c-sum-off-${i}`, sMem, 'off-proj');
        sumIds.push(
          store.importSessionSummary({
            memory_session_id: sMem,
            project: 'off-proj',
            request: `offsum 队列 ${suffix}`,
            investigated: null,
            learned: null,
            completed: null,
            next_steps: null,
            files_read: JSON.stringify(['src/off.ts']),
            files_edited: null,
            notes: null,
            prompt_number: i + 1,
            discovery_tokens: 0,
            created_at: new Date(ts).toISOString(),
            created_at_epoch: ts,
          }).id,
        );
        if (i < 22) {
          const row = store.db.prepare(`
            INSERT INTO user_prompts (session_db_id, content_session_id, prompt_number, prompt_text, created_at, created_at_epoch)
            VALUES (?, ?, ?, ?, ?, ?) RETURNING id
          `).get(sdkId, 'c-offset', i + 1, `offprompt ${suffix}`, new Date(ts).toISOString(), ts) as { id: number };
          promptIds.push(row.id);
        }
      }

      // SessionSearch non-isFolder methods: valid pagination + string numeric + invalid limit/offset fallback
      expect(search.searchObservations('offobs', { project: 'off-proj', orderBy: 'date_asc', limit: 2, offset: 1 }).map(r => r.id)).toEqual(obsIds.slice(1, 3));
      expect(search.searchObservations('offobs', { project: 'off-proj', orderBy: 'date_asc', limit: '2' as any, offset: '3' as any }).map(r => r.id)).toEqual(obsIds.slice(3, 5));
      expect(search.searchObservations('队列', { project: 'off-proj', orderBy: 'date_asc', limit: '2' as any, offset: '3' as any }).map(r => r.id)).toEqual(obsIds.slice(3, 5));
      expect(search.searchSessions('offsum', { project: 'off-proj', orderBy: 'date_asc', limit: '2' as any, offset: '3' as any }).map(r => r.id)).toEqual(sumIds.slice(3, 5));
      expect(search.searchSessions('队列', { project: 'off-proj', orderBy: 'date_asc', limit: '2' as any, offset: '3' as any }).map(r => r.id)).toEqual(sumIds.slice(3, 5));
      expect(search.findByConcept('offconcept', { project: 'off-proj', orderBy: 'date_asc', limit: '2' as any, offset: '3' as any }).map(r => r.id)).toEqual(obsIds.slice(3, 5));
      expect(search.findByFile('src/off.ts', { project: 'off-proj', orderBy: 'date_asc', limit: '2' as any, offset: '3' as any }).observations.map(r => r.id)).toEqual(obsIds.slice(3, 5));
      expect(search.findByType('discovery', { project: 'off-proj', orderBy: 'date_asc', limit: '2' as any, offset: '3' as any }).map(r => r.id)).toEqual(obsIds.slice(3, 5));
      expect(search.searchUserPrompts('offprompt', { project: 'off-proj', orderBy: 'date_asc', limit: '2' as any, offset: '3' as any }).map(r => r.id)).toEqual(promptIds.slice(3, 5));

      for (const bad of [-1, 1.5, '2 rows', '1e20', NaN, true]) {
        // Invalid limit falls back to method default (50 for obs/sessions/concept/file/type, 20 for prompts), not LIMIT -1 dump (52/22) or SQLiteError
        expect(search.searchObservations(undefined, { project: 'off-proj', orderBy: 'date_asc', limit: bad as any, offset: bad as any }).map(r => r.id)).toEqual(obsIds.slice(0, 50));
        expect(search.searchObservations('offobs', { project: 'off-proj', orderBy: 'date_asc', limit: bad as any, offset: bad as any }).map(r => r.id)).toEqual(obsIds.slice(0, 50));
        expect(search.searchObservations('队列', { project: 'off-proj', orderBy: 'date_asc', limit: bad as any, offset: bad as any }).map(r => r.id)).toEqual(obsIds.slice(0, 50));
        expect(search.searchSessions(undefined, { project: 'off-proj', orderBy: 'date_asc', limit: bad as any, offset: bad as any }).map(r => r.id)).toEqual(sumIds.slice(0, 50));
        expect(search.searchSessions('offsum', { project: 'off-proj', orderBy: 'date_asc', limit: bad as any, offset: bad as any }).map(r => r.id)).toEqual(sumIds.slice(0, 50));
        expect(search.searchSessions('队列', { project: 'off-proj', orderBy: 'date_asc', limit: bad as any, offset: bad as any }).map(r => r.id)).toEqual(sumIds.slice(0, 50));
        expect(search.findByConcept('offconcept', { project: 'off-proj', orderBy: 'date_asc', limit: bad as any, offset: bad as any }).map(r => r.id)).toEqual(obsIds.slice(0, 50));
        expect(search.findByFile('src/off.ts', { project: 'off-proj', orderBy: 'date_asc', limit: bad as any, offset: bad as any }).observations.map(r => r.id)).toEqual(obsIds.slice(0, 50));
        expect(search.findByType('discovery', { project: 'off-proj', orderBy: 'date_asc', limit: bad as any, offset: bad as any }).map(r => r.id)).toEqual(obsIds.slice(0, 50));
        expect(search.searchUserPrompts(undefined, { project: 'off-proj', orderBy: 'date_asc', limit: bad as any, offset: bad as any }).map(r => r.id)).toEqual(promptIds.slice(0, 20));
        expect(search.searchUserPrompts('offprompt', { project: 'off-proj', orderBy: 'date_asc', limit: bad as any, offset: bad as any }).map(r => r.id)).toEqual(promptIds.slice(0, 20));
      }

      // ChromaSearchStrategy, SearchOrchestrator, and SearchManager across valid, string, and invalid offset/limit
      const first4Obs = obsIds.slice(0, 4);
      const relevanceObs = [first4Obs[2], first4Obs[0], first4Obs[3], first4Obs[1]];
      const chromaMock = {
        queryChroma: mock(async () => ({
          ids: relevanceObs,
          distances: relevanceObs.map((_, idx) => idx * 0.1),
          metadatas: relevanceObs.map((id, idx) => ({
            sqlite_id: id,
            doc_type: 'observation' as const,
            created_at_epoch: baseTs + idx * 1000,
          })),
        })),
      };

      const strategy = new ChromaSearchStrategy(chromaMock as any, store);
      expect((await strategy.search({ query: 'offobs', searchType: 'observations', project: 'off-proj', orderBy: 'relevance', limit: 2, offset: 1 })).results.observations.map(r => r.id)).toEqual(relevanceObs.slice(1, 3));
      expect((await strategy.search({ query: 'offobs', searchType: 'observations', project: 'off-proj', orderBy: 'date_asc', limit: '2' as any, offset: '1' as any })).results.observations.map(r => r.id)).toEqual(first4Obs.slice(1, 3));
      expect((await strategy.search({ query: 'offobs', searchType: 'observations', project: 'off-proj', orderBy: 'relevance', limit: -1 as any, offset: '2 rows' as any })).results.observations.map(r => r.id)).toEqual(relevanceObs);

      const orchestratorSqlite = new SearchOrchestrator(search, store, null);
      expect((await orchestratorSqlite.search({ query: 'offobs', searchType: 'observations', project: 'off-proj', orderBy: 'date_asc', limit: '2' as any, offset: '3' as any })).results.observations.map(r => r.id)).toEqual(obsIds.slice(3, 5));
      const orchBad = await orchestratorSqlite.search({ query: 'offobs', searchType: 'observations', project: 'off-proj', orderBy: 'date_asc', limit: -1 as any, offset: 1.5 as any });
      expect([20, 50]).toContain(orchBad.results.observations.length);
      expect(orchBad.results.observations.map(r => r.id)).toEqual(obsIds.slice(0, orchBad.results.observations.length));

      const orchestratorChroma = new SearchOrchestrator(search, store, chromaMock as any);
      expect((await orchestratorChroma.search({ query: 'offobs', searchType: 'observations', project: 'off-proj', orderBy: 'relevance', limit: 2, offset: 1 })).results.observations.map(r => r.id)).toEqual(relevanceObs.slice(1, 3));
      expect((await orchestratorChroma.search({ query: 'offobs', searchType: 'observations', project: 'off-proj', orderBy: 'relevance', limit: 'bad' as any, offset: -1 as any })).results.observations.map(r => r.id)).toEqual(relevanceObs);

      const managerSqlite = new SearchManager(search, store, null, {} as any, new TimelineService());
      expect((await managerSqlite.search({ query: 'offobs', type: 'observations', project: 'off-proj', orderBy: 'date_asc', limit: '2', offset: '3', format: 'json' })).observations.map((r: any) => r.id)).toEqual(obsIds.slice(3, 5));
      const mgrBad = await managerSqlite.search({ query: 'offobs', type: 'observations', project: 'off-proj', orderBy: 'date_asc', limit: -1, offset: 'bad', format: 'json' });
      expect([20, 50]).toContain(mgrBad.observations.length);
      expect(mgrBad.observations.map((r: any) => r.id)).toEqual(obsIds.slice(0, mgrBad.observations.length));

      const managerChroma = new SearchManager(search, store, chromaMock as any, {} as any, new TimelineService());
      expect((await managerChroma.search({ query: 'offobs', type: 'observations', project: 'off-proj', orderBy: 'relevance', limit: 2, offset: 1, format: 'json' })).observations.map((r: any) => r.id)).toEqual(relevanceObs.slice(1, 3));
      expect((await managerChroma.search({ query: 'offobs', type: 'observations', project: 'off-proj', orderBy: 'relevance', limit: 'bad', offset: 1.5, format: 'json' })).observations.map((r: any) => r.id)).toEqual(relevanceObs);
    });

    it('merges both project and projects in scopedProjects and public search entrypoints', async () => {
      expect(scopedProjects({ project: 'alpha', projects: ['beta', 'alpha'] }).sort()).toEqual(['alpha', 'beta']);

      registerSession('c-a', 'm-a', 'alpha');
      registerSession('c-b', 'm-b', 'beta');
      store.storeObservation('m-a', 'alpha', {
        type: 'discovery', title: 'from-alpha', subtitle: null, facts: [], narrative: null, concepts: [], files_read: [], files_modified: [],
      }, 1);
      store.storeObservation('m-b', 'beta', {
        type: 'discovery', title: 'from-beta', subtitle: null, facts: [], narrative: null, concepts: [], files_read: [], files_modified: [],
      }, 1);

      const orchestrator = new SearchOrchestrator(search, store, null);
      const orchResult = await orchestrator.search({ project: 'alpha', projects: ['beta'], searchType: 'observations' });
      expect(orchResult.results.observations.map(r => r.title).sort()).toEqual(['from-alpha', 'from-beta']);

      const manager = new SearchManager(search, store, null, {} as any, new TimelineService());
      const mgrResult = await manager.search({ project: 'alpha', projects: ['beta'], type: 'observations', format: 'json' });
      expect(mgrResult.observations.map((r: any) => r.title).sort()).toEqual(['from-alpha', 'from-beta']);
    });
  });

  describe('4. Path separator candidates, json_each safety, and isDirectChild', () => {
    it('matches both forward and backslash paths across findByFile, filters.files, and getObservationsByIds', () => {
      registerSession('c-win', 'm-win', 'path-proj');
      const winObsId = store.storeObservation('m-win', 'path-proj', {
        type: 'discovery',
        title: 'win-stored',
        subtitle: null,
        facts: [],
        narrative: null,
        concepts: [],
        files_read: ['C:\\repo\\src\\utils\\helper.ts'],
        files_modified: [],
      }, 1).id;

      registerSession('c-posix', 'm-posix', 'path-proj');
      const posixObsId = store.storeObservation('m-posix', 'path-proj', {
        type: 'discovery',
        title: 'posix-stored',
        subtitle: null,
        facts: [],
        narrative: null,
        concepts: [],
        files_read: ['/repo/src/utils/other.ts'],
        files_modified: [],
      }, 1).id;

      // Relative POSIX file query hits Windows backslash stored path (isFolder=false)
      expect(search.findByFile('src/utils/helper.ts', { project: 'path-proj', isFolder: false }).observations.map(r => r.id)).toEqual([winObsId]);
      // Relative Windows file query hits POSIX stored path (isFolder=false)
      expect(search.findByFile('src\\utils\\other.ts', { project: 'path-proj', isFolder: false }).observations.map(r => r.id)).toEqual([posixObsId]);

      // POSIX absolute folder query (isFolder=true) hits POSIX abs & Windows relative backslash stored paths
      registerSession('c-win-rel', 'm-win-rel', 'path-proj');
      const winRelId = store.storeObservation('m-win-rel', 'path-proj', {
        type: 'discovery',
        title: 'win-rel',
        subtitle: null,
        facts: [],
        narrative: null,
        concepts: [],
        files_read: ['src\\utils\\nested.ts'],
        files_modified: [],
      }, 1).id;

      registerSession('c-posix-rel', 'm-posix-rel', 'path-proj');
      const posixRelId = store.storeObservation('m-posix-rel', 'path-proj', {
        type: 'discovery',
        title: 'posix-rel',
        subtitle: null,
        facts: [],
        narrative: null,
        concepts: [],
        files_read: ['src/utils/rel.ts'],
        files_modified: [],
      }, 1).id;

      expect(search.findByFile('/repo/src/utils', { project: 'path-proj', isFolder: true }).observations.map(r => r.id).sort())
        .toEqual([posixObsId, winRelId, posixRelId].sort());
      expect(search.findByFile('C:\\repo\\src\\utils', { project: 'path-proj', isFolder: true }).observations.map(r => r.id).sort())
        .toEqual([winObsId, winRelId, posixRelId].sort());

      // filters.files in searchObservations and getObservationsByIds
      expect(search.searchObservations(undefined, { project: 'path-proj', files: ['src/utils/helper.ts'] }).map(r => r.id)).toEqual([winObsId]);
      expect(search.searchObservations(undefined, { project: 'path-proj', files: ['src\\utils\\other.ts'] }).map(r => r.id)).toEqual([posixObsId]);

      expect(store.getObservationsByIds([winObsId, posixObsId], { files: ['src/utils/helper.ts'] }).map(r => r.id)).toEqual([winObsId]);
      expect(store.getObservationsByIds([winObsId, posixObsId], { files: ['src\\utils\\other.ts'] }).map(r => r.id)).toEqual([posixObsId]);
    });

    it('guards json_each against malformed JSON in concepts, files_read, files_modified, files_edited', () => {
      registerSession('c-dirty', 'm-dirty', 'dirty-proj');
      const validObsId = store.storeObservation('m-dirty', 'dirty-proj', {
        type: 'discovery',
        title: 'valid obs',
        subtitle: null,
        facts: [],
        narrative: null,
        concepts: ['auth'],
        files_read: ['src/auth.ts'],
        files_modified: ['src/auth.ts'],
      }, 1).id;

      // Insert a row with malformed JSON directly into SQLite
      store.db.prepare(`
        INSERT INTO observations (memory_session_id, project, type, title, facts, concepts, files_read, files_modified, created_at, created_at_epoch)
        VALUES ('m-dirty', 'dirty-proj', 'discovery', 'corrupt obs', '[]', 'not-valid-json', '{"bad":1}', '123', ?, ?)
      `).run(new Date().toISOString(), Date.now());

      const validSumId = store.importSessionSummary({
        memory_session_id: 'm-dirty',
        project: 'dirty-proj',
        request: 'valid sum',
        investigated: null,
        learned: null,
        completed: null,
        next_steps: null,
        files_read: JSON.stringify(['src/auth.ts']),
        files_edited: JSON.stringify(['src/auth.ts']),
        notes: null,
        prompt_number: 1,
        discovery_tokens: 0,
        created_at: new Date().toISOString(),
        created_at_epoch: Date.now(),
      }).id;

      store.db.prepare(`
        INSERT INTO session_summaries (memory_session_id, project, request, files_read, files_edited, created_at, created_at_epoch)
        VALUES ('m-dirty', 'dirty-proj', 'corrupt sum', 'not-json', '{"x":true}', ?, ?)
      `).run(new Date().toISOString(), Date.now());

      expect(search.searchObservations(undefined, { project: 'dirty-proj', concepts: ['auth'] }).map(r => r.id)).toEqual([validObsId]);
      expect(search.searchObservations(undefined, { project: 'dirty-proj', files: ['src/auth.ts'] }).map(r => r.id)).toEqual([validObsId]);

      const byFile = search.findByFile('src/auth.ts', { project: 'dirty-proj' });
      expect(byFile.observations.map(r => r.id)).toEqual([validObsId]);
      expect(byFile.sessions.map(r => r.id)).toEqual([validSumId]);

      const allObsIds = (store.db.prepare('SELECT id FROM observations').all() as Array<{ id: number }>).map(r => r.id);
      expect(store.getObservationsByIds(allObsIds, { concepts: ['auth'] }).map(r => r.id)).toEqual([validObsId]);
      expect(store.getObservationsByIds(allObsIds, { files: ['src/auth.ts'] }).map(r => r.id)).toEqual([validObsId]);
    });

    it('matches direct folder children across POSIX/Windows separators while excluding nested and sibling-prefix paths via findByFile(isFolder=true)', () => {
      registerSession('c-folder', 'm-folder', 'folder-proj');
      const posixAbs = store.storeObservation('m-folder', 'folder-proj', {
        type: 'discovery',
        title: 'posix-abs',
        subtitle: null,
        facts: [],
        narrative: null,
        concepts: [],
        files_read: ['/repo/src/utils/a.ts'],
        files_modified: [],
      }, 1).id;
      const winAbs = store.storeObservation('m-folder', 'folder-proj', {
        type: 'discovery',
        title: 'win-abs',
        subtitle: null,
        facts: [],
        narrative: null,
        concepts: [],
        files_read: ['C:\\repo\\src\\utils\\b.ts'],
        files_modified: [],
      }, 1).id;
      const posixRel = store.storeObservation('m-folder', 'folder-proj', {
        type: 'discovery',
        title: 'posix-rel',
        subtitle: null,
        facts: [],
        narrative: null,
        concepts: [],
        files_read: ['src/utils/rel.ts'],
        files_modified: [],
      }, 1).id;
      const winRel = store.storeObservation('m-folder', 'folder-proj', {
        type: 'discovery',
        title: 'win-rel',
        subtitle: null,
        facts: [],
        narrative: null,
        concepts: [],
        files_read: ['src\\utils\\win-rel.ts'],
        files_modified: [],
      }, 1).id;
      // Nested descendants and sibling-prefix folders must be excluded
      store.storeObservation('m-folder', 'folder-proj', {
        type: 'discovery',
        title: 'nested-posix',
        subtitle: null,
        facts: [],
        narrative: null,
        concepts: [],
        files_read: ['/repo/src/utils/nested/c.ts'],
        files_modified: [],
      }, 1);
      store.storeObservation('m-folder', 'folder-proj', {
        type: 'discovery',
        title: 'nested-win',
        subtitle: null,
        facts: [],
        narrative: null,
        concepts: [],
        files_read: ['C:\\repo\\src\\utils\\nested\\d.ts'],
        files_modified: [],
      }, 1);
      store.storeObservation('m-folder', 'folder-proj', {
        type: 'discovery',
        title: 'sibling-prefix',
        subtitle: null,
        facts: [],
        narrative: null,
        concepts: [],
        files_read: ['/repo/src/utils-v2/e.ts'],
        files_modified: [],
      }, 1);

      expect(search.findByFile('src/utils', { project: 'folder-proj', isFolder: true }).observations.map(r => r.id).sort())
        .toEqual([posixAbs, winAbs, posixRel, winRel].sort());
      expect(search.findByFile('src\\utils', { project: 'folder-proj', isFolder: true }).observations.map(r => r.id).sort())
        .toEqual([posixAbs, winAbs, posixRel, winRel].sort());
      expect(search.findByFile('/repo/src/utils', { project: 'folder-proj', isFolder: true }).observations.map(r => r.id).sort())
        .toEqual([posixAbs, posixRel, winRel].sort());
      expect(search.findByFile('C:\\repo\\src\\utils', { project: 'folder-proj', isFolder: true }).observations.map(r => r.id).sort())
        .toEqual([winAbs, posixRel, winRel].sort());
    });
  });
});
