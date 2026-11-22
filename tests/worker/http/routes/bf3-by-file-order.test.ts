import { expect, it } from 'bun:test';
import { SearchRoutes } from '../../../../src/services/worker/http/routes/SearchRoutes.js';
import { SearchManager } from '../../../../src/services/worker/SearchManager.js';
import { SessionStore } from '../../../../src/services/sqlite/SessionStore.js';
import { SessionSearch } from '../../../../src/services/sqlite/SessionSearch.js';
import { TimelineService } from '../../../../src/services/worker/TimelineService.js';

it('renders an ascending by-file page in chronological order across days and categories', async () => {
  const store = new SessionStore(':memory:');
  try {
    const search = new SessionSearch(store.db);
    const base = Date.UTC(2025, 0, 1, 12);
    for (let index = 0; index < 3; index++) {
      const memory = `mem-${index}`;
      const sdk = store.createSDKSession(`content-${index}`, 'repo', 'prompt');
      store.ensureMemorySessionIdRegistered(sdk, memory);
      const epoch = base + index * 86400_000;
      store.storeObservation(memory, 'repo', {
        type: 'discovery', title: `obs-${index}`, subtitle: null, narrative: null,
        facts: [], concepts: [], files_read: ['src/file.ts'], files_modified: [],
      }, 1, 0, epoch);
      store.importSessionSummary({
        memory_session_id: memory, project: 'repo', request: `summary-${index}`,
        investigated: null, learned: null, completed: null, next_steps: null, notes: null,
        files_read: JSON.stringify(['src/file.ts']), files_edited: null, prompt_number: 1,
        discovery_tokens: 0, created_at: new Date(epoch + 1).toISOString(), created_at_epoch: epoch + 1,
      });
    }
    const formatter = {
      formatTableHeader: () => '',
      formatObservationIndex: (row: { title: string }) => `OBS ${row.title}`,
      formatSessionIndex: (row: { request: string }) => `SESSION ${row.request}`,
    };
    const manager = new SearchManager(search, store, null, formatter as never, new TimelineService());
    const handlers = new Map<string, (req: never, res: never) => void>();
    new SearchRoutes(manager).setupRoutes({
      use: () => {}, post: () => {},
      get: (path: string, handler: (req: never, res: never) => void) => { handlers.set(path, handler); },
    } as never);
    const body = await new Promise<{ content: Array<{ text: string }> }>(resolve => {
      const response = { json: resolve, status: () => response, headersSent: false, locals: {} };
      handlers.get('/api/search/by-file')!({ path: '/api/search/by-file', query: {
        filePath: 'src/file.ts', project: 'repo', orderBy: 'date_asc', limit: '2', offset: '1',
      }, body: {}, get: () => undefined } as never, response as never);
    });
    const entries = body.content[0].text.match(/(?:OBS obs|SESSION summary)-\d/g);
    expect(entries).toEqual(['OBS obs-1', 'SESSION summary-1', 'OBS obs-2', 'SESSION summary-2']);
  } finally {
    store.close();
  }
});
