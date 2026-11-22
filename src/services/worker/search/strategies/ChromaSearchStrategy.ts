
import {
  StrategySearchOptions,
  SearchSelection,
  SearchCategory,
  StrategySearchResult,
  SEARCH_CONSTANTS,
  isCategoryRequested,
  buildCategoryWhereFilter,
  ChromaMetadata,
  DateRange,
  ObservationSearchResult,
  SessionSummarySearchResult,
  UserPromptSearchResult
} from '../types.js';
import { ChromaSync } from '../../../sync/ChromaSync.js';
import { SessionStore } from '../../../sqlite/SessionStore.js';
import { logger } from '../../../../utils/logger.js';
import { normalizePlatformSource } from '../../../../shared/platform-source.js';
import { resolveDateBound } from '../../../../shared/date-bounds.js';
import { buildProjectWhereFilter, projectReadKeysFor } from '../project-where-filter.js';

function safePositiveLimit(value: unknown, fallback: number): number {
  if (typeof value === 'boolean') return fallback;
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : fallback;
}

function safeNonNegativeOffset(value: unknown): number {
  if (typeof value === 'boolean') return 0;
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : 0;
}

export class ChromaSearchStrategy {
  constructor(
    private chromaSync: ChromaSync,
    private sessionStore: SessionStore
  ) {}

  private emptyResult(strategy: 'chroma'): StrategySearchResult {
    return {
      results: { observations: [], sessions: [], prompts: [] },
      usedChroma: true,
      strategy
    };
  }

  async search(options: StrategySearchOptions): Promise<StrategySearchResult> {
    const {
      query,
      searchType = 'all',
      obsType,
      concepts,
      files,
      limit = SEARCH_CONSTANTS.DEFAULT_LIMIT,
      offset = 0,
      project,
      projects,
      platformSource,
      dateRange,
      orderBy = 'date_desc',
      ignoreDefaultRecencyWindow = false
    } = options;

    if (!query) {
      return this.emptyResult('chroma');
    }

    const safeLimit = safePositiveLimit(limit, SEARCH_CONSTANTS.DEFAULT_LIMIT);
    const safeOffset = safeNonNegativeOffset(offset);

    const searchObservations = isCategoryRequested(searchType, 'observations');
    const searchSessions = isCategoryRequested(searchType, 'sessions');
    const searchPrompts = isCategoryRequested(searchType, 'prompts');

    // The keys SearchManager scopes a search by, so this path agrees with it:
    // the requested projects, their stored spellings, and one merge hop.
    const readKeys = projectReadKeysFor(this.sessionStore, project, projects);
    const whereFilter = this.buildWhereFilter(searchType, readKeys, platformSource);

    logger.debug('SEARCH', 'ChromaSearchStrategy: Querying Chroma', { query, searchType });

    return await this.executeChromaSearch(query, whereFilter, {
      searchObservations, searchSessions, searchPrompts,
      obsType, concepts, files, orderBy, limit: safeLimit, offset: safeOffset, project,
      projects: readKeys.length > 0 ? readKeys : undefined,
      platformSource, dateRange, ignoreDefaultRecencyWindow
    });
  }

  private async executeChromaSearch(
    query: string,
    whereFilter: Record<string, any> | undefined,
    options: {
      searchObservations: boolean;
      searchSessions: boolean;
      searchPrompts: boolean;
      obsType?: string | string[];
      concepts?: string | string[];
      files?: string | string[];
      orderBy: 'relevance' | 'date_desc' | 'date_asc';
      limit: number;
      offset: number;
      project?: string;
      projects?: string[];
      platformSource?: string;
      dateRange?: DateRange;
      ignoreDefaultRecencyWindow: boolean;
    }
  ): Promise<StrategySearchResult> {
    const batchSize = Math.max(SEARCH_CONSTANTS.CHROMA_BATCH_SIZE, options.offset + options.limit);
    const chromaResults = await this.chromaSync.queryChroma(
      query,
      batchSize,
      whereFilter
    );

    if (chromaResults.ids.length === 0) {
      return {
        results: { observations: [], sessions: [], prompts: [] },
        usedChroma: true,
        strategy: 'chroma'
      };
    }

    const recentItems = this.filterByRecency(chromaResults, options.dateRange, options.ignoreDefaultRecencyWindow);
    const categorized = this.categorizeByDocType(recentItems, options);

    let observations: ObservationSearchResult[] = [];
    let sessions: SessionSummarySearchResult[] = [];
    let prompts: UserPromptSearchResult[] = [];
    const hasHitsBeforeOffset: Partial<Record<SearchCategory, boolean>> = {};

    const sqlOrderBy = options.orderBy;

    if (categorized.obsIds.length > 0) {
      const obsOptions = {
        type: options.obsType,
        concepts: options.concepts,
        files: options.files,
        orderBy: sqlOrderBy,
        limit: options.limit,
        offset: options.offset,
        project: options.project,
        projects: options.projects,
        platformSource: options.platformSource
      };
      observations = this.sessionStore.getObservationsByIds(categorized.obsIds, obsOptions);
      if (observations.length === 0 && options.offset > 0) {
        const unpaged = this.sessionStore.getObservationsByIds(categorized.obsIds, {
          ...obsOptions,
          limit: 1,
          offset: 0
        });
        if (unpaged.length > 0) {
          hasHitsBeforeOffset.observations = true;
        }
      }
    }

    if (categorized.sessionIds.length > 0) {
      const sessionOptions = {
        orderBy: sqlOrderBy,
        limit: options.limit,
        offset: options.offset,
        project: options.project,
        projects: options.projects,
        platformSource: options.platformSource
      };
      sessions = this.sessionStore.getSessionSummariesByIds(categorized.sessionIds, sessionOptions);
      if (sessions.length === 0 && options.offset > 0) {
        const unpaged = this.sessionStore.getSessionSummariesByIds(categorized.sessionIds, {
          ...sessionOptions,
          limit: 1,
          offset: 0
        });
        if (unpaged.length > 0) {
          hasHitsBeforeOffset.sessions = true;
        }
      }
    }

    if (categorized.promptIds.length > 0) {
      const promptOptions = {
        orderBy: sqlOrderBy,
        limit: options.limit,
        offset: options.offset,
        project: options.project,
        projects: options.projects,
        platformSource: options.platformSource
      };
      prompts = this.sessionStore.getUserPromptsByIds(categorized.promptIds, promptOptions);
      if (prompts.length === 0 && options.offset > 0) {
        const unpaged = this.sessionStore.getUserPromptsByIds(categorized.promptIds, {
          ...promptOptions,
          limit: 1,
          offset: 0
        });
        if (unpaged.length > 0) {
          hasHitsBeforeOffset.prompts = true;
        }
      }
    }

    return {
      results: { observations, sessions, prompts },
      usedChroma: true,
      strategy: 'chroma',
      ...(Object.keys(hasHitsBeforeOffset).length > 0 ? { hasHitsBeforeOffset } : {}),
    };
  }

  private buildWhereFilter(searchType: SearchSelection, readKeys: string[], platformSource?: string): Record<string, any> | undefined {
    const filters: Array<Record<string, any>> = [];

    const categoryFilter = buildCategoryWhereFilter(searchType);
    if (categoryFilter) filters.push(categoryFilter);

    if (readKeys.length > 0) {
      filters.push(buildProjectWhereFilter(readKeys));
    }

    if (platformSource) {
      filters.push({ platform_source: normalizePlatformSource(platformSource) });
    }

    if (filters.length === 0) {
      return undefined;
    }
    if (filters.length === 1) {
      return filters[0];
    }
    return { $and: filters };
  }

  private filterByRecency(chromaResults: {
    ids: number[];
    metadatas: ChromaMetadata[];
  }, dateRange: DateRange | undefined, ignoreDefaultRecencyWindow: boolean): Array<{ id: number; meta: ChromaMetadata }> {
    let startEpoch: number | undefined;
    let endEpoch: number | undefined;

    if (dateRange) {
      if (dateRange.start) {
        startEpoch = resolveDateBound(dateRange.start, 'start');
      }
      if (dateRange.end) {
        endEpoch = resolveDateBound(dateRange.end, 'end');
      }
    } else if (!ignoreDefaultRecencyWindow) {
      startEpoch = Date.now() - SEARCH_CONSTANTS.RECENCY_WINDOW_MS;
    }

    // ChromaSync deduplicates by (document type, SQLite id) and returns
    // aligned arrays. IDs are table-local: an observation and a prompt can
    // both be id 1, so keying metadata by the numeric id loses one category.
    return chromaResults.ids
      .map((id, index) => ({ id, meta: chromaResults.metadatas[index] }))
      .filter(item => item.meta && item.meta.created_at_epoch != null
        && (!startEpoch || item.meta.created_at_epoch >= startEpoch)
        && (!endEpoch || item.meta.created_at_epoch <= endEpoch));
  }

  private categorizeByDocType(
    items: Array<{ id: number; meta: ChromaMetadata }>,
    options: {
      searchObservations: boolean;
      searchSessions: boolean;
      searchPrompts: boolean;
    }
  ): { obsIds: number[]; sessionIds: number[]; promptIds: number[] } {
    const obsIds: number[] = [];
    const sessionIds: number[] = [];
    const promptIds: number[] = [];

    for (const item of items) {
      const docType = item.meta?.doc_type;
      if (docType === 'observation' && options.searchObservations) {
        obsIds.push(item.id);
      } else if (docType === 'session_summary' && options.searchSessions) {
        sessionIds.push(item.id);
      } else if (docType === 'user_prompt' && options.searchPrompts) {
        promptIds.push(item.id);
      }
    }

    return { obsIds, sessionIds, promptIds };
  }
}
