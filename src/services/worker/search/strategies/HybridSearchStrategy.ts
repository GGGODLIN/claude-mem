
import {
  StrategySearchOptions,
  SEARCH_CONSTANTS,
  ObservationSearchResult,
  SessionSummarySearchResult
} from '../types.js';
import { ChromaSync } from '../../../sync/ChromaSync.js';
import { SessionStore } from '../../../sqlite/SessionStore.js';
import { SessionSearch } from '../../../sqlite/SessionSearch.js';
import { logger } from '../../../../utils/logger.js';
import { normalizePlatformSource } from '../../../../shared/platform-source.js';
import { buildProjectWhereFilter, projectReadKeysFor } from '../project-where-filter.js';

export class HybridSearchStrategy {
  constructor(
    private chromaSync: ChromaSync,
    private sessionStore: SessionStore,
    private sessionSearch: SessionSearch
  ) {}

  async findByFile(
    filePath: string,
    options: StrategySearchOptions
  ): Promise<{
    observations: ObservationSearchResult[];
    sessions: SessionSummarySearchResult[];
    usedChroma: boolean;
  }> {
    const { limit, offset, project, projects, platformSource, dateRange, orderBy, isFolder } = options;
    const readKeys = projectReadKeysFor(this.sessionStore, project, projects);
    const projectScope = readKeys.length > 0 ? { projects: readKeys } : {};
    const filters = { ...projectScope, platformSource, dateRange, orderBy, isFolder };
    const requestedLimit = Number(limit ?? SEARCH_CONSTANTS.DEFAULT_LIMIT);
    const requestedOffset = Number(offset ?? 0);
    if (isFolder && (!Number.isInteger(requestedLimit) || requestedLimit < 0 || !Number.isInteger(requestedOffset))) {
      // Preserve the native folder pagination validation for invalid requests.
      return { ...this.sessionSearch.findByFile(filePath, { ...filters, limit, offset }), usedChroma: false };
    }
    const pageLimit = isFolder || (typeof limit !== 'boolean' && Number.isSafeInteger(requestedLimit) && requestedLimit > 0)
      ? requestedLimit : SEARCH_CONSTANTS.DEFAULT_LIMIT;
    const pageOffset = typeof offset !== 'boolean' && Number.isSafeInteger(requestedOffset) ? Math.max(0, requestedOffset) : 0;
    if (pageLimit === 0) return { observations: [], sessions: [], usedChroma: false };
    const pageEnd = pageOffset + pageLimit;

    logger.debug('SEARCH', 'HybridSearchStrategy: findByFile', { filePath });

    const metadataResults = this.sessionSearch.findByFile(filePath, { ...filters, limit: pageEnd, offset: 0 });
    const sessions = metadataResults.sessions.slice(pageOffset, pageEnd);
    if (metadataResults.observations.length === 0) {
      return { observations: [], sessions, usedChroma: false };
    }

    const chromaResults = await this.chromaSync.queryChroma(
      filePath,
      SEARCH_CONSTANTS.CHROMA_BATCH_SIZE,
      this.buildObservationWhereFilter(readKeys, platformSource)
    );

    // An exact match outside the SQLite date prefix can rank on the first page.
    const matchingRanked = chromaResults.ids.length > 0
      ? this.sessionSearch.findByFile(filePath, { ...filters, limit: chromaResults.ids.length }, chromaResults.ids).observations
      : [];
    const rankedIds = this.intersectWithRanking(matchingRanked.map(row => row.id), chromaResults.ids);
    if (rankedIds.length === 0) {
      return { observations: metadataResults.observations.slice(pageOffset, pageEnd), sessions, usedChroma: false };
    }

    const dateOrder = orderBy === 'date_asc' || orderBy === 'date_desc' ? orderBy : undefined;
    const metadataIds = metadataResults.observations.map(row => row.id);
    const orderedIds = dateOrder ? metadataIds : [...rankedIds, ...metadataIds.filter(id => !rankedIds.includes(id))];
    const pageIds = orderedIds.slice(pageOffset, pageEnd);
    const observations = this.sessionStore.getObservationsByIds(pageIds, {
      orderBy: dateOrder ?? 'relevance',
      limit: pageLimit,
      ...projectScope,
      platformSource
    });
    if (!dateOrder) {
      observations.sort((a, b) => pageIds.indexOf(a.id) - pageIds.indexOf(b.id));
    }
    return { observations, sessions, usedChroma: true };
  }

  private buildObservationWhereFilter(readKeys: string[], platformSource?: string): Record<string, any> {
    const filters: Array<Record<string, any>> = [{ doc_type: 'observation' }];
    if (readKeys.length > 0) {
      filters.push(buildProjectWhereFilter(readKeys));
    }
    if (platformSource) {
      filters.push({ platform_source: normalizePlatformSource(platformSource) });
    }
    return filters.length === 1 ? filters[0] : { $and: filters };
  }

  private intersectWithRanking(metadataIds: number[], chromaIds: number[]): number[] {
    const metadataSet = new Set(metadataIds);
    const rankedIds: number[] = [];

    for (const chromaId of chromaIds) {
      if (metadataSet.has(chromaId) && !rankedIds.includes(chromaId)) {
        rankedIds.push(chromaId);
      }
    }

    return rankedIds;
  }
}
