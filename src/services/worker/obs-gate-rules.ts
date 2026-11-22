import type { ConversationMessage } from '../worker-types.js';
import type { ParsedObservation } from '../../sdk/parser.js';

export const OBS_GATE_PRIOR_TITLES_LIMIT = 15;
const OBS_GATE_PRIOR_TITLE_MAX_CHARS = 180;
const RULES_PRIOR_WINDOW = 25;

const DISCOVERY_TITLE_JACCARD = 0.62;
const DISCOVERY_TITLE_OVERLAP = 0.80;
const DISCOVERY_BODY_JACCARD = 0.28;
const NEGATIVE_TITLE_JACCARD = 0.78;
const MUTATION_TITLE_JACCARD = 0.82;

const STOP_WORDS = new Set([
  'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
  'in', 'on', 'at', 'to', 'for', 'of', 'by', 'from', 'with', 'into',
  'and', 'or', 'as', 'that', 'this', 'these', 'those', 'its', 'their',
  'has', 'have', 'had', 'after', 'before', 'under', 'over', 'now', 'all',
  'new', 'old', 'both', 'more', 'some', 'any', 'still', 'again', 'then',
]);

// Protects negative investigative findings, root causes, and defect clues from routine-echo drops.
const NEGATIVE_OR_DEFECT_RE = /\b(not|no|none|never|without|cannot|can't|couldn't|didn't|doesn't|isn't|aren't|wasn't|weren't|did\s+not|does\s+not|do\s+not|will\s+not|could\s+not|missing|omit|omits|omitted|absent|lacks?|unavailable|unreachable|unconfigured|fail|fails|failed|failing|failure|error|errors|err|exception|crash|crashed|panic|broken|bug|defect|mismatch|conflict|conflicts|reject|rejected|refused|denied|blocked|stale|outdated|exceeds?|overflow|timeout|timed\s+out|loop|stall|stalls|hang|hung|exited|exit\s+status|exit\s+code|ex_config|warning|disappeared|reverses?|hardcodes?)\b/i;

const GIT_BOOKKEEPING_CMD_RE = /^\s*(?:\(\s*)?(?:cd\s+[^&;|\n]+\s*&&\s*)?git\s+(?:status|add|commit|push)\b/i;
const GIT_BOOKKEEPING_TITLE_RE = /^(?:committed|pushed|staged)\b|\b(?:are\s+uncommitted|uncommitted\s+.*changes|working\s+tree\s+contains\s+uncommitted)\b/i;
const GIT_RELEASE_TAG_RE = /(?:--tags\b|\bgit\s+tag\b|\brelease\s+tag\b|\bgh\s+release\b)/i;

const ROUTINE_SYNTAX_CMD_RE = /^\s*(?:bash\s+-n|sh\s+-n|zsh\s+-n|node\s+--check|python3?\s+-m\s+py_compile|tsc\s+--noEmit)\b/i;
const ROUTINE_PASS_TITLE_RE = /\b(syntax\s+check\s+passed|test\s+suite\s+passes|tests?\s+pass(?:ed|es)?\b|passes\s+.*verification|now\s+validate[sd]?|parse[sd]?\s+across\s+all|is\s+healthy\b|remained\s+stable|completed\s+successfully|was\s+quit\s+before|started\s+.*(?:ablation|benchmark|run)\b)\b/i;

const PURE_LS_CMD_RE = /^\s*ls(?:\s+-[A-Za-z0-9]+)*\s+[^;&\n]+(?:\|\s*(?:grep|head|tail)\b[^;&\n]*)?\s*$/i;
const LOCATION_ONLY_TITLE_RE = /^(?:located\s+the\b|.*implementation\s+is\s+in\b)/i;

export interface PriorObservationSummary {
  type: string;
  titleNorm: string;
  titleTokens: Set<string>;
  bodyTokens: Set<string>;
  filesModified: Set<string>;
  hasNeg: boolean;
}

export interface ObsRuleDecision {
  verdict: 'KEEP' | 'DROP';
  reason: string;
}

export function tokenizeObsText(text: string): Set<string> {
  const matches = (text || '').toLowerCase().match(/[a-z0-9][a-z0-9._/-]*/g) ?? [];
  const tokens = new Set<string>();
  for (const raw of matches) {
    let w = raw.replace(/^[._/-]+|[._/-]+$/g, '');
    if (w.length <= 1 || STOP_WORDS.has(w)) continue;
    if (w.length > 4 && w.endsWith('ies')) {
      w = `${w.slice(0, -3)}y`;
    } else if (w.length > 4 && w.endsWith('ing')) {
      w = w.slice(0, -3);
    } else if (w.length > 4 && w.endsWith('ed')) {
      w = w.slice(0, -2);
    } else if (w.length > 3 && w.endsWith('s') && !w.endsWith('ss')) {
      w = w.slice(0, -1);
    }
    tokens.add(w);
  }
  return tokens;
}

function jaccardSimilarity(a: ReadonlySet<string>, b: ReadonlySet<string>): number {
  if (a.size === 0 || b.size === 0) return 0;
  let intersection = 0;
  for (const item of a) {
    if (b.has(item)) intersection += 1;
  }
  const union = a.size + b.size - intersection;
  return union > 0 ? intersection / union : 0;
}

function overlapCoefficient(a: ReadonlySet<string>, b: ReadonlySet<string>): number {
  if (a.size === 0 || b.size === 0) return 0;
  let intersection = 0;
  for (const item of a) {
    if (b.has(item)) intersection += 1;
  }
  const denom = Math.min(a.size, b.size);
  return denom > 0 ? intersection / denom : 0;
}

function normalizeFileBasenames(files: readonly string[] | undefined): Set<string> {
  const out = new Set<string>();
  for (const f of files ?? []) {
    const trimmed = String(f ?? '').trim().toLowerCase();
    if (!trimmed) continue;
    const slash = trimmed.lastIndexOf('/');
    out.add(slash >= 0 ? trimmed.slice(slash + 1) : trimmed);
  }
  return out;
}

function hasIntersection(a: ReadonlySet<string>, b: ReadonlySet<string>): boolean {
  for (const item of a) {
    if (b.has(item)) return true;
  }
  return false;
}

function extractBashCommand(toolInput: unknown): string {
  if (toolInput && typeof toolInput === 'object' && !Array.isArray(toolInput)) {
    const cmd = (toolInput as Record<string, unknown>).command;
    if (typeof cmd === 'string') return cmd.trim();
  }
  if (typeof toolInput === 'string') {
    const trimmed = toolInput.trim();
    if (trimmed.startsWith('{')) {
      try {
        const parsed = JSON.parse(trimmed) as Record<string, unknown>;
        if (typeof parsed.command === 'string') return parsed.command.trim();
      } catch {
        // Fall through to raw string
      }
    }
    return trimmed;
  }
  return '';
}

function stripXmlTags(text: string): string {
  return text.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
}

function extractTagText(block: string, tag: string): string {
  const match = new RegExp(`<${tag}\\b[^>]*>([\\s\\S]*?)<\\/${tag}>`, 'i').exec(block);
  return match ? stripXmlTags(match[1] ?? '') : '';
}

function extractFileList(block: string, wrapperTag: string): string[] {
  const wrapper = new RegExp(`<${wrapperTag}\\b[^>]*>([\\s\\S]*?)<\\/${wrapperTag}>`, 'i').exec(block);
  if (!wrapper) return [];
  const files: string[] = [];
  const fileRe = /<file\b[^>]*>([\s\S]*?)<\/file>/gi;
  let m: RegExpExecArray | null;
  while ((m = fileRe.exec(wrapper[1] ?? '')) !== null) {
    const val = stripXmlTags(m[1] ?? '');
    if (val) files.push(val);
  }
  return files;
}

/**
 * Extracts up to `limit` unique prior observation titles from in-memory session history
 * (zero extra SQLite queries). Used by opt-in `model` gate mode (Arm 2).
 */
export function extractPriorSessionTitles(
  history: readonly ConversationMessage[],
  limit = OBS_GATE_PRIOR_TITLES_LIMIT,
): string[] {
  const titles: string[] = [];
  const seen = new Set<string>();
  const titleRe = /<observation\b[^>]*>[\s\S]*?<title>([\s\S]*?)<\/title>/gi;
  for (const msg of history) {
    if (msg.role !== 'assistant' || !msg.content) continue;
    titleRe.lastIndex = 0;
    let match: RegExpExecArray | null;
    while ((match = titleRe.exec(msg.content)) !== null) {
      const title = stripXmlTags(match[1] ?? '');
      if (!title) continue;
      const key = title.toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      titles.push(title.slice(0, OBS_GATE_PRIOR_TITLE_MAX_CHARS));
    }
  }
  return titles.length <= limit ? titles : titles.slice(-limit);
}

/**
 * Extracts lightweight prior observation summaries from in-memory session history
 * (zero extra SQLite queries) for Arm 1 local deterministic deduplication.
 */
export function extractPriorSessionObservations(
  history: readonly ConversationMessage[],
  limit = RULES_PRIOR_WINDOW,
): PriorObservationSummary[] {
  const entries: PriorObservationSummary[] = [];
  const obsBlockRe = /<observation\b[^>]*>[\s\S]*?<\/observation>/gi;
  for (const msg of history) {
    if (msg.role !== 'assistant' || !msg.content) continue;
    obsBlockRe.lastIndex = 0;
    let match: RegExpExecArray | null;
    while ((match = obsBlockRe.exec(msg.content)) !== null) {
      const block = match[0];
      const title = extractTagText(block, 'title');
      if (!title) continue;
      const type = (extractTagText(block, 'type') || 'discovery').toLowerCase();
      const subtitle = extractTagText(block, 'subtitle');
      const narrative = extractTagText(block, 'narrative');
      const facts = extractTagText(block, 'facts');
      const filesModified = normalizeFileBasenames(extractFileList(block, 'files_modified'));
      entries.push({
        type,
        titleNorm: title.toLowerCase(),
        titleTokens: tokenizeObsText(title),
        bodyTokens: tokenizeObsText(`${title} ${subtitle} ${narrative.slice(0, 260)} ${facts.slice(0, 260)}`),
        filesModified,
        hasNeg: NEGATIVE_OR_DEFECT_RE.test(title),
      });
    }
  }
  return entries.length <= limit ? entries : entries.slice(-limit);
}

function evaluateSingleObservation(
  obs: ParsedObservation,
  toolName: string,
  toolInput: unknown,
  priorWindow: readonly PriorObservationSummary[],
): ObsRuleDecision {
  const obsType = (obs.type || 'discovery').toLowerCase().trim();
  const title = (obs.title ?? '').replace(/\s+/g, ' ').trim();
  const subtitle = obs.subtitle ?? '';
  const narrative = obs.narrative ?? '';
  const factsStr = (obs.facts ?? []).join(' ');
  const titleNorm = title.toLowerCase();
  const titleTokens = tokenizeObsText(title);
  const bodyTokens = tokenizeObsText(`${title} ${subtitle} ${narrative.slice(0, 260)} ${factsStr.slice(0, 260)}`);
  const filesMod = normalizeFileBasenames(obs.files_modified);
  const hasNeg = NEGATIVE_OR_DEFECT_RE.test(title);
  const isEditTool = toolName === 'Edit' || toolName === 'Write' || toolName === 'NotebookEdit';

  if (obsType === 'security_alert' || obsType === 'decision') {
    for (const prev of priorWindow) {
      if (titleNorm && prev.titleNorm === titleNorm) {
        return { verdict: 'DROP', reason: 'exact_title_duplicate' };
      }
    }
    return { verdict: 'KEEP', reason: 'protected_type' };
  }

  // Rule 1: Pure Bash VCS bookkeeping echoes (protects release tag pushes and error/conflict findings).
  if (toolName === 'Bash' && !hasNeg) {
    const cmd = extractBashCommand(toolInput);
    if (
      GIT_BOOKKEEPING_CMD_RE.test(cmd)
      && GIT_BOOKKEEPING_TITLE_RE.test(title)
      && !GIT_RELEASE_TAG_RE.test(cmd)
      && !GIT_RELEASE_TAG_RE.test(title)
    ) {
      return { verdict: 'DROP', reason: 'routine_git_bookkeeping' };
    }
  }

  // Rule 2: Same-session, same-type near-duplicate (strict guards on mutations and negative findings).
  for (let i = priorWindow.length - 1; i >= 0; i -= 1) {
    const prev = priorWindow[i]!;
    if (titleTokens.size === 0) break;
    if (prev.titleNorm === titleNorm) {
      return { verdict: 'DROP', reason: 'exact_title_duplicate' };
    }
    if (prev.type !== obsType) continue;

    const jTitle = jaccardSimilarity(titleTokens, prev.titleTokens);
    const ovTitle = overlapCoefficient(titleTokens, prev.titleTokens);
    const jBody = jaccardSimilarity(bodyTokens, prev.bodyTokens);

    if (isEditTool || obsType === 'change' || obsType === 'bugfix' || obsType === 'feature' || obsType === 'refactor') {
      const sharedMod = filesMod.size > 0 && prev.filesModified.size > 0
        ? hasIntersection(filesMod, prev.filesModified)
        : true;
      if (sharedMod && jTitle >= MUTATION_TITLE_JACCARD && jBody >= 0.45) {
        return { verdict: 'DROP', reason: 'dup_mutation' };
      }
      continue;
    }

    if (hasNeg && !prev.hasNeg) continue;
    if (hasNeg) {
      if (jTitle >= NEGATIVE_TITLE_JACCARD && jBody >= 0.35) {
        return { verdict: 'DROP', reason: 'dup_negative' };
      }
      continue;
    }

    if (jTitle >= DISCOVERY_TITLE_JACCARD && jBody >= DISCOVERY_BODY_JACCARD) {
      return { verdict: 'DROP', reason: 'near_dup_discovery' };
    }
    if (titleTokens.size >= 4 && ovTitle >= DISCOVERY_TITLE_OVERLAP && jBody >= 0.35) {
      return { verdict: 'DROP', reason: 'subset_dup_discovery' };
    }
  }

  // Rule 3: Routine Bash syntax-check / pass-echo / pure directory-location echoes (zero negative cues).
  if (toolName === 'Bash' && !hasNeg) {
    const cmd = extractBashCommand(toolInput);
    if (ROUTINE_SYNTAX_CMD_RE.test(cmd)) {
      return { verdict: 'DROP', reason: 'routine_syntax_check_pass' };
    }
    if (ROUTINE_PASS_TITLE_RE.test(title)) {
      if (
        obsType === 'discovery'
        || (obsType === 'change' && /\b(?:was\s+quit\s+before|started\s+.*(?:ablation|benchmark|run))\b/i.test(title))
      ) {
        return { verdict: 'DROP', reason: 'routine_verification_pass' };
      }
      if (obsType === 'bugfix' && /\bparse[sd]?\s+across\s+all\b/i.test(title)) {
        return { verdict: 'DROP', reason: 'routine_syntax_check_pass' };
      }
    }
    if (obsType === 'discovery' && PURE_LS_CMD_RE.test(cmd) && LOCATION_ONLY_TITLE_RE.test(title)) {
      return { verdict: 'DROP', reason: 'routine_dir_listing' };
    }
  }

  return { verdict: 'KEEP', reason: 'pass' };
}

/**
 * Evaluates a candidate observation turn using zero-LLM deterministic rules (Arm 1).
 * Drops the turn only when EVERY observation in the draft is classified as DROP.
 */
export function evaluateObservationRules(params: {
  toolName: string;
  toolInput: unknown;
  observations: readonly ParsedObservation[];
  priorObservations: readonly PriorObservationSummary[];
}): ObsRuleDecision {
  if (params.observations.length === 0) {
    return { verdict: 'KEEP', reason: 'empty' };
  }
  const dropReasons: string[] = [];
  for (const obs of params.observations) {
    const decision = evaluateSingleObservation(obs, params.toolName, params.toolInput, params.priorObservations);
    if (decision.verdict === 'KEEP') {
      return decision;
    }
    dropReasons.push(decision.reason);
  }
  return { verdict: 'DROP', reason: dropReasons[0] ?? 'rules_drop' };
}
