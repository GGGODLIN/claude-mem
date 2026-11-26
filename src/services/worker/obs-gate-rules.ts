import type { ConversationMessage } from '../worker-types.js';
import type { ParsedObservation } from '../../sdk/parser.js';

export const OBS_GATE_PRIOR_TITLES_LIMIT = 15;
const OBS_GATE_PRIOR_TITLE_MAX_CHARS = 180;
const RULES_PRIOR_WINDOW = 25;
const SIMILARITY_FIELD_MAX_CHARS = 260;
const MATCHED_DETAILS_MAX_CHARS = 1024;

const DISCOVERY_TITLE_JACCARD = 0.62;
const DISCOVERY_TITLE_OVERLAP = 0.80;
const DISCOVERY_BODY_JACCARD = 0.28;
const CORROBORATED_TITLE_OVERLAP = 0.40;
const CORROBORATING_SUMMARY_JACCARD = 0.65;
const MATCHED_DETAILS_TITLE_OVERLAP = 0.60;
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
const NEGATIVE_OR_DEFECT_RE = /\b(not|no|none|never|without|cannot|can't|couldn't|didn't|doesn't|isn't|aren't|wasn't|weren't|did\s+not|does\s+not|do\s+not|will\s+not|could\s+not|missing|omit|omits|omitted|absent|lacks?|unavailable|unreachable|unconfigured|fail|fails|failed|failing|failure|failures|error|errors|err|exception|crash|crashed|panic|panicked|broken|bug|defect|mismatch|mismatched|conflict|conflicts|reject|rejected|refused|denied|blocked|stale|outdated|exceeds?|overflow|timeout|timed\s+out|loop|stall|stalls|hang|hung|exited|exit\s+status|exit\s+code|ex_config|warning|disappeared|reverses?|hardcodes?|incompatible|incompatibility|deprecated|unsupported|invalid|corrupt(?:ed|ion)?|leak(?:s|ed|ing|age)?|vulnerab(?:le|ility)|bypass(?:es|ed|ing)?|violat(?:e|es|ed|ion)|deadlock(?:ed|s)?|stuck|regressed|regression(?!\s+tests?\b)|flaky|intermittent|oom|out\s+of\s+memory|aborted?|unhealthy|unverified|silently\s+drop(?:s|ped|ping)?|false\s+(?:negative|positive)|syntax\s*error|type\s*error|runtime\s*error|segfault|sigkill|sigsegv|non-?zero|rate[- ]limit(?:ed)?|throttl(?:ed|ing)|symlink\s+escape|(?:path|directory)\s+traversal|dangling|misconfigur(?:ed|ation)|unhandled|uncaught|unresolved)\b/i;

// 僅在句尾、換行、標點或明確回執詞前才削掉「零失敗片語」，避免把「No error handling」或「沒有錯誤處理」誤削成無負面訊號。
const ZERO_FAILURE_PHRASE_RE = /\b(?:0|zero|no)\s+(?:(?:test|type|syntax)\s+)?(?:fail(?:s|ed|ures?)?|errors?|warnings?|broken)(?=[ \t]*(?:$|[\r\n.,;:!?)}\]]|\b(?:found|reported|detected|observed|remaining|passing|passed)\b))/gi;
const HAN_ZERO_FAILURE_RE = /(?:(?<![0-9０-９])[0０零]\s*(?:個|个|項|项|次|處|处)?\s*|(?:未發現|未发现|沒有|没有|無|无)(?:任何)?)(?:錯誤|错误|失敗|失败|異常|异常|警告)(?=[ \t]*(?:$|[\r\n，。；！？、,.;:!?)）\]】]|通[過过]|回[報报]|發[現现]|結束|结束))/g;
const SUMMARY_FAILURE_RE = /\b(?:fails?|failed|failing|failures?|broken|crash(?:ed)?|panic(?:ked)?|timed\s+out|aborted|unhealthy|unverified|unsupported|mismatch|warnings?|incompatib(?:le|ility)|dangling\s+(?:sym)?links?|blocked\s+by|security\s+alerts?|(?:credential|secret)s?\s+(?:leaked|exposed)|oom|out\s+of\s+memory|deadlock(?:ed|s)?|segfault|exit(?:ed)?\s+(?:with\s+)?(?:status|code)\s+(?!0\b)\d+)\b/i;
const HAN_DEFECT_CUES = [
  /失[敗败]|[錯错][誤误]|[異异]常|逾時|超[時时]|崩[潰溃]|崩跌|[斷断][鏈链]|[壞坏][鏈链]/u,
  /不相容|不兼容|[無无]法|不能|尚未|仍未|未(?:能|完成|解[決决]|(?:全[部數数])?通[過过])|[並并]非全[部數数]|不通[過过]|不支[援持]/u,
  /缺失|缺少|[遺遗]漏|漏[攔拦遮]|[誤误](?:[刪删]|判|用|[棄弃]答|[殺杀])|[衝冲]突|[過过][期時时]|拒[絕绝]|被阻[擋挡斷断]/u,
  /警[告訊讯示]|[資资]料外[洩泄]|安全(?:風險|风险|漏洞)|記憶體|内存|吃[掉光]|[佔占]用|膨[脹胀]|爆(?:掉|到|量|[滿满])|暴[漲涨衝冲]|危[機机]|瓶[頸颈]|耗[盡尽]/u,
  /(?:退出|結束|结束)(?:狀態|状态|代[碼码]|[碼码])\s*[:：]?\s*[1-9１-９]/u,
];
const PARTIAL_PASS_RE = /\bpass(?:ed|es)?\s+only\b|\bonly\s+[^.!?;]{1,40}\s+pass(?:ed|es)?\b|(?:[僅仅只]|部分)[^，。；！？,.;!?\n]{0,24}(?:通[過过]|[過过][關关])|(?:通[過过]|[過过][關关])(?:了)?\s*部分/i;
// 偵測中英雙語「過關／全綠／通過 … 但／卻／however／but」轉折，避免把藏在過關前綴後的危機當成純例行回執丟棄。
const CONTRAST_CAVEAT_RE = /(?:通[過过]|[過过][關关]|全[綠绿]|成功|\bpass(?:ed|es)?\b|\bsucceeded\b|\bok\b)[^。！？；.!?\n]{0,40}(?:但|卻|却|不[過过]|然而|可是|只是|反而|\bbut\b|\byet\b|\bhowever\b|\bexcept\b|\bwhile\b)/iu;

function hasNegativeCue(title: string, summary: string): boolean {
  const cleanedTitle = title.replace(ZERO_FAILURE_PHRASE_RE, ' ').replace(HAN_ZERO_FAILURE_RE, ' ');
  const cleanedSummary = summary.replace(ZERO_FAILURE_PHRASE_RE, ' ').replace(HAN_ZERO_FAILURE_RE, ' ');
  return NEGATIVE_OR_DEFECT_RE.test(cleanedTitle)
    || [cleanedTitle, cleanedSummary].some((text) => SUMMARY_FAILURE_RE.test(text)
      || PARTIAL_PASS_RE.test(text)
      || CONTRAST_CAVEAT_RE.test(text)
      || HAN_DEFECT_CUES.some((cue) => cue.test(text)));
}

const GIT_BOOKKEEPING_TITLE_RE = /^(?:committed|pushed|staged)\b|\b(?:are\s+uncommitted|uncommitted\s+.*changes|working\s+tree\s+contains\s+uncommitted)\b|^已(?:[將将]\s*)?[^。！？;!?\n]{0,80}(?:提交|推送)/i;
const GIT_CLEAN_STATUS_RE = /\b(?:working\s+tree|branch|repository)\b[^,;.!?]{0,48}\b(?:is\s+clean|clean\s+and\s+synced|up[- ]to[- ]date)\b|(?:工作[樹树]|分支)[^，。；！？]{0,24}(?:乾淨|干净|已同步)/i;
const GIT_RELEASE_TAG_RE = /--(?:follow-)?tags\b|refs\/tags\/|\bgit\s+tag\b|\brelease\s+tags?\b|\bpushed\s+(?:the\s+)?tag\b|\bgh\s+release\b|(?:版本|[發发][布佈行])\s*(?:[標标][籤签]|tag\b)/i;
const GIT_PUSH_VERSION_RE = /(?:^|&&)\s*git(?:\s+-C\s+[^\s&]+)?\s+push\s+[^&\n]*?(?:\s|:)(?:v\d+(?:\.\d+)*|\d+(?:\.\d+)+)(?:[-+][\w.-]+)?(?=\s|&|$)/i;

const ROUTINE_SYNTAX_CMD_RE = /^\s*(?:bash\s+-n|sh\s+-n|zsh\s+-n|node\s+--check|python3?\s+-m\s+py_compile|tsc\s+--noEmit)\b/i;
const ROUTINE_PASS_TITLE_RE = /\b(syntax\s+check\s+passed|test\s+suite\s+passes|tests?\s+pass(?:ed|es)?\b|passes\s+.*verification|now\s+validate[sd]?|parse[sd]?\s+across\s+all|is\s+healthy\b|remained\s+stable|completed\s+successfully|was\s+quit\s+before|started\s+.*(?:ablation|benchmark|run)\b)\b/i;

// 整句錨定中文過關回執形狀，不允許後方夾帶其他發現或轉折說明。
const HAN_PASS_TITLE_RE = /^(?:全部|全[數数]|所有)?\s*\d*\s*[個个項项支條条]?\s*(?:[單单]元|整合|自動化|自动化)?(?:[測测][試试]|[單单][測测]|[檢检]查|[驗验][證证]|[編编][譯译])(?:套件)?(?:在\s*[\d.]+\s*(?:秒|ms|s)\s*[內内])?\s*(?:全部|全[數数]|均|皆|[順顺]利)?(?:通[過过]|[過过][關关]|全[綠绿])[。.!！]?$/u;
const VERIFY_CMD_RE = /^(?:node\s+--test|pytest|claude\s+plugin\s+(?:validate|test)|python3?\s+-m\s+(?:unittest|pytest)|(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?(?:test|check|typecheck|lint)(?::[\w-]+)*)(?:\s+[\w./*:@=+%-]+)*$/i;
const SHELL_OPERATORS_RE = /[;&|`$<>\r\n]/;

const TEMP_ROOT_RE = /^\/(?:tmp|private\/tmp|(?:private\/)?var\/folders)\/[^/]/;
const TEMP_WORK_RE = /\b(?:temporary|scratch(?:pad)?|sandbox|ephemeral)\b|[暫暂]存|沙[盒箱]|[臨临][時时]/iu;
const DURABLE_WRITE_RE = /\b(?:production|tracked\s+(?:file|source)|canonical\s+(?:repo|repository))\b|正式(?:[倉仓]|[環环]境)|[主正]式[儲储]存[庫库]/i;

const PURE_LS_CMD_RE = /^\s*ls(?:\s+-[A-Za-z0-9]+)*\s+[^;&\n]+(?:\|\s*(?:grep|head|tail)\b[^;&\n]*)?\s*$/i;
const LOCATION_ONLY_TITLE_RE = /^(?:located\s+the\b|.*implementation\s+is\s+in\b)/i;

const EXPLORATION_TITLES = new Map([
  ['Glob', /\b(?:keeps|stores|has)\s+(?:its\s+)?(?:tests?|fixtures?)\s+(?:under|in)\s+`?[\w.~/\-]+`?\.?$/i],
  ['Read', /\b(?:entry\s+point|implementation|configuration\s+file)\s+is\s+located\s+(?:at|in|under)\s+`?[\w.~/\-]+`?\.?$/i],
  ['Grep', /^(?:found|located)\s+[^.!?;]{1,80}\s+(?:references?|matches|occurrences)\s+(?:in|across)\s+[^!?;]{1,100}$/i],
]);
const DIRECTORY_INVENTORY_RE = /^(?:listed|enumerated)\s+[^.!?;]{0,80}\b(?:directories|folders|files)\b[^!?;]{0,80}$/i;
const LINE_COUNT_RE = /\b(?:total(?:s|ed)?|contain(?:s|ed)?|has|have)\s+[\d,]+\s+(?:source\s+)?lines\.?$/i;
const PATH_CHECK_RE = /^(?:verified|confirmed)\s+[^.!?;]{1,100}\b(?:exist(?:s)?|(?:were\s+)?found|resolved)\s+(?:in|on)\s+(?:the\s+)?\$?PATH\.?$/i;
const ROW_COUNT_CMD_RE = /^sqlite3[ \t]+(?:-readonly[ \t]+)?[\w./~-]+[ \t]+(['"])select\s+count\(\s*\*\s*\)\s+from\s+[a-z_]\w*(?:\.[a-z_]\w*)?\s*;?\1$/i;
const ROW_COUNT_TITLE_RE = /\b(?:database|table)\b[^,;.!?]{0,48}\b(?:contains?|has|holds)\s+[\d,]+\s+rows?(?:\s+total)?\.?$/i;
const FINDING_DETAIL_RE = /\b(?:because|due\s+to|caus(?:e|es|ed|ing)|instead\s+of|despite|requires?|only|migrat(?:ed|ion)|renamed|regression|unexpected(?:ly)?|previously)\b|因為|由於|導致|导致|必[須须]|只(?:有|能)|改[為为]|[遷迁]移|[迴回][歸归]|原本|意外/i;

const HAN_RUN_RE = /\p{Script=Han}+/gu;
const HAN_CHAR_RE = /\p{Script=Han}/u;
const chineseWords = new Intl.Segmenter('zh', { granularity: 'word' });

export interface PriorObservationSummary {
  type: string;
  titleNorm: string;
  titleTokens: Set<string>;
  bodyTokens: Set<string>;
  summaryTokens?: ReadonlySet<string>;
  detailsNorm?: string;
  filesModified: Set<string>;
  hasNeg: boolean;
}

export interface ObsRuleDecision {
  verdict: 'KEEP' | 'DROP';
  reason: string;
}

export function tokenizeObsText(text: string): Set<string> {
  const normalized = (text || '').normalize('NFKC');
  const matches = normalized.toLowerCase().match(/[a-z0-9][a-z0-9._/-]*/g) ?? [];
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
  for (const { 0: run } of normalized.matchAll(HAN_RUN_RE)) {
    for (const { segment } of chineseWords.segment(run)) tokens.add(segment);
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

function hasSharedTitleTerms(a: ReadonlySet<string>, b: ReadonlySet<string>, hasHan: boolean): boolean {
  let shared = 0;
  for (const token of a) {
    if (b.has(token) && (!hasHan || (HAN_CHAR_RE.test(token) && token.length > 1))) {
      shared += 1;
      if (shared >= 2) return true;
    }
  }
  return false;
}

function normalizeFilePaths(files: readonly string[] | undefined): Set<string> {
  const paths = new Set<string>();
  for (const file of files ?? []) {
    const path = String(file ?? '').trim();
    if (path) paths.add(path);
  }
  return paths;
}

function equalPaths(a: ReadonlySet<string>, b: ReadonlySet<string>): boolean {
  if (a.size !== b.size) return false;
  for (const path of a) {
    if (!b.has(path)) return false;
  }
  return true;
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

function isRoutineGitBookkeeping(command: string, title: string, summary: string): boolean {
  if (FINDING_DETAIL_RE.test(title) || FINDING_DETAIL_RE.test(summary)) return false;
  if (/^git\s+status(?:\s+[-\w]+)*$/.test(command) && GIT_CLEAN_STATUS_RE.test(title)) return true;
  if (!GIT_BOOKKEEPING_TITLE_RE.test(title)) return false;

  const steps = command.split('&&').map((step) => step.trim());
  if (/^cd\s+[\w./~-]+$/.test(steps[0] ?? '')) steps.shift();
  return steps.length > 0 && steps.every((step) => /^git\s+(?:status|add|commit|push)\b/.test(step) && !SHELL_OPERATORS_RE.test(step));
}

function isRoutineExploration(toolName: string, command: string, title: string, summary: string): boolean {
  if (FINDING_DETAIL_RE.test(title) || FINDING_DETAIL_RE.test(summary)) return false;
  if (toolName !== 'Bash') return EXPLORATION_TITLES.get(toolName)?.test(title) ?? false;
  if (ROW_COUNT_TITLE_RE.test(title) && ROW_COUNT_CMD_RE.test(command)) return true;
  if (!/^[\w./~-]+(?:[ \t]+[\w./~*?+-]+)+$/.test(command)) return false;
  return (/^ls\b/.test(command) && DIRECTORY_INVENTORY_RE.test(title))
    || (/^wc\s+-l\b/.test(command) && LINE_COUNT_RE.test(title))
    || (/^(?:which|command\s+-v)\s/.test(command) && PATH_CHECK_RE.test(title));
}

function isVerificationStep(step: string): boolean {
  return !SHELL_OPERATORS_RE.test(step) && (VERIFY_CMD_RE.test(step) || ROUTINE_SYNTAX_CMD_RE.test(step));
}

function isTemporaryPath(path: string): boolean {
  return TEMP_ROOT_RE.test(path) && !path.split('/').some((segment) => segment === '.' || segment === '..');
}

function getEditTarget(toolName: string, toolInput: unknown): string | undefined {
  if (typeof toolInput === 'string') {
    try {
      toolInput = JSON.parse(toolInput);
    } catch {
      return undefined;
    }
  }
  if (!toolInput || typeof toolInput !== 'object' || Array.isArray(toolInput)) return undefined;
  const input = toolInput as Record<string, unknown>;
  const path = input[toolName === 'NotebookEdit' ? 'notebook_path' : 'file_path'];
  return typeof path === 'string' ? path : undefined;
}

function isTemporaryShellChange(command: string): boolean {
  return command.split('&&').every((part) => {
    const step = part.trim();
    if (!/^[\w./-]+(?:[ \t]+[\w./-]+)+$/.test(step)) return false;
    const [name, ...args] = step.split(/[ \t]+/);
    if (name === 'cp') {
      const [source, destination] = args;
      return args.length === 2 && !source.startsWith('-') && isTemporaryPath(destination);
    }
    if (name === 'mkdir' && args[0] === '-p') args.shift();
    if (name === 'rm' && /^-(?:r|f|rf|fr)$/.test(args[0] ?? '')) args.shift();
    return (name === 'mkdir' || name === 'touch' || name === 'rm')
      && args.length > 0 && args.every(isTemporaryPath);
  });
}

function isTemporaryOnlyChange(
  toolName: string,
  toolInput: unknown,
  command: string,
  files: readonly string[],
  text: string,
): boolean {
  if (!files.every(isTemporaryPath) || !TEMP_WORK_RE.test(text) || DURABLE_WRITE_RE.test(text)) return false;
  if (toolName === 'Bash') return isTemporaryShellChange(command);
  const target = getEditTarget(toolName, toolInput);
  return target !== undefined && files.length > 0 && files.every((file) => file === target);
}

function stripXmlTags(text: string): string {
  return text.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
}

function tokenizeSummary(subtitle: string, narrative: string, facts: string): Set<string> {
  return tokenizeObsText([subtitle, narrative, facts].map((field) => field.slice(0, SIMILARITY_FIELD_MAX_CHARS)).join(' '));
}

function normalizeDetails(text: string): string | undefined {
  if (text.length > MATCHED_DETAILS_MAX_CHARS) return undefined;
  return text.replace(/\s+/g, ' ').trim();
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
      const filesModified = normalizeFilePaths(extractFileList(block, 'files_modified'));
      const titleTokens = tokenizeObsText(title.slice(0, SIMILARITY_FIELD_MAX_CHARS));
      const summaryTokens = tokenizeSummary(subtitle, narrative, facts);
      const summary = `${subtitle} ${narrative} ${facts}`;
      const summaryLines = [subtitle, narrative, facts].filter(Boolean).join('\n');
      if (limit > 0 && entries.length === limit) entries.shift();
      entries.push({
        type,
        titleNorm: title.toLowerCase(),
        titleTokens,
        bodyTokens: new Set([...titleTokens, ...summaryTokens]),
        summaryTokens,
        filesModified,
        hasNeg: hasNegativeCue(title, summaryLines),
        detailsNorm: normalizeDetails(summary),
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
  const titleTokens = tokenizeObsText(title.slice(0, SIMILARITY_FIELD_MAX_CHARS));
  const titleHasHan = HAN_CHAR_RE.test(title);
  const summaryTokens = tokenizeSummary(subtitle, narrative, factsStr);
  const bodyTokens = new Set([...titleTokens, ...summaryTokens]);
  const filesMod = normalizeFilePaths(obs.files_modified);
  const summary = `${subtitle} ${narrative} ${factsStr}`;
  const summaryLines = [subtitle, narrative, ...(obs.facts ?? [])].filter(Boolean).join('\n');
  const hasNeg = hasNegativeCue(title, summaryLines);
  const detailsNorm = normalizeDetails(summary);
  const isEditTool = toolName === 'Edit' || toolName === 'Write' || toolName === 'NotebookEdit';
  const isMutation = isEditTool || filesMod.size > 0
    || obsType === 'change' || obsType === 'bugfix' || obsType === 'feature' || obsType === 'refactor';
  const command = toolName === 'Bash' ? extractBashCommand(toolInput) : '';

  if (GIT_RELEASE_TAG_RE.test(command) || GIT_PUSH_VERSION_RE.test(command)
    || GIT_RELEASE_TAG_RE.test(`${title} ${summary}`)) {
    return { verdict: 'KEEP', reason: 'release_tag' };
  }

  if (obsType === 'security_alert' || obsType === 'decision') {
    for (const prev of priorWindow) {
      if (titleNorm && prev.type === obsType && prev.titleNorm === titleNorm
        && detailsNorm !== undefined && detailsNorm === prev.detailsNorm && equalPaths(filesMod, prev.filesModified)) {
        return { verdict: 'DROP', reason: 'exact_title_duplicate' };
      }
    }
    return { verdict: 'KEEP', reason: 'protected_type' };
  }

  if (obsType === 'change' && !hasNeg && (isEditTool || toolName === 'Bash')
    && isTemporaryOnlyChange(toolName, toolInput, command, obs.files_modified ?? [], `${title} ${summary}`)) {
    return { verdict: 'DROP', reason: 'temporary_only_change' };
  }

  if (toolName === 'Bash' && !hasNeg && isRoutineGitBookkeeping(command, title, summary)) {
    return { verdict: 'DROP', reason: 'routine_git_bookkeeping' };
  }

  // Compare only matching outcomes, with stricter detail and path evidence for mutations.
  for (let i = priorWindow.length - 1; i >= 0; i -= 1) {
    const prev = priorWindow[i]!;
    if (titleTokens.size === 0) break;
    if (hasNeg !== prev.hasNeg) continue;
    if ((isMutation || prev.filesModified.size > 0) && (!filesMod.size || !equalPaths(filesMod, prev.filesModified))) continue;
    const sameDetails = detailsNorm !== undefined && detailsNorm === prev.detailsNorm;
    if ((hasNeg || isMutation) && !sameDetails) continue;
    if (prev.titleNorm === titleNorm) {
      return { verdict: 'DROP', reason: 'exact_title_duplicate' };
    }
    if (prev.type !== obsType) continue;

    const hasHanTitle = titleHasHan || HAN_CHAR_RE.test(prev.titleNorm);
    if (hasHanTitle && !hasSharedTitleTerms(titleTokens, prev.titleTokens, true)) continue;
    const jTitle = jaccardSimilarity(titleTokens, prev.titleTokens);
    const ovTitle = overlapCoefficient(titleTokens, prev.titleTokens);
    const jBody = jaccardSimilarity(bodyTokens, prev.bodyTokens);

    const corroboratedBySameDetails = sameDetails && summaryTokens.size >= 4 && ovTitle >= MATCHED_DETAILS_TITLE_OVERLAP;
    if (isMutation) {
      if ((jTitle >= MUTATION_TITLE_JACCARD && jBody >= 0.45) || corroboratedBySameDetails) {
        return { verdict: 'DROP', reason: 'dup_mutation' };
      }
      continue;
    }

    if (hasNeg) {
      if ((jTitle >= NEGATIVE_TITLE_JACCARD && jBody >= 0.35)
        || (obsType === 'discovery' && corroboratedBySameDetails)) {
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
    // A paraphrase can share fewer title words when the details independently agree.
    if (obsType === 'discovery' && ovTitle >= CORROBORATED_TITLE_OVERLAP
      && (hasHanTitle || hasSharedTitleTerms(titleTokens, prev.titleTokens, false))
      && prev.summaryTokens && summaryTokens.size >= 4 && prev.summaryTokens.size >= 4
      && jaccardSimilarity(summaryTokens, prev.summaryTokens) >= CORROBORATING_SUMMARY_JACCARD) {
      return { verdict: 'DROP', reason: 'corroborated_dup_discovery' };
    }
  }

  if (obsType === 'discovery' && !hasNeg && filesMod.size === 0
    && isRoutineExploration(toolName, command, title, summary)) {
    return { verdict: 'DROP', reason: 'routine_exploration' };
  }

  // Routine Bash verification and directory-location echoes.
  if (toolName === 'Bash' && !hasNeg && filesMod.size === 0) {
    const verificationOnly = command.split('&&').every((part) => isVerificationStep(part.trim()));
    if (verificationOnly && ROUTINE_SYNTAX_CMD_RE.test(command)) {
      return { verdict: 'DROP', reason: 'routine_syntax_check_pass' };
    }
    if (verificationOnly && obsType === 'discovery' && HAN_PASS_TITLE_RE.test(title)) {
      return { verdict: 'DROP', reason: 'routine_verification_pass' };
    }
    if ((verificationOnly || (command && !SHELL_OPERATORS_RE.test(command))) && ROUTINE_PASS_TITLE_RE.test(title)) {
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
    if (obsType === 'discovery' && PURE_LS_CMD_RE.test(command) && LOCATION_ONLY_TITLE_RE.test(title)) {
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
