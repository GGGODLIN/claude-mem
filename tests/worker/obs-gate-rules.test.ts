import { describe, expect, it } from 'bun:test';
import type { ParsedObservation } from '../../src/sdk/parser.js';
import {
  evaluateObservationRules,
  extractPriorSessionObservations,
  extractPriorSessionTitles,
} from '../../src/services/worker/obs-gate-rules.js';

function makeObs(overrides: Partial<ParsedObservation> = {}): ParsedObservation {
  return {
    type: 'discovery',
    title: 'Sample observation title',
    subtitle: 'Sample subtitle',
    facts: ['sample fact'],
    narrative: 'Sample narrative describing concrete findings.',
    concepts: [],
    files_read: [],
    files_modified: [],
    ...overrides,
  };
}

describe('obs-gate-rules (Arm 1 zero-LLM deterministic filter)', () => {
  it('protects negative investigative findings even when paired with earlier similar discoveries', () => {
    const prior = extractPriorSessionObservations([
      {
        role: 'assistant',
        content: '<observation><type>discovery</type><title>Refreshing launchd registration did not resolve EX_CONFIG</title><narrative>launchd still failed with exit status 78.</narrative></observation>',
      },
    ]);
    const decision = evaluateObservationRules({
      toolName: 'Bash',
      toolInput: { command: 'xattr -d com.apple.macl logs/*' },
      observations: [
        makeObs({
          type: 'discovery',
          title: 'Removing com.apple.macl did not resolve launchd EX_CONFIG',
          narrative: 'Clearing com.apple.macl xattr left exit status 78 unchanged.',
        }),
      ],
      priorObservations: prior,
    });
    expect(decision.verdict).toBe('KEEP');
  });

  it('drops routine git commit/push/status echoes but protects release tag pushes', () => {
    const commitDrop = evaluateObservationRules({
      toolName: 'Bash',
      toolInput: { command: 'git add src/a.ts && git commit -m "fix"' },
      observations: [makeObs({ type: 'change', title: 'Committed statusline separator update' })],
      priorObservations: [],
    });
    expect(commitDrop).toEqual({ verdict: 'DROP', reason: 'routine_git_bookkeeping' });

    const releaseKeep = evaluateObservationRules({
      toolName: 'Bash',
      toolInput: { command: 'git push origin v0.1.0 --tags' },
      observations: [makeObs({ type: 'change', title: 'Committed dist ignore rule and pushed release tag' })],
      priorObservations: [],
    });
    expect(releaseKeep.verdict).toBe('KEEP');
  });

  it('drops routine syntax-check, test-pass, and pure directory location echoes while keeping failures', () => {
    expect(
      evaluateObservationRules({
        toolName: 'Bash',
        toolInput: { command: 'bash -n scripts/wrapper.sh' },
        observations: [makeObs({ title: 'Wrapper syntax check passed after edits' })],
        priorObservations: [],
      }),
    ).toEqual({ verdict: 'DROP', reason: 'routine_syntax_check_pass' });

    expect(
      evaluateObservationRules({
        toolName: 'Bash',
        toolInput: { command: 'ls ~/Desktop/projects/ | grep quota' },
        observations: [makeObs({ title: 'Located the cc-quota-fetcher repository' })],
        priorObservations: [],
      }),
    ).toEqual({ verdict: 'DROP', reason: 'routine_dir_listing' });

    // Failure or negative result must ALWAYS be kept
    expect(
      evaluateObservationRules({
        toolName: 'Bash',
        toolInput: { command: 'pytest tests/test_render.py' },
        observations: [makeObs({ title: 'pytest failed with No module named pytest' })],
        priorObservations: [],
      }).verdict,
    ).toBe('KEEP');
  });

  it('protects mixed pass+fail summaries from routine_verification_pass drops while still dropping clean exit-0 passes', () => {
    // Regression for 2026-11-23 14:20 false drop: positive-looking pass title + failure in subtitle/narrative
    expect(
      evaluateObservationRules({
        toolName: 'Bash',
        toolInput: { command: 'pytest tests/' },
        observations: [
          makeObs({
            title: 'Wave 13 BF3 cross-seat tests pass only for me-distill cron',
            subtitle: 'Both cron runs passed, while the Python and Bun suites failed in both directions across all three BF3 repositories.',
            narrative: 'Cron verification passed, but Python and Bun suites failed.',
          }),
        ],
        priorObservations: [],
      }).verdict,
    ).toBe('KEEP');

    // Failure only in narrative/facts with a plain passing title must also be kept
    expect(
      evaluateObservationRules({
        toolName: 'Bash',
        toolInput: { command: 'bun test' },
        observations: [
          makeObs({
            title: 'Smoke tests passed after restart',
            subtitle: 'Core checks completed',
            narrative: 'Health check passed, but integration check exited with code 1.',
          }),
        ],
        priorObservations: [],
      }).verdict,
    ).toBe('KEEP');

    // Benign "exit code 0" / "0 failed" summaries must still drop as routine verification pass
    expect(
      evaluateObservationRules({
        toolName: 'Bash',
        toolInput: { command: 'python3 -m unittest discover' },
        observations: [
          makeObs({
            title: 'All 157 chatgpt-desktop-factory tests passed',
            subtitle: 'Python unittest discovery completed in 640.130 seconds with exit code 0.',
            narrative: '157 passed, 0 failed, 0 errors.',
          }),
        ],
        priorObservations: [],
      }),
    ).toEqual({ verdict: 'DROP', reason: 'routine_verification_pass' });
  });

  it('never drops security_alert or decision unless exact title duplicate', () => {
    const sec = evaluateObservationRules({
      toolName: 'Bash',
      toolInput: { command: 'git status' },
      observations: [makeObs({ type: 'security_alert', title: 'Committed credential leaked in working tree' })],
      priorObservations: [],
    });
    expect(sec).toEqual({ verdict: 'KEEP', reason: 'protected_type' });
  });

  it('deduplicates Chinese paraphrases but keeps unrelated findings sharing an English identifier', () => {
    const narrative = '啟動時從本機讀取搜尋字典並載入索引設定。';
    const prior = extractPriorSessionObservations([{
      role: 'assistant',
      content: `<observation><title>IndexKit 啟動索引服務時會載入詞彙字典和設定檔</title><narrative>${narrative}</narrative></observation>`,
    }]);
    const run = (title: string) => evaluateObservationRules({
      toolName: 'Read', toolInput: {}, priorObservations: prior,
      observations: [makeObs({ title, subtitle: '', narrative, facts: [] })],
    }).verdict;

    expect(run('IndexKit 索引服務啟動時會載入設定檔與詞彙字典')).toBe('DROP');
    expect(run('IndexKit 透過分段掃描減少磁碟負載')).toBe('KEEP');
  });

  it('drops clean anchored Chinese verification receipts and keeps caveats, failures, and missing error handling', () => {
    const title = '全部 32 項單元測試在 0.041 秒內通過';
    const run = (observations: ParsedObservation[], command = 'bun test packages/cache') => evaluateObservationRules({
      toolName: 'Bash', toolInput: { command }, observations, priorObservations: [],
    }).verdict;
    const clean = makeObs({ title, subtitle: '零錯誤', narrative: '0 failed', facts: ['退出代碼 0'] });

    expect(run([clean])).toBe('DROP');
    expect(run([makeObs({ ...clean, subtitle: '只通過了單元測試' })])).toBe('KEEP');
    expect(run([makeObs({ ...clean, narrative: '舊版執行環境不相容' })])).toBe('KEEP');
    expect(run([clean, makeObs({ ...clean, facts: ['发现文档入口断链'] })])).toBe('KEEP');
    expect(run([clean], 'bun test packages/cache && cp tmp/settings.json src/settings.json')).toBe('KEEP');

    // Opus §4.4 / §5 對抗保留回歸：過關轉折（繁簡英）、沒有錯誤處理、資源危機與高風險動作均必須 KEEP。
    const mustKeep: Array<[string, string, ParsedObservation]> = [
      ['Bash', 'python3 -m unittest', makeObs({ title: '單元測試全綠但實測 715 MB 真實檔案會吃掉 2.4 GB 記憶體', narrative: '大檔記憶體暴漲' })],
      ['Bash', 'pytest', makeObs({ title: '单元测试全绿但真实文件会吃掉 2.4 GB 内存' })],
      ['Bash', 'pytest', makeObs({ title: '全數通過，然而線程池殘留連線未釋放' })],
      ['Bash', 'pytest', makeObs({ title: 'Tests pass but RSS spikes to 2.4 GB on large files' })],
      ['Bash', 'pytest', makeObs({ title: 'Test suite passed — No error handling for 429 responses' })],
      ['Bash', 'pytest', makeObs({ title: '測試通過：沒有錯誤處理與重試機制' })],
      ['Bash', 'pytest', makeObs({ title: '測試通過：無錯誤重試機制會直接中止' })],
      ['Bash', 'ps aux | grep worker', makeObs({ title: 'Worker is running but CPU 100% in spin loop' })],
      ['Grep', '', makeObs({ title: 'Rate limiter is defined in two places with different defaults' })],
      ['Write', '', makeObs({ type: 'discovery', title: 'Benchmark in /private/tmp reveals parser memory spike', files_modified: ['/private/tmp/exp/report.md'] })],
    ];
    for (const [toolName, command, obs] of mustKeep) {
      const toolInput = toolName === 'Bash' ? { command } : toolName === 'Write' ? { file_path: obs.files_modified?.[0] } : { pattern: 'rateLimit' };
      expect(evaluateObservationRules({ toolName, toolInput, observations: [obs], priorObservations: [] }).verdict).toBe('KEEP');
    }
  });

  it('keeps distinct-title negative findings even with identical narrative and keeps new negative details under same title', () => {
    const title = '快取整合驗證結果';
    const prior = extractPriorSessionObservations([{
      role: 'assistant',
      content: `<observation><title>${title}</title><narrative>上一版讀取測試失敗。</narrative></observation>
        <observation><type>decision</type><title>Deployment strategy</title><narrative>Deploy manually.</narrative></observation>
        <observation><title>Parser OOM on 700MB input</title><narrative>Verified via tracemalloc reproduction.</narrative></observation>`,
    }]);
    const run = (obs: ParsedObservation) => evaluateObservationRules({
      toolName: 'Read', toolInput: {}, observations: [obs], priorObservations: prior,
    }).verdict;

    expect(run(makeObs({ title, subtitle: '', narrative: '新版寫入測試失敗。', facts: [] }))).toBe('KEEP');
    expect(run(makeObs({
      type: 'decision', title: 'Deployment strategy', subtitle: '', narrative: 'Deploy via the scheduler.', facts: [],
    }))).toBe('KEEP');
    expect(run(makeObs({
      title: 'SQLite lock contention under burst writes', subtitle: '', narrative: 'Verified via tracemalloc reproduction.', facts: [],
    }))).toBe('KEEP');
  });

  it('drops only confirmed scratch changes and keeps mixed or ambiguous destinations', () => {
    const path = '/var/folders/ab/session/snapshot.json';
    const scratch = makeObs({ type: 'change', title: '暫存沙盒寫入測試快照', files_modified: [path] });
    const run = (toolName: string, toolInput: unknown, observation = scratch) => evaluateObservationRules({
      toolName, toolInput, observations: [observation], priorObservations: [],
    }).verdict;

    expect(run('Write', JSON.stringify({ file_path: path }))).toBe('DROP');
    expect(run('Write', { file_path: '/workspace/src/snapshot.json' })).toBe('KEEP');
    expect(run('Edit', { file_path: path }, { ...scratch, facts: ['寫入途中失敗'] })).toBe('KEEP');
    expect(run('Edit', { file_path: path }, { ...scratch, files_modified: [path, 'src/snapshot.json'] })).toBe('KEEP');
    const traversal = '/tmp/../workspace/snapshot.json';
    expect(run('Write', { file_path: traversal }, { ...scratch, files_modified: [traversal] })).toBe('KEEP');
    expect(run('Bash', { command: `mkdir -p /var/folders/ab/session && cp source.json ${path}` })).toBe('DROP');
    expect(run('Bash', { command: `cp source.json ${path} && cp source.json src/snapshot.json` })).toBe('KEEP');
  });

  it('keeps same-named mutations in different directories and new details on the same path', () => {
    const title = 'Updated serializer defaults';
    const narrative = 'The serializer defaults to compact JSON.';
    const prior = extractPriorSessionObservations([{
      role: 'assistant',
      content: `<observation><type>change</type><title>${title}</title><narrative>${narrative}</narrative>
        <files_modified><file>packages/api/config.ts</file></files_modified></observation>`,
    }]);
    const run = (path: string, details = narrative) => evaluateObservationRules({
      toolName: 'Edit', toolInput: { file_path: path }, priorObservations: prior,
      observations: [makeObs({ type: 'change', title, subtitle: '', narrative: details, facts: [], files_modified: [path] })],
    }).verdict;

    expect(run('packages/api/config.ts')).toBe('DROP');
    expect(run('packages/web/config.ts')).toBe('KEEP');
    expect(run('packages/api/config.ts', 'The serializer defaults to pretty JSON.')).toBe('KEEP');
  });

  it('drops routine status and inventory but keeps tags, compound writes and findings', () => {
    const run = (command: string, observation: ParsedObservation) => evaluateObservationRules({
      toolName: 'Bash', toolInput: { command }, observations: [observation], priorObservations: [],
    }).verdict;
    const count = makeObs({ title: 'Local database contains 24 rows' });

    expect(run('git status --short --branch', makeObs({ title: 'Working tree is clean' }))).toBe('DROP');
    expect(run('git status', makeObs({ title: 'Working tree is clean because pre-commit hook silently reverted config.json' }))).toBe('KEEP');
    expect(run('git push origin v2.3.0', makeObs({ type: 'change', title: 'Pushed v2.3.0' }))).toBe('KEEP');
    expect(run('git status && cp new.json src/settings.json', makeObs({ title: 'Working tree is clean' }))).toBe('KEEP');
    expect(run("sqlite3 local.db 'SELECT count(*) FROM records;'", count)).toBe('DROP');
    expect(run("sqlite3 local.db 'SELECT count(*) FROM records;'", { ...count, facts: ['Unexpectedly doubled overnight'] })).toBe('KEEP');
    expect(run("sqlite3 local.db 'SELECT count(*) FROM records; DELETE FROM records;'", count)).toBe('KEEP');
  });

  it('extracts the most recent 25 titled observations in chronological order', () => {
    const content = Array.from({ length: 31 }, (_, i) => `<observation><title>索引紀錄 ${i}</title></observation>`).join('');
    const prior = extractPriorSessionObservations([{ role: 'assistant', content }]);
    expect(prior).toHaveLength(25);
    expect(prior[0].titleNorm).toBe('索引紀錄 6');
    expect(prior[24].titleNorm).toBe('索引紀錄 30');
  });

  it('extracts bounded unique prior_session_titles from conversationHistory with zero DB queries', () => {
    const history = Array.from({ length: 20 }, (_, i) => ({
      role: 'assistant' as const,
      content: `<observation><title>Title #${i + 1}</title></observation>`,
    }));
    const titles = extractPriorSessionTitles(history);
    expect(titles).toHaveLength(15);
    expect(titles[0]).toBe('Title #6');
    expect(titles[14]).toBe('Title #20');
  });
});
