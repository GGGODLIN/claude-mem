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
