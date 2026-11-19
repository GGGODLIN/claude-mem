# Claude-Mem: AI Development Instructions

Claude-mem is a Claude Code plugin providing persistent memory across sessions. It captures tool usage, compresses observations using the Claude Agent SDK, and injects relevant context into future sessions.

## Build

```bash
# Lightweight worker sync (current default on this machine — avoids scripts/build-hooks.js
# rejecting hand-edited shell templates unless --write-shell-templates is intentionally run):
# sync only built worker-service.cjs (+ .map) into the active cache dir and restart:
cp plugin/scripts/worker-service.cjs plugin/scripts/worker-service.cjs.map \
  ~/.claude/plugins/cache/thedotmack/claude-mem/<version>/scripts/
bun ~/.claude/plugins/cache/thedotmack/claude-mem/<version>/scripts/worker-service.cjs restart

# Full plugin build & sync (only when hooks/shell templates are intentionally regenerated):
# npm run build               # runs scripts/build-hooks.js; pass --write-shell-templates if hooks.json changed
# rsync -a --delete --exclude=node_modules plugin/ ~/.claude/plugins/cache/thedotmack/claude-mem/<version>/
# (cd ~/.claude/plugins/cache/thedotmack/claude-mem/<version> && bun install)
# bun ~/.claude/plugins/cache/thedotmack/claude-mem/<version>/scripts/worker-service.cjs restart
```

> `npm run build-and-sync` assumes `~/.claude/plugins/marketplaces/thedotmack/` exists (`scripts/restart-marketplace-worker.cjs` exits if that path is absent).

## File Locations

- **Source**: `<project-root>/src/`
- **Built Plugin**: `<project-root>/plugin/`
- **Installed Plugin & Active Worker**: `~/.claude/plugins/cache/thedotmack/claude-mem/<version>/` (e.g., `13.35.0/`; `~/.claude/plugins/marketplaces/thedotmack/` does not exist on this machine)
- **Database**: `~/.claude-mem/claude-mem.db`
- **Chroma**: `~/.claude-mem/chroma/`

## Requirements

- **Bun** (all platforms - auto-installed if missing)
- **uv** (all platforms - auto-installed if missing, provides Python for Chroma)
- Node.js

## Documentation

**Public Docs**: https://docs.claude-mem.ai (Mintlify)
**Source**: `docs/public/` - MDX files, edit `docs.json` for navigation
**Deploy**: Auto-deploys from GitHub on push to main

## Important

No need to edit the changelog ever, it's generated automatically.
