#!/usr/bin/env bash
# Rebuild the vendored, pinned OSS assets for assembly-mode web design.
#
# Produces src/static/vendor/{app.css,htmx.min.js,alpine.min.js}. These are the
# self-contained assets the assembler INLINES into every generated page, so the
# artifact renders identically under Playwright, in a live-preview iframe, and as
# a download — no external asset fetch, no CSP surprises.
#
# app.css is compiled by Tailwind v4 + DaisyUI v5, scanning ONLY the v2 block
# partials (scripts/asset_build/input.css @source), so it carries only the
# utilities/components those blocks actually use (~48KB). Re-run this after adding
# or editing catalog blocks. Requires node + npm; node_modules is gitignored.
#
# Pinned versions live in scripts/asset_build/package.json + the curl URLs below.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
VENDOR="$ROOT/src/static/vendor"
HTMX_VERSION="2.0.10"
ALPINE_VERSION="3.15.12"

mkdir -p "$VENDOR"

echo "==> installing pinned build tooling (Tailwind v4 + DaisyUI v5)"
( cd "$HERE/asset_build" && npm install --silent )

echo "==> compiling app.css from v2 block partials"
( cd "$HERE/asset_build" && npx @tailwindcss/cli -i input.css -o "$VENDOR/app.css" --minify )

echo "==> fetching htmx@$HTMX_VERSION"
curl -fsSL -o "$VENDOR/htmx.min.js" \
  "https://cdn.jsdelivr.net/npm/htmx.org@${HTMX_VERSION}/dist/htmx.min.js"

echo "==> fetching alpinejs@$ALPINE_VERSION"
curl -fsSL -o "$VENDOR/alpine.min.js" \
  "https://cdn.jsdelivr.net/npm/alpinejs@${ALPINE_VERSION}/dist/cdn.min.js"

echo "==> done:"
ls -la "$VENDOR"
