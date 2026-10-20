#!/usr/bin/env bash
# Rebuild the guide PDFs in docs/ from the HTML sources in this folder.
# Needs Google Chrome (or Chromium); set CHROME to override the path.
#
#   bash docs/src/build_pdfs.sh
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
OUT="$(dirname "$SRC")"

if [ -z "${CHROME:-}" ]; then
  for c in "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
           "/Applications/Chromium.app/Contents/MacOS/Chromium" \
           google-chrome chromium chromium-browser; do
    if [ -x "$c" ] || command -v "$c" >/dev/null 2>&1; then CHROME="$c"; break; fi
  done
fi
[ -n "${CHROME:-}" ] || { echo "Chrome/Chromium not found; set CHROME=/path/to/chrome" >&2; exit 1; }

build() {   # build <source.html> <output.pdf>
  "$CHROME" --headless --disable-gpu --no-pdf-header-footer \
            --print-to-pdf="$OUT/$2" "file://$SRC/$1" 2>/dev/null
  echo "wrote $OUT/$2"
}

build getting_started.html Kinefly2_Getting_Started.pdf
build daily_checklist.html Kinefly2_Daily_Checklist.pdf
