#!/bin/sh
# Vercel build: the site is static, so all this does is point the page at
# the hosted API (set PIPESPACING_API in the Vercel project's env vars).
set -e

if [ -z "$PIPESPACING_API" ]; then
  echo "error: set PIPESPACING_API to the API URL (e.g. https://<user>-pipespacing.hf.space)" >&2
  exit 1
fi
printf 'window.PIPESPACING_API = "%s";\n' "${PIPESPACING_API%/}" > frontend/config.js

# The label-review tool edits files on the machine running the API; it is
# local-only, so keep it out of the public site.
rm -f frontend/review.html

echo "frontend -> $PIPESPACING_API"
