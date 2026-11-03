#!/bin/sh
# Vercel build: the site is static, so all this does is point the page at
# the hosted API (set PIPESPACING_API in the Vercel project's env vars).
set -e

if [ -n "$PIPESPACING_API" ]; then
  printf 'window.PIPESPACING_API = "%s";\n' "${PIPESPACING_API%/}" > frontend/config.js
  echo "frontend -> $PIPESPACING_API"
else
  # Still deploy, so the page is live before the API is. It tells visitors the
  # API isn't connected; set the variable and redeploy once the API is up.
  printf 'window.PIPESPACING_API = null;\n' > frontend/config.js
  echo "warning: PIPESPACING_API is not set; the page will show 'API not configured'." >&2
fi

# The label-review tool edits files on the machine running the API; it is
# local-only, so keep it out of the public site.
rm -f frontend/review.html
