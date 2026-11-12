#!/usr/bin/env bash
# End-to-end local demo against the fake four-platform server.
set -euo pipefail
cd "$(dirname "$0")/.."

export CARSCOUT_DATABASE_URL="${CARSCOUT_DATABASE_URL:-postgresql://carscout:carscout@127.0.0.1:5432/carscout}"
export REDIS_URL="${REDIS_URL:-redis://127.0.0.1:6379/0}"
export FIXTURE_BASE_URL="${FIXTURE_BASE_URL:-http://127.0.0.1:8088}"
export DATA_DIR="${DATA_DIR:-./data}"

if [ -z "${LD_LIBRARY_PATH:-}" ] && [ -d /tmp/chromelibs/usr/lib/x86_64-linux-gnu ]; then
  export LD_LIBRARY_PATH=/tmp/chromelibs/usr/lib/x86_64-linux-gnu  # dev sandbox only
fi
echo "== init-db"; carscout init-db >/dev/null
echo "== fixture"; carscout fixture --port 8088 >/tmp/carscout-fixture.log 2>&1 &
FIXTURE_PID=$!
trap 'kill $FIXTURE_PID 2>/dev/null || true' EXIT
sleep 1

echo "== run (A/B/C/D)"; carscout run --date 2026-11-12
echo "== status"; carscout status | head -40
echo "== export 凯美瑞"; carscout export --series 凯美瑞 --days 30
echo "== review export"; carscout review --export --out /tmp/review_queue.csv
echo "   -> edit /tmp/review_queue.csv (decision=merge|split) then:"
echo "   carscout review --import-file /tmp/review_queue.csv"
