#!/bin/sh
# Update BREE on an edge box with automatic rollback.
#   deploy/update.sh [GIT_REF]        default HEAD of the checkout this script lives in
#   deploy/update.sh --rollback       switch back to the previous release and restart
# Layout under $BREE_HOME (default ~/bree):   releases/<stamp>_<sha>/   current -> releases/...   previous -> ...
# Data (~/bree-data) and config (~/.config/bree) live outside releases and are never touched.
# Steps: export the ref -> reuse the previous venv if pyproject.toml is unchanged, else build a new one -> smoke test ->
# switch `current` atomically -> restart -> wait for health -> on any failure switch back and restart the old release.
# Overridable for tests / other init systems: RESTART_CMD, HEALTH_CMD, SMOKE_CMD, INSTALL_CMD, HEALTH_WAIT, KEEP, PYTHON.
set -eu
SRC=$(cd "$(dirname "$0")/.." && pwd)
BASE=${BREE_HOME:-$HOME/bree}
RESTART_CMD=${RESTART_CMD:-systemctl --user restart bree-shadow}
HEALTH_CMD=${HEALTH_CMD:-systemctl --user is-active --quiet bree-shadow}
HEALTH_WAIT=${HEALTH_WAIT:-45}
KEEP=${KEEP:-3}
PYTHON=${PYTHON:-python3}
mkdir -p "$BASE/releases"

switch() {   # atomic symlink swap: $1 = target release dir
    ln -sfn "$1" "$BASE/current.new"
    mv -T "$BASE/current.new" "$BASE/current"
}

restart_and_check() {
    sh -c "$RESTART_CMD" || return 1
    sleep "$HEALTH_WAIT"
    sh -c "$HEALTH_CMD"
}

if [ "${1:-}" = "--rollback" ]; then
    [ -L "$BASE/previous" ] || { echo "no previous release recorded" >&2; exit 1; }
    cur=$(readlink "$BASE/current"); prev=$(readlink "$BASE/previous")
    switch "$prev"; ln -sfn "$cur" "$BASE/previous"
    restart_and_check || { echo "rolled back to $prev but it is not healthy: check journalctl --user -u bree-shadow" >&2; exit 2; }
    echo "rolled back to $prev"; exit 0
fi

REF=${1:-HEAD}
SHA=$(git -C "$SRC" rev-parse --short "$REF")
NEW="$BASE/releases/$(date +%Y%m%d_%H%M%S)_$SHA"
mkdir -p "$NEW"
git -C "$SRC" archive "$REF" | tar -x -C "$NEW"

OLD=$(readlink "$BASE/current" 2>/dev/null || true)
if [ -n "$OLD" ] && [ -d "$OLD/venv" ] && cmp -s "$OLD/pyproject.toml" "$NEW/pyproject.toml"; then
    ln -s "$OLD/venv" "$NEW/venv"          # same dependencies: share the environment (torch alone is GBs)
else
    "$PYTHON" -m venv "$NEW/venv"
    INSTALL=${INSTALL_CMD:-"$NEW/venv/bin/pip install --quiet $NEW[vision,export]"}
    sh -c "$INSTALL" || { echo "dependency install failed; nothing switched" >&2; rm -rf "$NEW"; exit 1; }
fi
for d in models; do [ -e "$SRC/$d" ] && [ ! -e "$NEW/$d" ] && ln -s "$(cd "$SRC" && pwd -P)/$d" "$NEW/$d"; done   # weights are not in git

SMOKE=${SMOKE_CMD:-PYTHONPATH=$NEW/src $NEW/venv/bin/python -c 'import bree.pipeline, bree.trust.cli'}
sh -c "$SMOKE" || { echo "smoke test failed; nothing switched" >&2; rm -rf "$NEW"; exit 1; }

[ -n "$OLD" ] && ln -sfn "$OLD" "$BASE/previous"
switch "$NEW"
if restart_and_check; then
    echo "updated to $NEW"
else
    echo "new release unhealthy: rolling back" >&2
    if [ -n "$OLD" ]; then
        switch "$OLD"
        restart_and_check || echo "old release also failed its health check: look at the logs" >&2
    fi
    exit 1
fi
# keep the newest $KEEP releases (never current or previous)
ls -1dt "$BASE"/releases/* | tail -n +$((KEEP + 1)) | while read -r r; do
    [ "$r" = "$(readlink "$BASE/current")" ] || [ "$r" = "$(readlink "$BASE/previous" 2>/dev/null)" ] || rm -rf "$r"
done
