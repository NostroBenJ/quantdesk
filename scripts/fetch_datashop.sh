#!/usr/bin/env bash
# Pull a Cboe DataShop order over SFTP.
#
#   scripts/fetch_datashop.sh <user> <host> [remote_path] [local_dir]
#
# Uses the dedicated key at ~/.ssh/cboe_datashop. DataShop requires RSA
# 2048/4096 -- an ed25519 key is silently unusable there, which looks like a
# permissions problem rather than an algorithm one.
#
# Resumable: sftp's `get -a` appends to partial files, so a dropped connection
# on file 2,900 of 3,500 costs you one file rather than the whole order.
#
# The count check at the end is the point. A partial download looks exactly
# like a complete one until a study quietly runs on two thirds of the history.

set -euo pipefail

USER_NAME="${1:?usage: fetch_datashop.sh <user> <host> [remote_path] [local_dir]}"
HOST="${2:?missing host}"
REMOTE="${3:-.}"
LOCAL="${4:-$HOME/Downloads/datashop}"
KEY="$HOME/.ssh/cboe_datashop"

if [ ! -f "$KEY" ]; then
  echo "no key at $KEY" >&2
  exit 1
fi

mkdir -p "$LOCAL"

echo "listing $REMOTE ..."
sftp -i "$KEY" -o BatchMode=yes -b - "${USER_NAME}@${HOST}" <<EOF || true
ls -l ${REMOTE}
EOF

echo
echo "downloading ${REMOTE} -> ${LOCAL}"
# -a resumes partial transfers; -r recurses into the order directory.
sftp -i "$KEY" -o BatchMode=yes -b - "${USER_NAME}@${HOST}" <<EOF
lcd ${LOCAL}
get -a -r ${REMOTE}
EOF

echo
echo "--- what landed ---"
FILES=$(find "$LOCAL" -type f \( -name '*.csv' -o -name '*.gz' -o -name '*.zip' \) | wc -l)
BYTES=$(du -sh "$LOCAL" 2>/dev/null | cut -f1)
echo "files: ${FILES}"
echo "size : ${BYTES}"
echo
echo "Check that file count against the order. A 2012-2026 daily-grouped SPY"
echo "order should be roughly 3,500 files (~250 trading days a year). Monthly"
echo "grouping, roughly 170. A short count means the transfer stopped early --"
echo "re-run this script, it resumes."
echo
echo "Then:  python scripts/ingest_datashop.py \"$LOCAL\" --dry-run"
