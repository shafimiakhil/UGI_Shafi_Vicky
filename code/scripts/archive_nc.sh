#!/usr/bin/env bash
# No-clobber archive copy (CLAUDE.md 13.4, archive rule): archive_nc.sh <destination folder> <file>...
# Copies each file into the folder (created if needed) keeping its timestamps; stops with exit 1, copying nothing, if
# any destination already exists or if two files of one call share a name.
set -u
dst=$1; shift
mkdir -p "$dst"
seen=""
for f in "$@"; do
  b=$(basename "$f")
  case "|$seen|" in *"|$b|"*) echo "archive_nc: two files named $b in one call, nothing copied; use one folder for each" >&2; exit 1;; esac
  seen="$seen|$b"
  if [ -e "$dst/$b" ]; then echo "archive_nc: $dst/$b exists, nothing copied" >&2; exit 1; fi
done
for f in "$@"; do cp -p "$f" "$dst/" || exit 1; done
echo "archived $# file(s) to $dst"
