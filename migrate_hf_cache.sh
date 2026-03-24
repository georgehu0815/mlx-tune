#!/usr/bin/env bash
# Migrate HuggingFace cache to external SSD
# Preserves symlinks (critical for HF hub blob/snapshot structure)

set -euo pipefail

SRC="$HOME/.cache/huggingface"
DST="/Volumes/ExternalSSD/.cache/huggingface"
SHELL_RC="$HOME/.zshrc"  # change to ~/.bashrc if using bash

# ── 1. Preflight checks ────────────────────────────────────────────────────────
echo "==> Checking source..."
if [[ ! -d "$SRC" ]]; then
  echo "ERROR: Source $SRC not found." >&2; exit 1
fi

echo "==> Checking destination volume..."
if [[ ! -d "/Volumes/ExternalSSD" ]]; then
  echo "ERROR: /Volumes/ExternalSSD is not mounted." >&2; exit 1
fi

echo "==> Source size:"
du -sh "$SRC"

echo ""
echo "==> Destination free space:"
df -h "/Volumes/ExternalSSD" | tail -1

echo ""
read -r -p "Continue with migration to $DST? [y/N] " confirm
[[ "$confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }

# ── 2. Copy files (preserve symlinks, permissions, timestamps) ─────────────────
echo ""
echo "==> Copying to $DST ..."
mkdir -p "$DST"
# rsync: -a = archive (symlinks, perms, times), --progress = live stats
rsync -a --progress --stats "$SRC/" "$DST/"

echo ""
echo "==> Verifying copy..."
SRC_COUNT=$(find "$SRC" | wc -l | tr -d ' ')
DST_COUNT=$(find "$DST" | wc -l | tr -d ' ')
echo "    Source files/dirs : $SRC_COUNT"
echo "    Dest   files/dirs : $DST_COUNT"
if [[ "$SRC_COUNT" -ne "$DST_COUNT" ]]; then
  echo "WARNING: File counts differ — please inspect before deleting source." >&2
else
  echo "    Counts match ✓"
fi

# ── 3. Replace source with a symlink ──────────────────────────────────────────
echo ""
read -r -p "Replace $SRC with a symlink to $DST? [y/N] " do_symlink
if [[ "$do_symlink" =~ ^[Yy]$ ]]; then
  BACKUP="${SRC}.bak.$(date +%Y%m%d%H%M%S)"
  echo "==> Moving original to $BACKUP ..."
  mv "$SRC" "$BACKUP"
  echo "==> Creating symlink: $SRC -> $DST"
  ln -s "$DST" "$SRC"
  echo "    Symlink created ✓"
  echo "    (Original backed up at $BACKUP — delete manually once satisfied)"
fi

# ── 4. Export HF env vars in shell rc ─────────────────────────────────────────
echo ""
read -r -p "Add HF_HOME env var to $SHELL_RC? [y/N] " do_env
if [[ "$do_env" =~ ^[Yy]$ ]]; then
  # Remove any existing HF_HOME / HF_HUB_CACHE lines first
  if grep -q "HF_HOME\|HF_HUB_CACHE\|TRANSFORMERS_CACHE" "$SHELL_RC" 2>/dev/null; then
    echo "==> Removing existing HF env vars from $SHELL_RC ..."
    sed -i '' '/HF_HOME\|HF_HUB_CACHE\|TRANSFORMERS_CACHE/d' "$SHELL_RC"
  fi

  cat >> "$SHELL_RC" << 'EOF'

# HuggingFace cache on external SSD
export HF_HOME="/Volumes/ExternalSSD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
EOF
  echo "    Env vars added to $SHELL_RC ✓"
  echo "    Run: source $SHELL_RC"
fi

echo ""
echo "==> Done! Summary:"
echo "    New cache location : $DST"
echo "    Symlink            : $SRC -> $DST  (if chosen)"
echo "    Env var (HF_HOME)  : $DST  (if chosen)"
echo ""
echo "    To verify HF CLI sees the new location, run:"
echo "      huggingface-cli env"


# The migration completed successfully. Here's the verification summary:

# All 4 steps completed:

# Copy — /Volumes/ExternalSSD/.cache/huggingface/hub exists with content (last modified Mar 20)
# Symlink — ~/.cache/huggingface → /Volumes/ExternalSSD/.cache/huggingface ✓
# Backup — ~/.cache/huggingface.bak.* exists (original preserved)
# Env vars — Added to ~/.zshrc (lines 59–62): HF_HOME, HF_HUB_CACHE, HUGGINGFACE_HUB_CACHE, TRANSFORMERS_CACHE
# One thing to note: The backup dir (~/.cache/huggingface.bak.*) contains only a hub subdirectory, which matches the destination — counts look consistent.

# Once you're satisfied with the new location, you can delete the backup:


# rm -rf ~/.cache/huggingface.bak.*
