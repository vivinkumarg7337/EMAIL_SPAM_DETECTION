#!/usr/bin/env bash
# Copy the checked-in project files into the existing GitHub repository.
set -euo pipefail

SOURCE="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="/tmp/opencode/EMAIL_SPAM_DETECTION"

rm -rf "$TARGET"
git clone https://github.com/vivinkumarg7337/EMAIL_SPAM_DETECTION.git "$TARGET"

for path in app scripts data model reports README.md GOOGLE_DOCS_REPORT.md requirements.txt run.sh; do
  rm -rf "$TARGET/$path"
  cp -R "$SOURCE/$path" "$TARGET/$path"
done

# Vendored training engine: copy source only, not local caches.
rm -rf "$TARGET/atlas-engine"
mkdir -p "$TARGET/atlas-engine"
cp -R "$SOURCE/atlas-engine/atlas_engine" "$SOURCE/atlas-engine/atlas_worker" "$TARGET/atlas-engine/"
cp "$SOURCE/atlas-engine/README.md" "$SOURCE/atlas-engine/LICENSE" "$TARGET/atlas-engine/"

find "$TARGET" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$TARGET" -type f -name '*.pyc' -delete
rm -rf "$TARGET/backups" "$TARGET/.venv"

cat > "$TARGET/.gitignore" <<'EOF'
.venv/
backups/
__pycache__/
*.pyc
*.pyo
.DS_Store
EOF

cd "$TARGET"
git add -A
git status --short | sed -n '1,120p'
git commit -m "Train spam detector with automatic threshold and add website" || true
git push origin HEAD
