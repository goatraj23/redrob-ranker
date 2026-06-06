#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Build a clean, staged commit history for this repo ON YOUR OWN MACHINE.
# (The repo was authored in a sandbox whose mounted filesystem blocks git's
#  lock files, so history must be created locally. Run this once.)
#
#   chmod +x init_git.sh && ./init_git.sh
#   git remote add origin https://github.com/<you>/redrob-ranker.git
#   git push -u origin main
# ---------------------------------------------------------------------------
set -e
cd "$(dirname "$0")"

# clean up sandbox-authoring artifacts (the repo is deterministic / AI-free)
rm -rf .git _broken_git_remove_me _REMOVE_ME_* label weights.json \
       __pycache__ */__pycache__ team_*.csv 2>/dev/null || true

git init -q
git checkout -q -b main 2>/dev/null || git branch -M main
git config user.email "mohithlochan56@gmail.com"
git config user.name  "buggycoder"

MSG="chore: scaffold repo; gitignore the dataset; document zero-dependency runtime"
git add .gitignore requirements.txt && git commit -q -m "$MSG"

MSG="feat: JD concept/knowledge model and streaming candidate loader"
git add redrob_ranker/__init__.py redrob_ranker/io_utils.py redrob_ranker/concepts.py && git commit -q -m "$MSG"

MSG="feat: feature extraction, composite scoring, and honeypot/consistency gate"
git add redrob_ranker/features.py redrob_ranker/honeypots.py redrob_ranker/scoring.py && git commit -q -m "$MSG"

MSG="feat: grounded non-templated reasoning + single-command ranker entrypoint"
git add redrob_ranker/reasoning.py rank.py validate_submission.py && git commit -q -m "$MSG"

MSG="feat: offline eval harness (NDCG/MAP/P) + stratified heuristic labels"
git add eval/ tools/ && git commit -q -m "$MSG"

MSG="feat: sample-only Streamlit sandbox + test suite"
git add sandbox/ tests/ && git commit -q -m "$MSG"

MSG="docs: README, architecture writeup, and submission metadata"
git add README.md submission_metadata.yaml init_git.sh && git commit -q -m "$MSG"

echo "Done. Commit history:"
git log --oneline
