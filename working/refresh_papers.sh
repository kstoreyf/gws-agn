#!/bin/bash
# Refresh the paper from the current run results: working/paper_prd/ (the one maintained paper; the letter is
# archived and nothing here reads it).
# Real results replace red placeholders wherever a run summary exists: a key of values/production_map.json turns real
# when every run it names has its summary.json; so does each macro of the experiment (map block "classes").
# Usage: bash working/refresh_papers.sh            (from anywhere)
# Exit status 1 if any step fails (values, audit, figure check, LaTeX build, notes left in the text).
set -u
set -o pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/.conda/envs/jax/bin/python
TEX="$HOME/bin/tectonic"
LOG="$(mktemp -d "${TMPDIR:-/tmp}/refresh_papers.XXXXXX")"
FAILED=()
step() { echo; echo "== $*"; }

# run NAME NLINES CMD...: run CMD (stdout and stderr to $LOG/NAME.log), show its last NLINES lines, record a failure
run() {
  local name=$1 n=$2; shift 2
  if "$@" > "$LOG/$name.log" 2>&1; then
    grep -v "RuntimeWarning\|return (gamma" "$LOG/$name.log" | tail -n "$n"
  else
    tail -n 15 "$LOG/$name.log"
    echo "!! $name FAILED (full log: $LOG/$name.log)"
    FAILED+=("$name")
  fi
}

# build DIR TAG: LaTeX build of DIR/main.tex; fails on a LaTeX error, undefined references or a PDF that was not rewritten
build() {
  local d=$1 tag=$2 t0
  t0=$(date +%s)
  cd "$W/$d" || { FAILED+=("$tag build"); return; }
  if ! "$TEX" main.tex > "$LOG/$tag-tex.log" 2>&1; then
    grep -E "^error" "$LOG/$tag-tex.log" | head -5
    echo "!! $tag LaTeX build FAILED (full log: $LOG/$tag-tex.log)"
    FAILED+=("$tag build")
    return
  fi
  grep -E "Writing" "$LOG/$tag-tex.log"
  if grep -qiE "undefined (reference|citation|control sequence)|Reference .* undefined|Citation .* undefined" "$LOG/$tag-tex.log"; then
    grep -iE "undefined" "$LOG/$tag-tex.log" | head -5
    echo "!! $tag has undefined references or macros"
    FAILED+=("$tag undefined")
  fi
  if [ "$(stat -c %Y main.pdf)" -lt "$t0" ]; then
    echo "!! $tag main.pdf was not rewritten"
    FAILED+=("$tag pdf")
  fi
  pdfinfo main.pdf | grep Pages
}

step "values (production keys and the experiment's macros from Analysis 16), audit, figures and their check"
cd "$W/paper_prd" || exit 1
run prd-values 2 "$PY" scripts/build_values.py
run prd-audit 1 "$PY" scripts/audit_values.py
run prd-figures 2 "$PY" scripts/make_figures.py
grep -q "MISMATCH" "$LOG/prd-figures.log" && { grep "MISMATCH" "$LOG/prd-figures.log" | head -5; FAILED+=("figure check"); }
step "build"
build paper_prd prd

step "notes to the author left in the rendered text (must be empty)"
# the red word "placeholder" alone on a line is the mark of a figure panel that waits for a run; anything else is a note
txt="$(pdftotext "$W/paper_prd/main.pdf" - 2>/dev/null)"
notes="$(printf '%s\n' "$txt" | grep -inE '\b(TODO|FIXME|TBD|XXX|placeholder)\b' | grep -viE '^[0-9]+:[[:space:]]*placeholder[[:space:]]*$')"
marks="$(printf '%s\n' "$txt" | grep -ciE '^[[:space:]]*placeholder[[:space:]]*$')"
if [ -n "$notes" ]; then
  echo "paper_prd:"; printf '%s\n' "$notes" | head -5
  FAILED+=("paper_prd notes")
fi
echo "paper_prd: figure panels still marked placeholder: $marks"

step "keys and experiment macros (values/production_map.json)"
"$PY" "$W/paper_prd/scripts/production.py" --status | grep -v "^wrote"

step "result macros: real, and waiting for runs"
[ ${#FAILED[@]} -gt 0 ] && echo "FAILED: ${FAILED[*]} (logs in $LOG)"
"$PY" "$W/paper_prd/scripts/production.py" --report
[ ${#FAILED[@]} -eq 0 ] && rm -rf "$LOG"
[ ${#FAILED[@]} -eq 0 ]
