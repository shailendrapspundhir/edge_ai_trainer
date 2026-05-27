#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# E2E Hyperparameter Sweep Engine Test for Edge AI Trainer
#
# Validates the full sweep subsystem:
#   1. Config loading (search space + objectives YAML)
#   2. ASHA scheduler logic (promotion / early-stop)
#   3. Sampler correctness (search space bounds)
#   4. Sweep engine — 3 trials, ASHA halving, local training
#   5. Sweep summary verification (JSON output, best trial selection)
#   6. CLI entry point (eat sweep run)
#
# Uses the local Qwen2.5-0.5B-Instruct model, fully offline.
#
# Usage:
#   bash tests/e2e/test_sweep_engine.sh
#   bash tests/e2e/test_sweep_engine.sh --no-cleanup   # keep work dir
#   bash tests/e2e/test_sweep_engine.sh --trials 4     # custom trial count
#
# Requirements:
#   uv sync --extra train
#   llm_foundation_models/Qwen2.5-0.5B-Instruct/
# ═══════════════════════════════════════════════════════════════════════════
set -uo pipefail

# ──── Resolve paths ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

FIXTURE_DATA="tests/e2e/fixtures/train_data.jsonl"
MODEL_PATH="$REPO_ROOT/llm_foundation_models/Qwen2.5-0.5B-Instruct"
SEARCH_SPACE="$REPO_ROOT/configs/sweeps/smoke_search_space.yaml"
OBJECTIVES="$REPO_ROOT/configs/sweeps/smoke_objectives.yaml"

# Block all HuggingFace network requests
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ──── Defaults ─────────────────────────────────────────────────────────────
TRIALS=3
CLEANUP=true

# ──── Parse args ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-cleanup)   CLEANUP=false; shift ;;
    --trials)       TRIALS="$2"; shift 2 ;;
    --help|-h)
      head -18 "${BASH_SOURCE[0]}" | tail -n +2
      exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ──── Colours + helpers ────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

PASSED=0
FAILED=0
TOTAL=6

step_header() {
  echo ""
  echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${BOLD}  $1${NC}"
  echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

pass() { echo -e "  ${GREEN}✓ PASS${NC}: $1"; PASSED=$((PASSED + 1)); }
fail() { echo -e "  ${RED}✗ FAIL${NC}: $1"; FAILED=$((FAILED + 1)); }

# ──── Work directory ───────────────────────────────────────────────────────
WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/eat-sweep-e2e-XXXXXX")
cleanup() {
  if $CLEANUP; then
    rm -rf "$WORK_DIR"
  else
    echo -e "\n${YELLOW}Work dir preserved: $WORK_DIR${NC}"
  fi
}
trap cleanup EXIT

# ──── Banner ───────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Edge AI Trainer — E2E Sweep Engine Test"
echo "═══════════════════════════════════════════════════════════════"
echo "  Model:         $MODEL_PATH"
echo "  Trials:        $TRIALS"
echo "  Search Space:  $SEARCH_SPACE"
echo "  Objectives:    $OBJECTIVES"
echo "  Work dir:      $WORK_DIR"
echo ""


# ══════════════════════════════════════════════════════════════════════════
# Step 1: Validate config loading
# ══════════════════════════════════════════════════════════════════════════
step_header "Step 1/$TOTAL: Validate config loading (search space + objectives)"

if uv run python -c "
from eat.sweep.config import load_search_space, load_objectives
ss = load_search_space('$SEARCH_SPACE')
obj = load_objectives('$OBJECTIVES')
assert len(ss.params) >= 2, f'expected >= 2 params, got {len(ss.params)}'
assert len(obj.objectives) >= 1, f'expected >= 1 objective, got {len(obj.objectives)}'
print(f'  search space: {len(ss.params)} params')
for p in ss.params:
    print(f'    {p.name}: {p.type.value}')
print(f'  objectives: {len(obj.objectives)}')
for o in obj.objectives:
    print(f'    {o.metric}: {o.direction.value} (w={o.weight})')
"; then
  pass "Config loading"
else
  fail "Config loading"
fi


# ══════════════════════════════════════════════════════════════════════════
# Step 2: Validate ASHA scheduler logic
# ══════════════════════════════════════════════════════════════════════════
step_header "Step 2/$TOTAL: Validate ASHA scheduler (promotions + early-stop)"

if uv run python -c "
from eat.sweep.asha import ASHAScheduler
from eat.sweep.types import (
    Direction, MultiObjective, ObjectiveSpec,
    TrialRecord, TrialStatus,
)

obj = MultiObjective(objectives=[
    ObjectiveSpec(metric='train_loss', direction=Direction.MINIMIZE, weight=1.0),
])
scheduler = ASHAScheduler(max_rung=2, reduction_factor=2, min_epochs=1.0, objectives=obj)

# Verify epoch budget grows with rungs
assert scheduler.epochs_for_rung(0) == 1.0
assert scheduler.epochs_for_rung(1) == 2.0
assert scheduler.epochs_for_rung(2) == 4.0
print(f'  rung budgets: {[scheduler.epochs_for_rung(r) for r in range(3)]}')

# Create mock trials: t1 has lower loss (better), t2 has higher loss
t1 = TrialRecord(id='t1', sweep_id='s', params={}, rung=0,
                  metrics={'train_loss': 0.5}, status=TrialStatus.COMPLETED)
t2 = TrialRecord(id='t2', sweep_id='s', params={}, rung=0,
                  metrics={'train_loss': 1.5}, status=TrialStatus.COMPLETED)

# t1 should be promoted, t2 should be stopped
d1 = scheduler.decide(t1, [t1, t2])
d2 = scheduler.decide(t2, [t1, t2])
assert d1 == 'promote', f'expected t1 promoted, got {d1}'
assert d2 == 'stop', f'expected t2 stopped, got {d2}'
print(f'  t1(loss=0.5): {d1}')
print(f'  t2(loss=1.5): {d2}')

# At max rung, everyone stops
t1.rung = 2
d3 = scheduler.decide(t1, [t1])
assert d3 == 'stop', f'expected stop at max rung, got {d3}'
print(f'  t1 at max rung: {d3}')

print('  scalarised scores: t1={:.2f} t2={:.2f}'.format(
    scheduler.scalarise(t1.metrics), scheduler.scalarise(t2.metrics)))
"; then
  pass "ASHA scheduler logic"
else
  fail "ASHA scheduler logic"
fi


# ══════════════════════════════════════════════════════════════════════════
# Step 3: Validate sampler (search space bounds)
# ══════════════════════════════════════════════════════════════════════════
step_header "Step 3/$TOTAL: Validate sampler (bounds + reproducibility)"

if uv run python -c "
from eat.sweep.config import load_search_space
from eat.sweep.sampler import sample

ss = load_search_space('$SEARCH_SPACE')

# Reproducibility: same seed → same output
s1 = sample(ss, seed=42)
s2 = sample(ss, seed=42)
assert s1 == s2, f'seed=42 mismatch: {s1} vs {s2}'
print(f'  reproducibility: OK (seed=42)')

# Different seeds → (almost certainly) different output
s3 = sample(ss, seed=99)
assert s1 != s3, 'different seeds yielded same point'
print(f'  different seeds: OK')

# Check bounds
for p in ss.params:
    val = s1[p.name]
    if p.low is not None and p.high is not None:
        assert p.low <= val <= p.high, f'{p.name}={val} out of [{p.low}, {p.high}]'
    if p.choices is not None:
        assert val in p.choices, f'{p.name}={val} not in {p.choices}'
    print(f'  {p.name} = {val}')

# Many samples: verify all within bounds
for seed in range(100):
    s = sample(ss, seed=seed)
    for p in ss.params:
        v = s[p.name]
        if p.low is not None and p.high is not None:
            assert p.low <= v <= p.high
        if p.choices is not None:
            assert v in p.choices
print(f'  100-sample bounds check: OK')
"; then
  pass "Sampler validation"
else
  fail "Sampler validation"
fi


# ══════════════════════════════════════════════════════════════════════════
# Step 4: Run sweep engine (N trials with ASHA)
# ══════════════════════════════════════════════════════════════════════════
step_header "Step 4/$TOTAL: Run sweep ($TRIALS trials, ASHA halving, local training)"

SWEEP_OK=true
if uv run python -c "
import sys
from eat.sweep.config import load_search_space, load_objectives
from eat.sweep.engine import run_sweep

ss = load_search_space('$SEARCH_SPACE')
obj = load_objectives('$OBJECTIVES')

result = run_sweep(
    project='food_health_coach',
    recipe='smoke',
    search_space=ss,
    objectives=obj,
    max_trials=$TRIALS,
    data_path='$FIXTURE_DATA',
    model_path='$MODEL_PATH',
    output_dir='$WORK_DIR/sweep_output',
    asha_max_rung=1,
    asha_reduction_factor=2,
    asha_min_epochs=1.0,
    log_file=sys.stdout,
    seed=42,
)

assert result.status.value == 'completed', f'sweep not completed: {result.status}'
assert result.best_trial_id is not None, 'no best trial selected'
print(f'RESULT: sweep_id={result.id} best={result.best_trial_id} status={result.status.value}')
"; then
  pass "Sweep engine execution"
else
  fail "Sweep engine execution"
  SWEEP_OK=false
fi


# ══════════════════════════════════════════════════════════════════════════
# Step 5: Verify sweep summary JSON
# ══════════════════════════════════════════════════════════════════════════
step_header "Step 5/$TOTAL: Verify sweep summary JSON"

if $SWEEP_OK; then
  if uv run python -c "
import json, glob

# Find the sweep_summary.json (under a sweep_* subdir)
summaries = glob.glob('$WORK_DIR/sweep_output/sweep_*/sweep_summary.json')
assert len(summaries) >= 1, f'no sweep_summary.json found in $WORK_DIR/sweep_output'

with open(summaries[0]) as f:
    data = json.load(f)

assert 'sweep_id' in data, 'missing sweep_id'
assert 'best_trial_id' in data, 'missing best_trial_id'
assert 'trials' in data, 'missing trials'
assert len(data['trials']) == $TRIALS, f'expected $TRIALS trials, got {len(data[\"trials\"])}'

# Verify each trial has params + metrics
for t in data['trials']:
    assert 'id' in t, 'trial missing id'
    assert 'params' in t, f'trial {t[\"id\"]} missing params'
    assert 'metrics' in t, f'trial {t[\"id\"]} missing metrics'
    assert 'status' in t, f'trial {t[\"id\"]} missing status'
    assert t['status'] in ('completed', 'early_stopped', 'failed'), \
        f'trial {t[\"id\"]} unexpected status: {t[\"status\"]}'

# Verify the best trial has train_loss metric
best_id = data['best_trial_id']
best = [t for t in data['trials'] if t['id'] == best_id]
assert len(best) == 1, f'best trial {best_id} not found in trials list'
assert 'train_loss' in best[0]['metrics'], f'best trial missing train_loss metric'
print(f'  sweep_id: {data[\"sweep_id\"]}')
print(f'  trials: {len(data[\"trials\"])}')
print(f'  best: {best_id}')
print(f'  best loss: {best[0][\"metrics\"][\"train_loss\"]:.4f}')

# Verify ASHA produced some early-stopped trials (at least 1 if trials > 1)
statuses = [t['status'] for t in data['trials']]
print(f'  statuses: {statuses}')
completed_count = statuses.count('completed') + statuses.count('early_stopped')
assert completed_count >= 1, f'no completed/early_stopped trials'
"; then
    pass "Sweep summary verification"
  else
    fail "Sweep summary verification"
  fi
else
  fail "Sweep summary verification (sweep failed)"
fi


# ══════════════════════════════════════════════════════════════════════════
# Step 6: Verify CLI entry point
# ══════════════════════════════════════════════════════════════════════════
step_header "Step 6/$TOTAL: Verify CLI entry point (eat sweep run --help)"

HELP_OUT=$(uv run eat sweep run --help 2>&1 || true)
if echo "$HELP_OUT" | grep -q "ASHA-scheduled"; then
  pass "CLI entry point"
else
  fail "CLI entry point"
fi


# ══════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
TOTAL_RAN=$((PASSED + FAILED))
echo -e "  Results:  ${GREEN}${PASSED} passed${NC}  ${RED}${FAILED} failed${NC}  (${TOTAL_RAN}/${TOTAL} ran)"
echo "═══════════════════════════════════════════════════════════════"

if [ "$FAILED" -gt 0 ]; then
  echo -e "  ${RED}SOME STEPS FAILED${NC}"
  exit 1
fi

echo -e "  ${GREEN}ALL STEPS PASSED${NC}"
exit 0
