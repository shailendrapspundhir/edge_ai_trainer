#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# E2E Training Pipeline Test for Edge AI Trainer
#
# Uses the local Qwen2.5-0.5B-Instruct model from llm_foundation_models/,
# LoRA-fine-tunes it on a committed fixture dataset, and verifies every
# pipeline stage: data → train → checkpoints → adapter → inference.
#
# No network requests to HuggingFace — fully offline.
#
# Usage:
#   bash tests/e2e/test_training_pipeline.sh              # full (3 epochs)
#   bash tests/e2e/test_training_pipeline.sh --quick       # fast (1 epoch)
#   bash tests/e2e/test_training_pipeline.sh --no-cleanup  # keep work dir
#   bash tests/e2e/test_training_pipeline.sh --epochs 5    # custom epochs
#
# Requirements:
#   uv sync --extra train       (torch, transformers, peft, trl, bitsandbytes)
#   llm_foundation_models/Qwen2.5-0.5B-Instruct/  (local model copy)
#   ~2 GB disk                  (training artefacts)
#   GPU recommended             (CPU works but takes 15-30 min)
# ═══════════════════════════════════════════════════════════════════════════
set -uo pipefail

# ──── Resolve paths ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

STEPS_PY="tests/e2e/pipeline_steps.py"
FIXTURE_DATA="tests/e2e/fixtures/train_data.jsonl"
MODEL_PATH="$REPO_ROOT/llm_foundation_models/Qwen2.5-0.5B-Instruct"

# Block all HuggingFace network requests — tests must run fully offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ──── Defaults ─────────────────────────────────────────────────────────────
EPOCHS=3
SAVE_STEPS=10
MIN_CHECKPOINTS=2
MIN_KEYWORD_HITS=1
CLEANUP=true

# ──── Parse args ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      EPOCHS=1; SAVE_STEPS=5; MIN_CHECKPOINTS=1; MIN_KEYWORD_HITS=0
      shift ;;
    --no-cleanup)
      CLEANUP=false; shift ;;
    --epochs)
      EPOCHS="$2"; shift 2 ;;
    --save-steps)
      SAVE_STEPS="$2"; shift 2 ;;
    --help|-h)
      head -18 "${BASH_SOURCE[0]}" | tail -n +2
      exit 0 ;;
    *)
      echo "Unknown arg: $1"; exit 1 ;;
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
SKIPPED=0
TOTAL=7

step_header() {
  echo ""
  echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${BOLD}  $1${NC}"
  echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

pass() { echo -e "  ${GREEN}✓ PASS${NC}: $1"; PASSED=$((PASSED + 1)); }
fail() { echo -e "  ${RED}✗ FAIL${NC}: $1"; FAILED=$((FAILED + 1)); }
skip() { echo -e "  ${YELLOW}⊘ SKIP${NC}: $1"; SKIPPED=$((SKIPPED + 1)); }

# ──── Work directory ───────────────────────────────────────────────────────
WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/eat-e2e-XXXXXX")
cleanup() {
  if $CLEANUP; then
    rm -rf "$WORK_DIR"
  else
    echo -e "\n${YELLOW}Work dir preserved: $WORK_DIR${NC}"
  fi
}
trap cleanup EXIT

OUTPUT_DIR="$WORK_DIR/training_output"
ADAPTER_DIR="$OUTPUT_DIR/final_adapter"

# ──── Banner ───────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Edge AI Trainer — E2E Training Pipeline Test"
echo "═══════════════════════════════════════════════════════════════"
echo "  Model:          $MODEL_PATH"
echo "  Offline:        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
echo "  Epochs:         $EPOCHS"
echo "  Save interval:  every $SAVE_STEPS optimizer steps"
echo "  Work dir:       $WORK_DIR"
echo "  Fixture data:   $FIXTURE_DATA"
echo ""

# ══════════════════════════════════════════════════════════════════════════
# Step 1: Validate fixture data
# ══════════════════════════════════════════════════════════════════════════
step_header "Step 1/$TOTAL: Validate fixture training data"

if uv run python "$STEPS_PY" validate-data \
    --data-path "$FIXTURE_DATA" \
    --min-records 20; then
  pass "Data validation"
else
  fail "Data validation"
  echo -e "${RED}FATAL: fixture data is invalid — cannot continue.${NC}"
  exit 1
fi

# ══════════════════════════════════════════════════════════════════════════
# Step 2: Check prerequisites
# ══════════════════════════════════════════════════════════════════════════
step_header "Step 2/$TOTAL: Check prerequisites (torch, peft, trl, transformers)"

if uv run python "$STEPS_PY" check-prereqs; then
  pass "Prerequisites"
else
  fail "Prerequisites"
  echo -e "${RED}FATAL: missing packages — run: uv sync --extra train${NC}"
  exit 1
fi

# ══════════════════════════════════════════════════════════════════════════
# Step 3: Verify local model
# ══════════════════════════════════════════════════════════════════════════
step_header "Step 3/$TOTAL: Verify local model ($MODEL_PATH)"

if uv run python "$STEPS_PY" verify-model \
    --model-path "$MODEL_PATH"; then
  pass "Local model verification"
else
  fail "Local model verification"
  echo -e "${RED}FATAL: local model not found — copy model to llm_foundation_models/.${NC}"
  exit 1
fi

# ══════════════════════════════════════════════════════════════════════════
# Step 4: Train with LoRA + checkpointing
# ══════════════════════════════════════════════════════════════════════════
step_header "Step 4/$TOTAL: Train ($EPOCHS epoch(s), checkpoint every $SAVE_STEPS steps)"

TRAIN_OK=true
if uv run python "$STEPS_PY" train \
    --data-path "$FIXTURE_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --model-path "$MODEL_PATH" \
    --epochs "$EPOCHS" \
    --save-steps "$SAVE_STEPS"; then
  pass "Training"
else
  fail "Training"
  TRAIN_OK=false
fi

# ══════════════════════════════════════════════════════════════════════════
# Step 5: Verify multiple checkpoints
# ══════════════════════════════════════════════════════════════════════════
step_header "Step 5/$TOTAL: Verify multiple checkpoints saved"

if $TRAIN_OK; then
  if uv run python "$STEPS_PY" verify-checkpoints \
      --output-dir "$OUTPUT_DIR" \
      --min-count "$MIN_CHECKPOINTS"; then
    pass "Checkpoint verification"
  else
    fail "Checkpoint verification"
  fi
else
  skip "Checkpoint verification (training failed)"
fi

# ══════════════════════════════════════════════════════════════════════════
# Step 6: Verify adapter artefacts
# ══════════════════════════════════════════════════════════════════════════
step_header "Step 6/$TOTAL: Verify adapter artefacts"

if $TRAIN_OK; then
  if uv run python "$STEPS_PY" verify-adapter \
      --adapter-dir "$ADAPTER_DIR"; then
    pass "Adapter verification"
  else
    fail "Adapter verification"
  fi
else
  skip "Adapter verification (training failed)"
fi

# ══════════════════════════════════════════════════════════════════════════
# Step 7: Verify inference — base vs fine-tuned
# ══════════════════════════════════════════════════════════════════════════
step_header "Step 7/$TOTAL: Verify inference (base vs fine-tuned)"

if $TRAIN_OK; then
  if uv run python "$STEPS_PY" verify-inference \
      --adapter-dir "$ADAPTER_DIR" \
      --model-path "$MODEL_PATH" \
      --min-hits "$MIN_KEYWORD_HITS"; then
    pass "Inference verification"
  else
    fail "Inference verification"
  fi
else
  skip "Inference verification (training failed)"
fi

# ══════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
TOTAL_RAN=$((PASSED + FAILED))
echo -e "  Results:  ${GREEN}${PASSED} passed${NC}  ${RED}${FAILED} failed${NC}  ${YELLOW}${SKIPPED} skipped${NC}  (${TOTAL_RAN}/${TOTAL} ran)"
echo "═══════════════════════════════════════════════════════════════"

if [ "$FAILED" -gt 0 ]; then
  echo -e "  ${RED}SOME STEPS FAILED${NC}"
  exit 1
fi

echo -e "  ${GREEN}ALL STEPS PASSED${NC}"
exit 0