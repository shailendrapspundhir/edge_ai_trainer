.PHONY: help install install-train install-all sync fmt lint test test-e2e test-e2e-quick clean \
        redis orchestrator worker-gpu worker-cpu worker-android worker-cloud \
        dashboard smoke train-demo collect-data project-init

PY ?= python
UV ?= uv

help:
	@echo "Edge AI Trainer — common commands"
	@echo ""
	@echo "  make install         # base deps via uv"
	@echo "  make install-train   # adds torch + transformers + peft + trl + bnb"
	@echo "  make install-all     # everything (train + export + cloud + android + dev)"
	@echo ""
	@echo "  make redis           # start redis via docker compose"
	@echo "  make orchestrator    # FastAPI control plane (foreground)"
	@echo "  make worker-gpu      # RQ worker, gpu queue"
	@echo "  make worker-cpu      # RQ worker, cpu queue"
	@echo "  make worker-android  # RQ worker, android queue"
	@echo "  make worker-cloud    # RQ worker, cloud queue"
	@echo "  make dashboard       # streamlit dashboard"
	@echo ""
	@echo "  make smoke           # end-to-end smoke run (fake training, no GPU needed)"
	@echo "  make train-demo      # standalone: generate data → LoRA train → compare base vs fine-tuned"
	@echo "  make collect-data    # generate all 10 food data categories (10 examples each)"
	@echo "  make project-init NAME=my_proj"
	@echo ""
	@echo "  make fmt | lint | test | clean"

install:
	$(UV) sync

install-train:
	$(UV) sync --extra train

install-all:
	$(UV) sync --extra train --extra export --extra cloud --extra android --extra dev --extra data-gen

sync: install

fmt:
	$(UV) run ruff format src tests

lint:
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

test:
	$(UV) run pytest

test-e2e:
	bash tests/e2e/test_training_pipeline.sh

test-e2e-quick:
	bash tests/e2e/test_training_pipeline.sh --quick

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

redis:
	docker compose up -d redis

orchestrator:
	$(UV) run eat-orchestrator

worker-gpu:
	EAT_WORKER_QUEUES=$$EAT_QUEUE_GPU $(UV) run eat-worker

worker-cpu:
	EAT_WORKER_QUEUES=$$EAT_QUEUE_CPU $(UV) run eat-worker

worker-android:
	EAT_WORKER_QUEUES=$$EAT_QUEUE_ANDROID $(UV) run eat-worker

worker-cloud:
	EAT_WORKER_QUEUES=$$EAT_QUEUE_CLOUD $(UV) run eat-worker

dashboard:
	$(UV) run eat-dashboard

smoke:
	$(UV) run eat run --project food_health_coach --recipe smoke --dry-run

train-demo:
	$(UV) run scripts/train_and_verify.py --epochs 8

train-demo-strong:
	$(UV) run scripts/train_and_verify.py --epochs 15

collect-data:
	$(UV) run scripts/collect_food_data.py

collect-data-hf:
	$(UV) run scripts/collect_food_data.py --fetch-hf

project-init:
	@test -n "$(NAME)" || (echo "usage: make project-init NAME=my_proj" && exit 1)
	$(UV) run eat project init $(NAME)
