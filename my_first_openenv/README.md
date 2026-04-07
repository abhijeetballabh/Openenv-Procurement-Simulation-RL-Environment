---
title: Procurement Environment
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
  - procurement
---

# Procurement Environment

## Overview

This environment simulates a procurement decision workflow where an agent must choose the best vendor under business constraints. Each episode provides a set of vendor options and rule-based constraints, and the agent must return a structured decision that matches the task difficulty.

The setup is deterministic at the policy level by using the baseline `solve_task()` decision function, with no external LLM or API dependency.

## Observation Space

Each observation contains:

- `vendors`: a list of vendor candidates.
  - `price`: quoted cost.
  - `delivery_days`: delivery time.
  - `rating`: quality score.
- `constraints`: rule set for acceptable vendors.
  - `max_delivery_days`
  - `min_rating`

## Action Space

- `easy` -> `valid_vendor_ids`
- `medium` -> `selected_vendor_id`
- `hard` -> `selected_vendor_id`

## Tasks

- `easy`: filtering task. Identify all vendors that satisfy constraints.
- `medium`: selection task. Choose the best vendor among valid vendors.
- `hard`: optimization task. Choose the vendor with the best weighted score.

## Reward Design

Rewards are in the range `[0, 1]` and reflect decision quality:

- Partial rewards are used when decisions are partially correct.
- Constraint satisfaction is explicitly rewarded for valid filtering/selection.
- Optimality scoring is used for harder decisions, with full reward for the best choice and fractional reward for near-optimal choices.

## How to Run

Local:

```bash
uvicorn server.app:app --reload
```

Docker:

```bash
docker build -t procurement-env -f my_first_openenv/server/Dockerfile .
docker run -p 8000:8000 procurement-env
```

## Baseline Inference

The submission baseline script is [inference.py](inference.py). It emits strict structured logs:

- `[START] task=<task> env=<benchmark> model=<model_name>`
- `[STEP] step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>`

Set environment variables before running:

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=<your_api_key>
python inference.py
```

Windows PowerShell:

```powershell
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
$env:HF_TOKEN="<your_api_key>"
python inference.py
```

If no API key is provided, the script safely falls back to the deterministic `solve_task()` baseline so execution still completes.

## Baseline Scores

Example reproducible baseline run (seed = 42, deterministic fallback policy):

- easy: 1.00
- medium: 1.00
- hard: 1.00

All scores are in `[0.0, 1.0]` and are produced by the same `inference.py` script used for submission.

## Notes

- Deterministic baseline policy is implemented in `client.py` via `solve_task()`.
- Inference script supports OpenAI-compatible API routing via `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`.
- Results are reproducible under fixed random seed and deterministic fallback action selection.
