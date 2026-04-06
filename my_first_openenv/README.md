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

## Notes

- Deterministic baseline policy is implemented in `client.py` via `solve_task()`.
- No external API dependency is required for inference.
- Results are reproducible under fixed random seed and deterministic action selection.
