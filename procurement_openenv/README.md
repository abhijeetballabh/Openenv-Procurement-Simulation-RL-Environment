# Procurement Vendor Optimization Environment

## Why this project matters

Procurement teams routinely evaluate multiple vendors under hard business constraints, then choose the best option for cost, speed, and quality trade-offs. This repository models that exact decision workflow as an OpenEnv-compatible environment so agent systems can be evaluated on practical, structured, and reproducible business decisions.

## Project at a glance

- Domain: Real-world vendor selection and procurement optimization.
- Environment type: FastAPI + Docker OpenEnv environment.
- Difficulty levels: easy, medium, hard.
- Scoring: deterministic, bounded rewards in [0.0, 1.0].
- Deployment target: Hugging Face Space (Docker SDK).

## Repository structure

- Main environment folder: [procurement_openenv](procurement_openenv)
- OpenEnv manifest: [procurement_openenv/openenv.yaml](procurement_openenv/openenv.yaml)
- Inference baseline: [procurement_openenv/inference.py](procurement_openenv/inference.py)
- Environment API app: [procurement_openenv/server/app.py](procurement_openenv/server/app.py)
- Environment dynamics: [procurement_openenv/server/my_first_openenv_environment.py](procurement_openenv/server/my_first_openenv_environment.py)
- Typed models: [procurement_openenv/models.py](procurement_openenv/models.py)
- Baseline policy: [procurement_openenv/client.py](procurement_openenv/client.py)

## Environment design

The agent receives a set of vendors and operational constraints, then emits structured actions depending on task difficulty.

Observation contains:

- vendors with fields id, price, delivery_days, rating
- constraints with fields max_delivery_days, min_rating
- task_type indicating easy, medium, or hard

Action contains:

- action_type in {filter, select, optimize}
- valid_vendor_ids for easy filtering
- selected_vendor_id for medium and hard selection

## Tasks and grader behavior

- easy: identify all vendors that satisfy constraints
- medium: choose the minimum-price valid vendor
- hard: choose the highest weighted-score vendor

Difficulty progression is deterministic and programmatically graded with explicit criteria per task.

## Reward shaping

- easy: precision/recall-style partial reward
- medium: full reward for optimal valid choice, partial reward for non-optimal but valid choice
- hard: full reward for best weighted choice, proportional reward for near-optimal choice
- all rewards are clamped to [0.0, 1.0]

## Baseline inference

The baseline runner is [procurement_openenv/inference.py](procurement_openenv/inference.py).

It:

- uses OpenAI client with API_BASE_URL, MODEL_NAME, and HF_TOKEN
- emits strict structured logs in START, STEP, END format
- supports deterministic fallback via solve_task when API calls are unavailable

## Local run

```bash
cd procurement_openenv
openenv validate
python inference.py
uvicorn server.app:app --reload
```

## Docker run

```bash
docker build -t procurement-env -f procurement_openenv/server/Dockerfile .
docker run -p 8000:8000 procurement-env
```

## Hugging Face Space

```bash
cd procurement_openenv
openenv push --repo-id <your-username>/procurement-env
```
