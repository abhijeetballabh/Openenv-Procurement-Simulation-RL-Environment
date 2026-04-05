# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the procurement vendor selection environment."""

from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query

try:
    from ..models import MyFirstOpenenvAction, MyFirstOpenenvObservation
    from .my_first_openenv_environment import MyFirstOpenenvEnvironment
except (ModuleNotFoundError, ImportError):
    from models import MyFirstOpenenvAction, MyFirstOpenenvObservation
    from server.my_first_openenv_environment import MyFirstOpenenvEnvironment


def _observation_payload(observation: MyFirstOpenenvObservation) -> dict[str, Any]:
    return {
        "vendors": [vendor.model_dump() for vendor in observation.vendors],
        "constraints": observation.constraints,
        "task_type": observation.task_type,
        "done": observation.done,
        "reward": observation.reward,
    }


def _state_payload(state_obj: Any) -> dict[str, Any]:
    return {
        "episode_id": getattr(state_obj, "episode_id", None),
        "step_count": getattr(state_obj, "step_count", 0),
        "vendors": getattr(state_obj, "vendors", []),
        "constraints": getattr(state_obj, "constraints", {}),
        "task_type": getattr(state_obj, "task_type", None),
    }


def _current_vendor_ids(env: MyFirstOpenenvEnvironment) -> set[int]:
    return {vendor.id for vendor in getattr(env, "_vendors", [])}


app = FastAPI(title="Procurement Vendor Selection Environment", version="1.0.0")
env = MyFirstOpenenvEnvironment()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset(
    task_type: Literal["easy", "medium", "hard"] | None = Query(default=None),
) -> dict[str, Any]:
    observation = env.reset(task_type)
    return _observation_payload(observation)


@app.post("/step")
def step(action: MyFirstOpenenvAction) -> dict[str, Any]:
    if action.action_type not in {"filter", "select", "optimize"}:
        raise HTTPException(status_code=422, detail="action_type must be one of: filter, select, optimize")

    current_ids = _current_vendor_ids(env)
    if action.selected_vendor_id is not None and action.selected_vendor_id not in current_ids:
        current_state = env.state
        current_observation = MyFirstOpenenvObservation(
            vendors=getattr(env, "_vendors", []),
            constraints=getattr(env, "_constraints", {}),
            task_type=getattr(env, "_task_type", "easy"),
            done=True,
            reward=0.0,
        )
        return {
            "observation": _observation_payload(current_observation),
            "reward": 0.0,
            "done": True,
            "info": {
                "error": "invalid_vendor_id",
                "step_count": getattr(current_state, "step_count", 0),
            },
        }

    try:
        result = env.step(action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"step failed: {exc}") from exc

    return {
        "observation": _observation_payload(result),
        "reward": float(result.reward or 0.0),
        "done": bool(result.done),
        "info": {
            "step_count": env.state.step_count,
            "task_type": result.task_type,
        },
    }


@app.get("/state")
def state() -> dict[str, Any]:
    return _state_payload(env.state)


@app.get("/schema")
def schema() -> dict[str, Any]:
    return {
        "action": MyFirstOpenenvAction.model_json_schema(),
        "observation": MyFirstOpenenvObservation.model_json_schema(),
    }


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for running the FastAPI server directly."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
