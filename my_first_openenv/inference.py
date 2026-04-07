"""OpenEnv inference runner for procurement tasks with strict structured logs."""

from __future__ import annotations

import json
import os
import random
from typing import Optional

from openai import OpenAI

from server.my_first_openenv_environment import MyFirstOpenenvEnvironment
from models import MyFirstOpenenvAction
from client import solve_task

ENV_NAME = "procurement"
TASKS = ("easy", "medium", "hard")
MAX_STEPS = 8
SUCCESS_SCORE_THRESHOLD = 0.5
RANDOM_SEED = 42

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = (
    "You are an expert procurement assistant. Return only valid JSON with keys "
    "action_type, valid_vendor_ids, selected_vendor_id. "
    "Use action_type=filter for easy, action_type=select for medium, "
    "action_type=optimize for hard."
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def _action_to_string(action: MyFirstOpenenvAction) -> str:
    if action.action_type == "filter":
        return f"valid_vendor_ids={action.valid_vendor_ids or []}"
    return f"selected_vendor_id={action.selected_vendor_id}"


def _clamp_score(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _build_user_prompt(observation) -> str:
    payload = {
        "task_type": observation.task_type,
        "constraints": observation.constraints,
        "vendors": [vendor.model_dump() for vendor in observation.vendors],
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    return text[start : end + 1]


def _llm_action_or_fallback(client: OpenAI, observation) -> MyFirstOpenenvAction:
    if not API_KEY:
        return solve_task(observation)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(observation)},
            ],
            temperature=0.0,
            max_tokens=220,
        )
        content = (response.choices[0].message.content or "").strip()
        action_payload = json.loads(_extract_json(content))
        return MyFirstOpenenvAction(**action_payload)
    except Exception:
        return solve_task(observation)


def run_task(env: MyFirstOpenenvEnvironment, task_type: str) -> None:
    rewards: list[float] = []
    steps_taken = 0

    log_start(task=task_type, env=ENV_NAME, model=MODEL_NAME)

    observation = env.reset(task_type)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "placeholder")

    for step in range(1, MAX_STEPS + 1):
        action = _llm_action_or_fallback(client, observation)
        result = env.step(action)

        reward = float(result.reward or 0.0)
        done = bool(result.done)

        rewards.append(reward)
        steps_taken = step

        log_step(
            step=step,
            action=_action_to_string(action),
            reward=reward,
            done=done,
            error=None,
        )

        observation = result
        if done:
            break

    average_reward = sum(rewards) / len(rewards) if rewards else 0.0
    score = _clamp_score(average_reward)
    success = score >= SUCCESS_SCORE_THRESHOLD
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    random.seed(RANDOM_SEED)
    env = MyFirstOpenenvEnvironment()
    for task_type in TASKS:
        run_task(env, task_type)


if __name__ == "__main__":
    main()