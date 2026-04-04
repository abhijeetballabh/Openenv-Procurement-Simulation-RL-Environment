# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""My First Openenv Environment Client and baseline procurement agent."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import MyFirstOpenenvAction, MyFirstOpenenvObservation, Vendor


class MyFirstOpenenvEnv(
    EnvClient[MyFirstOpenenvAction, MyFirstOpenenvObservation, State]
):
    """
    Client for the My First Openenv procurement environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with MyFirstOpenenvEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     action = solve_task(result.observation)
        ...     result = client.step(action)
        ...     print(result.reward)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = MyFirstOpenenvEnv.from_docker_image("my_first_openenv-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(solve_task(result.observation))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: MyFirstOpenenvAction) -> Dict:
        """
        Convert MyFirstOpenenvAction to JSON payload for step message.

        Args:
            action: MyFirstOpenenvAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[MyFirstOpenenvObservation]:
        """
        Parse server response into StepResult[MyFirstOpenenvObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with MyFirstOpenenvObservation
        """
        obs_data = payload.get("observation", {})

        vendors_data = obs_data.get("vendors", [])
        vendors = [Vendor(**vendor) for vendor in vendors_data]

        observation = MyFirstOpenenvObservation(
            vendors=vendors,
            constraints=obs_data.get("constraints", {}),
            task_type=obs_data.get("task_type", "easy"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id, step_count, and extra procurement fields
        """
        state_data: Dict[str, Any] = {
            "vendors": payload.get("vendors", []),
            "constraints": payload.get("constraints", {}),
            "task_type": payload.get("task_type"),
            "valid_vendor_ids": payload.get("valid_vendor_ids", []),
            "best_vendor_medium_id": payload.get("best_vendor_medium_id"),
            "best_vendor_hard_id": payload.get("best_vendor_hard_id"),
        }
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            **{k: v for k, v in state_data.items() if v is not None},
        )


def _meets_constraints(vendor: Vendor, constraints: dict[str, Any]) -> bool:
    max_delivery_days = constraints.get("max_delivery_days")
    min_rating = constraints.get("min_rating")

    if max_delivery_days is None or min_rating is None:
        return True

    return vendor.delivery_days <= max_delivery_days and vendor.rating >= min_rating


def _normalize(values: list[float], higher_is_better: bool) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if low == high:
        return [1.0 for _ in values]

    normalized: list[float] = []
    for value in values:
        scaled = (value - low) / (high - low)
        normalized.append(scaled if higher_is_better else 1.0 - scaled)
    return normalized


def _tightness_max_constraint(limit: float | None, values: list[float]) -> float:
    if limit is None or not values:
        return 0.0
    low = min(values)
    high = max(values)
    if low == high:
        return 1.0 if limit <= low else 0.0
    if limit <= low:
        return 1.0
    if limit >= high:
        return 0.0
    relaxed_fraction = (limit - low) / (high - low)
    return 1.0 - relaxed_fraction


def _tightness_min_constraint(limit: float | None, values: list[float]) -> float:
    if limit is None or not values:
        return 0.0
    low = min(values)
    high = max(values)
    if low == high:
        return 1.0 if limit >= high else 0.0
    if limit >= high:
        return 1.0
    if limit <= low:
        return 0.0
    return (limit - low) / (high - low)


def _constraint_aware_weights(
    all_vendors: list[Vendor],
    constraints: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    base = {"price": 0.5, "delivery": 0.3, "rating": 0.2}
    prices = [vendor.price for vendor in all_vendors]
    deliveries = [float(vendor.delivery_days) for vendor in all_vendors]
    ratings = [vendor.rating for vendor in all_vendors]

    price_tightness = _tightness_max_constraint(constraints.get("max_price"), prices)
    delivery_tightness = _tightness_max_constraint(
        constraints.get("max_delivery_days"), deliveries
    )
    rating_tightness = _tightness_min_constraint(constraints.get("min_rating"), ratings)

    adjusted = {
        "price": base["price"] * (1.0 + 0.02 * price_tightness),
        "delivery": base["delivery"] * (1.0 + 0.02 * delivery_tightness),
        "rating": base["rating"] * (1.0 + 0.02 * rating_tightness),
    }

    total = adjusted["price"] + adjusted["delivery"] + adjusted["rating"]
    weights = {
        "price": adjusted["price"] / total,
        "delivery": adjusted["delivery"] / total,
        "rating": adjusted["rating"] / total,
    }
    tightness = {
        "price": price_tightness,
        "delivery": delivery_tightness,
        "rating": rating_tightness,
    }
    return weights, tightness


def _score_vendors(
    vendors: list[Vendor],
    weights: dict[str, float],
) -> dict[int, dict[str, float]]:
    prices = [vendor.price for vendor in vendors]
    deliveries = [float(vendor.delivery_days) for vendor in vendors]
    ratings = [vendor.rating for vendor in vendors]

    price_scores = _normalize(prices, higher_is_better=False)
    delivery_scores = _normalize(deliveries, higher_is_better=False)
    rating_scores = _normalize(ratings, higher_is_better=True)

    breakdown: dict[int, dict[str, float]] = {}
    for idx, vendor in enumerate(vendors):
        price_component = weights["price"] * price_scores[idx]
        delivery_component = weights["delivery"] * delivery_scores[idx]
        rating_component = weights["rating"] * rating_scores[idx]
        weighted = (
            price_component
            + delivery_component
            + rating_component
        )
        breakdown[vendor.id] = {
            "raw_price": vendor.price,
            "raw_delivery_days": float(vendor.delivery_days),
            "raw_rating": vendor.rating,
            "price_score": price_scores[idx],
            "delivery_score": delivery_scores[idx],
            "rating_score": rating_scores[idx],
            "price_component": price_component,
            "delivery_component": delivery_component,
            "rating_component": rating_component,
            "weighted_score": weighted,
        }
    return breakdown


def solve_task(observation: MyFirstOpenenvObservation) -> MyFirstOpenenvAction:
    """Return a structured action for easy, medium, and hard procurement tasks."""
    vendors = observation.vendors
    constraints = observation.constraints or {}
    task_type = observation.task_type

    valid_vendors = [vendor for vendor in vendors if _meets_constraints(vendor, constraints)]
    valid_ids = [vendor.id for vendor in valid_vendors]

    if task_type == "easy":
        return MyFirstOpenenvAction(
            action_type="filter",
            valid_vendor_ids=valid_ids,
        )

    if task_type == "medium":
        if not valid_vendors:
            return MyFirstOpenenvAction(action_type="select", selected_vendor_id=None)
        selected = min(valid_vendors, key=lambda vendor: vendor.price)
        return MyFirstOpenenvAction(action_type="select", selected_vendor_id=selected.id)

    weights, tightness = _constraint_aware_weights(vendors, constraints)

    normalization_scope = "valid_vendors"
    scoring_pool = valid_vendors
    if not valid_vendors:
        normalization_scope = "all_vendors_fallback"
        scoring_pool = vendors

    score_table = _score_vendors(scoring_pool, weights) if scoring_pool else {}

    selected_vendor_id = None
    if len(scoring_pool) == 1:
        selected_vendor_id = scoring_pool[0].id
    elif scoring_pool:
        selected_vendor_id = max(
            scoring_pool,
            key=lambda vendor: score_table[vendor.id]["weighted_score"],
        ).id

    detailed_breakdown: dict[str, Any] = {
        "weights": {
            "price": weights["price"],
            "delivery": weights["delivery"],
            "rating": weights["rating"],
        },
        "constraint_tightness": tightness,
        "normalization_scope": normalization_scope,
        "valid_vendor_ids": valid_ids,
        "selected_vendor_id": selected_vendor_id,
        "vendors": {str(vendor_id): stats for vendor_id, stats in score_table.items()},
    }
    if not valid_vendors:
        detailed_breakdown["edge_case"] = "no_valid_vendors"
    elif len(valid_vendors) == 1:
        detailed_breakdown["edge_case"] = "single_valid_vendor"

    return MyFirstOpenenvAction(
        action_type="optimize",
        selected_vendor_id=selected_vendor_id,
        score_breakdown=detailed_breakdown,
    )


def run_local_evaluation(num_episodes: int = 100) -> float:
    """Run local episodes against the environment implementation and print average reward."""
    from .server.my_first_openenv_environment import MyFirstOpenenvEnvironment

    env = MyFirstOpenenvEnvironment()
    total_reward = 0.0

    for _ in range(num_episodes):
        observation = env.reset()
        action = solve_task(observation)
        result = env.step(action)
        total_reward += float(result.reward or 0.0)

    average_reward = total_reward / num_episodes if num_episodes > 0 else 0.0
    print(f"Average reward over {num_episodes} episodes: {average_reward:.4f}")
    return average_reward


def run_hard_mode_evaluation(num_episodes: int = 1000) -> float:
    """Run hard-only episodes and print average reward for the optimized hard policy."""
    from .server.my_first_openenv_environment import MyFirstOpenenvEnvironment

    env = MyFirstOpenenvEnvironment()
    total_reward = 0.0

    for _ in range(num_episodes):
        observation = env.reset("hard")
        action = solve_task(observation)
        result = env.step(action)
        total_reward += float(result.reward or 0.0)

    average_reward = total_reward / num_episodes if num_episodes > 0 else 0.0
    print(f"Hard mode average reward over {num_episodes} episodes: {average_reward:.4f}")
    return average_reward
