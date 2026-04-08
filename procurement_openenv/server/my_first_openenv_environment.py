# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
My First Openenv Environment Implementation.

Procurement simulation environment with vendor filtering, selection, and optimization tasks.
"""

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import MyFirstOpenenvAction, MyFirstOpenenvObservation, Vendor
except ImportError:
    from models import MyFirstOpenenvAction, MyFirstOpenenvObservation, Vendor


class MyFirstOpenenvEnvironment(Environment):
    """
    Procurement environment with three task types:
    - easy: identify vendors that meet constraints
    - medium: choose the lowest-price valid vendor
    - hard: choose the highest weighted-score valid vendor
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the procurement environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._vendors = []
        self._constraints = {}
        self._task_type = "easy"
        self._valid_vendor_ids = []
        self._best_vendor_medium_id = None
        self._best_vendor_hard_id = None
        self._normalized_scores = {
            "price": {},
            "delivery_days": {},
            "rating": {},
        }

    @staticmethod
    def _clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
        return max(min_value, min(max_value, value))

    def _shape_reward(self, raw_reward: float) -> float:
        """Map raw task reward into strict (0, 1) bands with difficulty variation."""
        bounded = self._clamp(raw_reward)
        task_bands = {
            "easy": (0.08, 0.84),
            "medium": (0.05, 0.90),
            "hard": (0.02, 0.96),
        }
        offset, scale = task_bands.get(self._task_type, (0.05, 0.90))
        return offset + scale * bounded

    def meets_constraints(self, vendor) -> bool:
        return (
            vendor.rating >= self._constraints["min_rating"]
            and vendor.delivery_days <= self._constraints["max_delivery_days"]
        )

    @staticmethod
    def normalize_scores(values: list[float], higher_is_better: bool) -> list[float]:
        if not values:
            return []
        min_value = min(values)
        max_value = max(values)
        if min_value == max_value:
            return [1.0 for _ in values]

        normalized = []
        for value in values:
            raw = (value - min_value) / (max_value - min_value)
            normalized.append(raw if higher_is_better else 1.0 - raw)
        return normalized

    def compute_score(self, vendor) -> float:
        price_score = self._normalized_scores["price"].get(vendor.id, 0.0)
        delivery_score = self._normalized_scores["delivery_days"].get(vendor.id, 0.0)
        rating_score = self._normalized_scores["rating"].get(vendor.id, 0.0)
        return 0.5 * price_score + 0.3 * delivery_score + 0.2 * rating_score

    def _build_state(self) -> State:
        return State(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
            vendors=[vendor.model_dump() for vendor in self._vendors],
            constraints=self._constraints,
            task_type=self._task_type,
            valid_vendor_ids=self._valid_vendor_ids,
            best_vendor_medium_id=self._best_vendor_medium_id,
            best_vendor_hard_id=self._best_vendor_hard_id,
        )

    def _compute_ground_truth(self) -> None:
        valid_vendors = [vendor for vendor in self._vendors if self.meets_constraints(vendor)]
        self._valid_vendor_ids = [vendor.id for vendor in valid_vendors]

        self._best_vendor_medium_id = None
        if valid_vendors:
            self._best_vendor_medium_id = min(valid_vendors, key=lambda v: v.price).id

        prices = [vendor.price for vendor in self._vendors]
        deliveries = [float(vendor.delivery_days) for vendor in self._vendors]
        ratings = [vendor.rating for vendor in self._vendors]

        normalized_price = self.normalize_scores(prices, higher_is_better=False)
        normalized_delivery = self.normalize_scores(deliveries, higher_is_better=False)
        normalized_rating = self.normalize_scores(ratings, higher_is_better=True)

        self._normalized_scores["price"] = {
            vendor.id: score for vendor, score in zip(self._vendors, normalized_price)
        }
        self._normalized_scores["delivery_days"] = {
            vendor.id: score for vendor, score in zip(self._vendors, normalized_delivery)
        }
        self._normalized_scores["rating"] = {
            vendor.id: score for vendor, score in zip(self._vendors, normalized_rating)
        }

        hard_candidates = valid_vendors if valid_vendors else self._vendors
        self._best_vendor_hard_id = None
        if hard_candidates:
            self._best_vendor_hard_id = max(hard_candidates, key=self.compute_score).id

    def reset(self, task_type: str | None = None) -> MyFirstOpenenvObservation:
        """
        Reset the environment.

        Returns:
            MyFirstOpenenvObservation with a new procurement scenario
        """
        self._task_type = task_type if task_type in {"easy", "medium", "hard"} else random.choice(["easy", "medium", "hard"])

        vendor_count = random.randint(3, 6)
        self._vendors = []
        for vendor_id in range(1, vendor_count + 1):
            self._vendors.append(
                Vendor(
                    id=vendor_id,
                    price=round(random.uniform(50.0, 200.0), 2),
                    delivery_days=random.randint(1, 10),
                    rating=round(random.uniform(3.0, 5.0), 2),
                )
            )

        self._constraints = {
            "max_delivery_days": random.randint(3, 8),
            "min_rating": round(random.uniform(3.5, 4.5), 2),
        }

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._compute_ground_truth()
        self._state = self._build_state()

        return MyFirstOpenenvObservation(
            vendors=self._vendors,
            constraints=self._constraints,
            task_type=self._task_type,
            done=False,
            reward=0.0,
            metadata={
                "valid_vendor_ids": self._valid_vendor_ids,
                "best_vendor_medium_id": self._best_vendor_medium_id,
                "best_vendor_hard_id": self._best_vendor_hard_id,
            },
        )

    def step(self, action: MyFirstOpenenvAction) -> MyFirstOpenenvObservation:  # type: ignore[override]
        """
        Execute a single decision step for the active procurement task.

        Args:
            action: MyFirstOpenenvAction containing structured decision inputs

        Returns:
            MyFirstOpenenvObservation with task reward and current scenario
        """
        self._state.step_count += 1

        reward = 0.0

        if self._task_type == "easy":
            predicted_ids = set(action.valid_vendor_ids or [])
            true_ids = set(self._valid_vendor_ids)

            true_positives = len(predicted_ids & true_ids)
            precision = (
                true_positives / len(predicted_ids)
                if predicted_ids
                else (1.0 if not true_ids else 0.0)
            )
            recall = (
                true_positives / len(true_ids)
                if true_ids
                else (1.0 if not predicted_ids else 0.0)
            )
            reward = (precision + recall) / 2.0

        elif self._task_type == "medium":
            selected_id = action.selected_vendor_id
            if selected_id is not None and selected_id == self._best_vendor_medium_id:
                reward = 1.0
            elif selected_id is not None and selected_id in self._valid_vendor_ids:
                reward = 0.5
            else:
                reward = 0.0

        elif self._task_type == "hard":
            selected_id = action.selected_vendor_id
            vendors_by_id = {vendor.id: vendor for vendor in self._vendors}

            selected_vendor = vendors_by_id.get(selected_id)
            selected_score = self.compute_score(selected_vendor) if selected_vendor else 0.0

            best_vendor = vendors_by_id.get(self._best_vendor_hard_id)
            best_score = self.compute_score(best_vendor) if best_vendor else 0.0

            if selected_id is not None and selected_id == self._best_vendor_hard_id:
                reward = 1.0
            elif best_score > 0.0:
                reward = selected_score / best_score
            else:
                reward = 0.0

        reward = self._shape_reward(reward)
        self._state = self._build_state()

        score_snapshot = {
            vendor.id: {
                "price_score": self._normalized_scores["price"].get(vendor.id, 0.0),
                "delivery_score": self._normalized_scores["delivery_days"].get(vendor.id, 0.0),
                "rating_score": self._normalized_scores["rating"].get(vendor.id, 0.0),
                "weighted_score": self.compute_score(vendor),
            }
            for vendor in self._vendors
        }

        return MyFirstOpenenvObservation(
            vendors=self._vendors,
            constraints=self._constraints,
            task_type=self._task_type,
            done=True,
            reward=reward,
            metadata={
                "step": self._state.step_count,
                "action_type": action.action_type,
                "provided_valid_vendor_ids": action.valid_vendor_ids,
                "provided_selected_vendor_id": action.selected_vendor_id,
                "valid_vendor_ids": self._valid_vendor_ids,
                "best_vendor_medium_id": self._best_vendor_medium_id,
                "best_vendor_hard_id": self._best_vendor_hard_id,
                "score_snapshot": score_snapshot,
                "provided_score_breakdown": action.score_breakdown,
            },
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with procurement scenario data
        """
        return self._build_state()
