# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the My First Openenv Environment.

The my_first_openenv environment simulates procurement vendor selection tasks.
"""

from typing import Any

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, ConfigDict, Field


class Vendor(BaseModel):
    """Vendor candidate used in procurement simulations."""

    id: int = Field(..., description="Vendor identifier")
    price: float = Field(..., description="Quoted price")
    delivery_days: int = Field(..., description="Delivery time in days")
    rating: float = Field(..., description="Quality rating")


class MyFirstOpenenvAction(Action):
    """Action payload for procurement tasks."""

    model_config = ConfigDict(extra="forbid")

    action_type: str = Field(..., pattern="^(filter|select|optimize)$", description="One of: filter, select, optimize")
    valid_vendor_ids: list[int] | None = Field(
        default=None,
        description="Predicted valid vendor IDs for filter tasks",
    )
    selected_vendor_id: int | None = Field(
        default=None,
        description="Selected vendor ID for selection/optimization tasks",
    )
    score_breakdown: dict[str, Any] | None = Field(
        default=None,
        description="Optional structured scoring details",
    )


class MyFirstOpenenvObservation(Observation):
    """Observation payload for procurement episodes."""

    vendors: list[Vendor] = Field(default_factory=list, description="Available vendors")
    constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Task constraints",
    )
    task_type: str = Field(default="easy", description="Task difficulty type")


class MyFirstOpenenvReward(BaseModel):
    """Typed reward payload for evaluation and logging semantics."""

    value: float = Field(..., ge=0.0, le=1.0, description="Reward value in [0, 1]")
    reason: str = Field(default="", description="Optional reward rationale")
