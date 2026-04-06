from __future__ import annotations

import json
from typing import Optional

import gradio as gr

from server.my_first_openenv_environment import MyFirstOpenenvEnvironment
from client import solve_task


env = MyFirstOpenenvEnvironment()
last_observation = None


def _format_vendors(observation) -> str:
    lines = []
    for vendor in observation.vendors:
        lines.append(
            f"ID={vendor.id} | price={vendor.price:.2f} | delivery_days={vendor.delivery_days} | rating={vendor.rating:.2f}"
        )
    return "\n".join(lines) if lines else "No vendors available"


def generate_scenario(task_type: str) -> tuple[str, str, str, str]:
    global last_observation
    observation = env.reset(task_type)
    last_observation = observation

    vendors_text = _format_vendors(observation)
    constraints_text = json.dumps(observation.constraints, indent=2, sort_keys=True)

    return vendors_text, constraints_text, "", ""


def run_agent() -> tuple[str, str, str]:
    global last_observation
    if last_observation is None:
        return "No scenario generated yet", "0.00", "false"

    action = solve_task(last_observation)
    result = env.step(action)
    last_observation = result

    if action.action_type == "filter":
        selected_vendor = f"valid_vendor_ids={action.valid_vendor_ids or []}"
    else:
        selected_vendor = f"selected_vendor_id={action.selected_vendor_id}"

    reward_text = f"{float(result.reward or 0.0):.2f}"
    done_text = str(bool(result.done)).lower()

    return selected_vendor, reward_text, done_text


with gr.Blocks(title="Procurement Environment UI") as app:
    gr.Markdown("# Procurement Environment Visualizer")

    task_type = gr.Dropdown(
        choices=["easy", "medium", "hard"],
        value="easy",
        label="Task Type",
    )

    with gr.Row():
        generate_btn = gr.Button("Generate Scenario")
        run_btn = gr.Button("Run Agent")

    vendors_output = gr.Textbox(label="Vendors", lines=8)
    constraints_output = gr.Textbox(label="Constraints", lines=6)
    selected_vendor_output = gr.Textbox(label="Selected Vendor")
    reward_output = gr.Textbox(label="Reward")
    done_output = gr.Textbox(label="Done")

    generate_btn.click(
        fn=generate_scenario,
        inputs=[task_type],
        outputs=[
            vendors_output,
            constraints_output,
            selected_vendor_output,
            reward_output,
        ],
    )

    run_btn.click(
        fn=run_agent,
        inputs=[],
        outputs=[selected_vendor_output, reward_output, done_output],
    )


if __name__ == "__main__":
    app.launch()
