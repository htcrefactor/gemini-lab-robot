#!/usr/bin/env python3
"""Main walkthrough entry point for the Gemini Lab Robot demo.

Reproducible on any macOS (Apple Silicon M-series), Linux, or Windows machine
with zero GPU requirement.

Usage:
    # Run individual steps:
    python demo.py --step 0    # Step 0: Inspect simulated lab bench & camera views
    python demo.py --step 1    # Step 1: Gemini 2D Pointing -> 3D Spatial Grounding
    python demo.py --step 2    # Step 2: Cartesian Pick-and-Place execution
    python demo.py --step 3    # Step 3: Natural language sample sorting
    python demo.py --step 4    # Step 4: Closed-loop verification & error recovery

    # Or run the full end-to-end workshop demo:
    python demo.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lab_robot.agent import GeminiLabAgent
from lab_robot.sim import LabSim

OUTPUT_DIR = Path("outputs")


def step_0_inspect(sim: LabSim) -> None:
    """Step 0: Render initial bench views."""
    print("=" * 60)
    print("🔬 STEP 0: Inspecting Lab Bench & Camera Feeds")
    print("=" * 60)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bench_img = Image.fromarray(sim.render("bench_cam"))
    top_img = Image.fromarray(sim.render("top_cam"))

    bench_path = OUTPUT_DIR / "step0_bench_cam.png"
    top_path = OUTPUT_DIR / "step0_top_cam.png"

    bench_img.save(bench_path)
    top_img.save(top_path)

    print(f"✅ Bench camera view saved to: {bench_path}")
    print(f"✅ Top-down camera view saved to: {top_path}")
    print(f"Initial vials detected in scene: {len(sim.vial_names)}")
    for v in sim.report():
        print(f"  - {v['name']} ({v['color']}): pos = {np.round(v['pos'], 3).tolist()} at {v['location']}")


def step_1_pointing_and_grounding(sim: LabSim, agent: GeminiLabAgent) -> None:
    """Step 1: Gemini 2D pointing and 3D metric back-projection."""
    print("\n" + "=" * 60)
    print("👁️ STEP 1: Gemini Visual Pointing & 3D Spatial Grounding")
    print("=" * 60)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    depth = sim.render_depth("bench_cam")
    perception = agent.perceive(sim, camera="bench_cam")

    # Draw detected points and labels onto the RGB render
    rgb = sim.render("bench_cam")
    annotated = Image.fromarray(rgb)
    draw = ImageDraw.Draw(annotated)

    print(f"Detected {len(perception.vials)} vials:")
    for v in perception.vials:
        u, v_coord = sim.norm_to_pixel(v.point_yx)
        r = 8
        draw.ellipse([u - r, v_coord - r, u + r, v_coord + r], fill=(255, 255, 0), outline=(0, 0, 0), width=2)
        draw.text((u + 10, v_coord - 10), f"{v.name}\n3D: {np.round(v.world_pos, 2).tolist()}", fill=(255, 255, 255))
        print(f"  📍 {v.name} ({v.color}) -> 2D [y,x]: {v.point_yx} => 3D world (m): {np.round(v.world_pos, 3).tolist()}")

    print(f"\nDetected {len(perception.racks)} racks:")
    for r in perception.racks:
        print(f"  🏷️ {r.label} ({r.name}): {len(r.world_slots)} empty slots")
        for idx, (slot_2d, slot_3d) in enumerate(zip(r.empty_slots, r.world_slots)):
            u, v_coord = sim.norm_to_pixel(slot_2d)
            draw.rectangle([u - 4, v_coord - 4, u + 4, v_coord + 4], outline=(0, 255, 255), width=2)

    annotated_path = OUTPUT_DIR / "step1_grounding_annotated.png"
    annotated.save(annotated_path)
    print(f"\n✅ Annotated perception image saved to: {annotated_path}")


def step_2_pick_and_place(sim: LabSim, agent: GeminiLabAgent) -> None:
    """Step 2: Single pick-and-place primitive execution."""
    print("\n" + "=" * 60)
    print("🦾 STEP 2: Executing Pick-and-Place Primitive")
    print("=" * 60)

    perception = agent.perceive(sim)
    if not perception.vials:
        print("❌ No vials available to pick.")
        return

    target_vial = perception.vials[0]
    target_rack = perception.racks[0]
    if not target_rack.world_slots:
        print("❌ No empty slots in target rack.")
        return
    slot_3d = target_rack.world_slots[0]

    print(f"👉 Picking {target_vial.name} at {np.round(target_vial.world_pos, 3).tolist()}...")
    pick_res = sim.pick(target_vial.world_pos)
    print(f"   Pick result: {pick_res['message']}")

    print(f"👉 Transporting & placing into {target_rack.name} at {np.round(slot_3d, 3).tolist()}...")
    place_res = sim.place(slot_3d)
    print(f"   Place result: {place_res['message']}")

    sim.go_home()
    res_path = OUTPUT_DIR / "step2_after_pick_place.png"
    Image.fromarray(sim.render("bench_cam")).save(res_path)
    print(f"✅ State snapshot saved to: {res_path}")


def step_3_natural_language_sorting(sim: LabSim, agent: GeminiLabAgent) -> None:
    """Step 3: Multi-step natural language instruction execution."""
    print("\n" + "=" * 60)
    print("🗣️ STEP 3: Natural Language Autonomous Sample Sorting")
    print("=" * 60)

    instruction = "Transfer all red sample vials to the COLD 4°C rack and all blue vials to INCUBATOR 37°C."
    result = agent.run_instruction(sim, instruction)

    print("\nExecution summary:")
    for res in result["results"]:
        status = "✅" if res["ok"] else "⚠️"
        print(f"  {status} Step {res['step']}: {res['msg']}")

    res_path = OUTPUT_DIR / "step3_sorted_bench.png"
    Image.fromarray(sim.render("bench_cam")).save(res_path)
    print(f"✅ Final sorted bench saved to: {res_path}")


def step_4_verification_and_recovery(sim: LabSim, agent: GeminiLabAgent) -> None:
    """Step 4: Disturbance injection, closed-loop verification, and self-correction."""
    print("\n" + "=" * 60)
    print("🔄 STEP 4: Disturbance Injection & Self-Correction")
    print("=" * 60)

    # 1. Simulate a physical bench disturbance: knock over one vial
    bumped_vial = sim.bump()
    print(f"💥 Disturbance injected: Vial '{bumped_vial}' was bumped and knocked out of position!")

    disturbed_path = OUTPUT_DIR / "step4_disturbed.png"
    Image.fromarray(sim.render("bench_cam")).save(disturbed_path)
    print(f"   Disturbed snapshot saved to: {disturbed_path}")

    # 2. Define expected state for the experiment
    expected_state = {v: "cold_rack" if "red" in v else "incubator_rack" for v in sim.vial_names if "red" in v or "blue" in v}

    # 3. Agent inspects, detects the discrepancy, and recovers
    recovery_report = agent.verify_and_recover(sim, expected_state)
    print(f"\nRecovery status: {'SUCCESS' if recovery_report['verified'] else 'PARTIAL'}")

    final_path = OUTPUT_DIR / "step4_recovered.png"
    Image.fromarray(sim.render("bench_cam")).save(final_path)
    print(f"✅ Post-recovery snapshot saved to: {final_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini Robotics Lab Assistant Demo")
    parser.add_argument("--step", type=int, choices=[0, 1, 2, 3, 4], help="Run a specific workshop step")
    parser.add_argument("--all", action="store_true", help="Run the complete end-to-end demo")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bench layout")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json containing gemini_api_key")
    parser.add_argument("--viewer", action="store_true", help="Launch interactive 3D simulation window (uses mjpython on macOS)")
    parser.add_argument("--mock", action="store_true", help="Force offline mock perception without API key")
    parser.add_argument("--video", action="store_true", default=True, help="Record MP4 video of execution")
    args = parser.parse_args()

    if args.step is None and not args.all:
        parser.print_help()
        print("\nTip: Run `python demo.py --all` or `python demo.py --step 0` to begin!")
        sys.exit(0)

    print("\n🤖 Initializing Gemini Lab Robot Simulation Environment...")
    sim = LabSim(seed=args.seed, record=args.video, viewer=args.viewer)
    agent = GeminiLabAgent(mock=args.mock, config_path=args.config)

    try:
        if args.step == 0:
            step_0_inspect(sim)
        elif args.step == 1:
            step_0_inspect(sim)
            step_1_pointing_and_grounding(sim, agent)
        elif args.step == 2:
            step_0_inspect(sim)
            step_1_pointing_and_grounding(sim, agent)
            step_2_pick_and_place(sim, agent)
        elif args.step == 3:
            step_0_inspect(sim)
            step_3_natural_language_sorting(sim, agent)
        elif args.step == 4:
            step_0_inspect(sim)
            step_3_natural_language_sorting(sim, agent)
            step_4_verification_and_recovery(sim, agent)
        elif args.all:
            step_0_inspect(sim)
            step_1_pointing_and_grounding(sim, agent)
            step_2_pick_and_place(sim, agent)
            step_3_natural_language_sorting(sim, agent)
            step_4_verification_and_recovery(sim, agent)

        if args.video and sim.frames:
            video_file = OUTPUT_DIR / "gemini_lab_robot_demo.mp4"
            sim.save_video(video_file)
            print(f"\n🎬 Simulation video successfully generated: {video_file} ({len(sim.frames)} frames)")

    finally:
        if args.viewer and sim.viewer is not None:
            print("\n👀 3D Simulation window is open. Close the window or press ESC in the viewer to exit...")
            import time

            while sim.viewer.is_running():
                time.sleep(0.1)
        sim.close()
        print("\n✨ Demo execution completed successfully!")


if __name__ == "__main__":
    main()
