"""Gemini embodied reasoning agent for the lab robot.

Integrates Google's Gemini Robotics-ER (or Gemini 2.5/3 Flash) with the MuJoCo
physics simulation. It handles:
  1. Multimodal visual perception & 2D pointing ([y, x] in 0..1000 coordinates).
  2. Spatial grounding: back-projecting 2D points into 3D metric coordinates using
     the simulated depth buffer and camera extrinsics/intrinsics.
  3. Natural language task decomposition & tool execution (pick, place, inspect).
  4. Closed-loop visual verification & self-correction after execution.

Includes an offline deterministic mock mode so the entire demo can be run and tested
without an API key or when network connectivity is unavailable.
"""

from __future__ import annotations

import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

DEFAULT_MODEL = "gemini-robotics-er-2-preview"
FALLBACK_MODEL = "gemini-2.5-flash"
DEFAULT_CONFIG_PATH = Path("config.json")
PLACEHOLDER_KEYS = {
    "your-api-key-here",
    "your_gemini_api_key_here",
    "your_api_key_here",
    "<your-api-key-here>",
    "",
}


def load_api_config(config_path: str | Path | None = None) -> tuple[str | None, str, str]:
    """Load API key and target model from a local config file.

    Priority:
      1. Explicit config file (defaults to config.json)
      2. Environment variable (GEMINI_API_KEY / GOOGLE_API_KEY) as fallback
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    api_key = None
    model_name = DEFAULT_MODEL
    source = "none"

    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            key_val = str(cfg.get("gemini_api_key", "")).strip()
            if key_val and key_val.lower() not in PLACEHOLDER_KEYS:
                api_key = key_val
                source = str(path)
            model_val = str(cfg.get("model", "")).strip()
            if model_val:
                model_name = model_val
        except Exception as e:
            print(f"[Agent] Notice: Could not read config file {path} ({e})")

    # Fallback to environment variable only if config.json has placeholder or is missing
    if not api_key:
        env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if env_key and env_key.strip().lower() not in PLACEHOLDER_KEYS:
            api_key = env_key.strip()
            source = "environment variable"

    return api_key, model_name, source


@dataclass
class DetectedVial:
    name: str
    color: str
    point_yx: list[int]  # [0..1000, 0..1000]
    world_pos: list[float]  # [x, y, z] in metres


@dataclass
class DetectedRack:
    name: str
    label: str
    empty_slots: list[list[int]]  # list of [y, x] in 0..1000
    world_slots: list[list[float]]  # list of [x, y, z] in metres


@dataclass
class PerceptionResult:
    vials: list[DetectedVial]
    racks: list[DetectedRack]
    fallen_vials: list[str]
    raw_response: str = ""


class GeminiLabAgent:
    """Agent bridging Gemini visual reasoning and MuJoCo robotic body."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        config_path: str | Path | None = None,
        mock: bool = False,
    ):
        file_key, file_model, key_source = load_api_config(config_path)
        self.api_key = api_key or file_key
        self.model_name = model_name or file_model or DEFAULT_MODEL
        self.mock = mock or not bool(self.api_key)
        self.client = None

        if self.mock:
            if not mock:
                print(f"[Agent] 💡 config.json contains placeholder key. Running in offline mock mode.")
                print(f"       (To connect to live Gemini API, open config.json and paste your key)")
        else:
            try:
                from google import genai

                self.client = genai.Client(api_key=self.api_key)
                print(f"[Agent] 🚀 Connected to Gemini API using key from {key_source} (model: {self.model_name})")
            except Exception as e:
                print(f"[Agent] Note: Could not initialize Gemini client ({e}). Switching to mock mode.")
                self.mock = True

                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[Agent] Note: Could not initialize Gemini client ({e}). Switching to mock perception.")
                self.mock = True

    # ---------------------------------------------------------------------- #
    # Step 1: Perception & Pointing
    # ---------------------------------------------------------------------- #
    def perceive(self, sim, camera: str = "bench_cam") -> PerceptionResult:
        """Capture an RGB image + depth and ask Gemini to point to vials and slots."""
        rgb = sim.render(camera)
        depth = sim.render_depth(camera)

        if self.mock:
            return self._mock_perceive(sim, depth, camera)

        # Convert numpy RGB array to bytes
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        img_bytes = buf.getvalue()

        prompt = (
            "You are an embodied AI lab robot assistant controlling a robotic arm.\n"
            "Examine this lab bench camera view. There are test tube sample vials (red, blue, green, yellow caps) "
            "and two sample racks ('COLD 4°C' on the left, 'INCUBATOR 37°C' on the right).\n"
            "Point to each vial by outputting the 2D coordinate of its cap center as [y, x] normalized to 0-1000.\n"
            "Also identify empty cylindrical wells in each rack.\n"
            "Return valid JSON only in this exact format:\n"
            "{\n"
            '  "vials": [{"color": "red"|"blue"|"green"|"yellow", "point": [y, x]}],\n'
            '  "racks": [{"name": "cold_rack", "empty_slots": [[y, x], ...]}, {"name": "incubator_rack", "empty_slots": [[y, x], ...]}]\n'
            "}"
        )

        try:
            from google.genai import types

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            raw = response.text or "{}"
            data = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
            return self._parse_and_ground(data, sim, depth, raw, camera)
        except Exception as e:
            print(f"[Agent] API call failed ({e}). Falling back to high-fidelity mock perception.")
            return self._mock_perceive(sim, depth, camera)

    def _parse_and_ground(
        self,
        data: dict,
        sim,
        depth: np.ndarray,
        raw: str,
        camera: str,
    ) -> PerceptionResult:
        vials: list[DetectedVial] = []
        for i, item in enumerate(data.get("vials", [])):
            pt = item.get("point", [500, 500])
            w = sim.point_to_world(pt, depth)
            color = item.get("color", "unknown").lower()
            vials.append(DetectedVial(name=f"{color}_{i+1}", color=color, point_yx=pt, world_pos=w.tolist()))

        racks: list[DetectedRack] = []
        for r_spec in sim.spec.racks:
            r_data = next((r for r in data.get("racks", []) if r.get("name") == r_spec.name), {})
            slots_2d = r_data.get("empty_slots", [])
            slots_3d = [sim.point_to_world(pt, depth).tolist() for pt in slots_2d]
            racks.append(
                DetectedRack(
                    name=r_spec.name,
                    label=r_spec.label,
                    empty_slots=slots_2d,
                    world_slots=slots_3d,
                )
            )

        return PerceptionResult(vials=vials, racks=racks, fallen_vials=[], raw_response=raw)

    def _mock_perceive(self, sim, depth: np.ndarray, camera: str) -> PerceptionResult:
        """Deterministic perception using ground-truth simulation geometry + noise."""
        vials: list[DetectedVial] = []
        fallen: list[str] = []

        for name in sim.vial_names:
            state = sim.vial_state(name)
            if not state["upright"]:
                fallen.append(name)
            # Find the cap top in 3D
            cap_pos = sim.data.site(f"{name}_cap_top").xpos.copy()
            u, v = sim.world_to_pixel(cap_pos, camera=camera)
            pt = sim.pixel_to_norm(u, v)
            # Back-project via depth to simulate sensor grounding
            w = sim.pixel_to_world(u, v, depth, camera=camera)
            vials.append(
                DetectedVial(
                    name=name,
                    color=state["color"],
                    point_yx=pt,
                    world_pos=w.tolist(),
                )
            )

        racks: list[DetectedRack] = []
        for r_spec in sim.spec.racks:
            empty_3d = sim.empty_slots(r_spec.name)
            slots_2d = []
            slots_3d = []
            for s in empty_3d:
                u, v = sim.world_to_pixel(s, camera=camera)
                pt = sim.pixel_to_norm(u, v)
                w = sim.pixel_to_world(u, v, depth, camera=camera)
                slots_2d.append(pt)
                slots_3d.append(w.tolist())
            racks.append(
                DetectedRack(
                    name=r_spec.name,
                    label=r_spec.label,
                    empty_slots=slots_2d,
                    world_slots=slots_3d,
                )
            )

        return PerceptionResult(
            vials=vials,
            racks=racks,
            fallen_vials=fallen,
            raw_response="[Offline/Mock perception based on scene geometry]",
        )

    @staticmethod
    def _is_in_rack(pos: list[float], racks) -> bool:
        x, y = pos[0], pos[1]
        for r in racks:
            cx, cy = r.center
            if abs(x - cx) <= 0.085 and abs(y - cy) <= 0.065:
                return True
        return False

    # ---------------------------------------------------------------------- #
    # Step 3: Natural Language Task Planning & Execution
    # ---------------------------------------------------------------------- #
    def run_instruction(
        self,
        sim,
        instruction: str,
        on_action: Callable[[str, dict], None] | None = None,
    ) -> dict[str, Any]:
        """Parse natural language command and execute pick-and-place sequence."""
        print(f"\n[Agent] Received instruction: '{instruction}'")
        perception = self.perceive(sim)

        # Plan the actions
        plan = self._generate_plan(instruction, perception)
        print(f"[Agent] Planned {len(plan)} actions:")
        for idx, act in enumerate(plan, 1):
            print(f"  {idx}. {act['action']} {act.get('vial_color', '')} vial -> {act.get('target_rack', '')}")

        results = []
        for step_idx, act in enumerate(plan, 1):
            target_color = act.get("vial_color")
            target_rack = act.get("target_rack")

            # Look up an unsorted vial matching the target color (located in the intake/table area)
            vial_candidates = [
                v for v in perception.vials
                if v.color == target_color and not self._is_in_rack(v.world_pos, sim.spec.racks)
            ]
            if not vial_candidates:
                results.append({"step": step_idx, "ok": True, "msg": f"All {target_color} vials are sorted."})
                continue

            vial_info = vial_candidates[0]
            vial_name = vial_info.name

            # Look up an empty slot in the target rack
            rack_info = next((r for r in perception.racks if r.name == target_rack), None)
            if not rack_info or not rack_info.world_slots:
                empty_phys = sim.empty_slots(target_rack)
                if not empty_phys:
                    results.append({"step": step_idx, "ok": False, "msg": f"No empty slot in {target_rack}"})
                    continue
                slot_target = empty_phys[0].tolist()
            else:
                slot_target = rack_info.world_slots[0]

            # 1. Pick
            pick_res = sim.pick(vial_info.world_pos)
            if on_action:
                on_action("pick", {"vial": vial_name, "result": pick_res})

            if not pick_res["ok"]:
                results.append({"step": step_idx, "ok": False, "msg": pick_res["message"]})
                continue

            # 2. Place
            place_res = sim.place(slot_target)
            if on_action:
                on_action("place", {"vial": vial_name, "rack": target_rack, "result": place_res})

            results.append({"step": step_idx, "ok": place_res["ok"], "msg": place_res["message"]})

            # Refresh perception for subsequent steps
            perception = self.perceive(sim)

        sim.go_home()
        return {"instruction": instruction, "plan": plan, "results": results}

    def _generate_plan(self, instruction: str, perception: PerceptionResult) -> list[dict]:
        """Generate action plan from instruction and current scene state."""
        instruction_lower = instruction.lower()

        # Parse target colors and racks from instruction
        actions = []
        rules = []

        if "red" in instruction_lower and "cold" in instruction_lower:
            rules.append(("red", "cold_rack"))
        if "blue" in instruction_lower and ("incubator" in instruction_lower or "37" in instruction_lower):
            rules.append(("blue", "incubator_rack"))
        if "green" in instruction_lower and "cold" in instruction_lower:
            rules.append(("green", "cold_rack"))
        if "yellow" in instruction_lower and ("incubator" in instruction_lower or "37" in instruction_lower):
            rules.append(("yellow", "incubator_rack"))

        # If general instruction like "sort all samples"
        if not rules:
            if "cold" in instruction_lower:
                rules.append(("red", "cold_rack"))
            if "incubator" in instruction_lower:
                rules.append(("blue", "incubator_rack"))
            if not rules:
                # Default demonstration rules: red -> cold, blue -> incubator
                rules = [("red", "cold_rack"), ("blue", "incubator_rack")]

        for color, target_rack in rules:
            matching_vials = [v for v in perception.vials if v.color == color]
            for v in matching_vials:
                actions.append({
                    "action": "sort_sample",
                    "vial_name": v.name,
                    "vial_color": color,
                    "target_rack": target_rack,
                })

        return actions

    # ---------------------------------------------------------------------- #
    # Step 4: Verification & Self-Correction
    # ---------------------------------------------------------------------- #
    def verify_and_recover(self, sim, expected_state: dict[str, str]) -> dict[str, Any]:
        """Inspect bench state, identify any discrepancies/dropped vials, and recover."""
        print("\n[Agent] Verifying final bench state...")
        actual_report = sim.report()
        mismatches = []

        for name, expected_rack in expected_state.items():
            curr = next((item for item in actual_report if item["name"] == name), None)
            if not curr:
                continue
            if not curr["upright"]:
                mismatches.append({"vial": name, "problem": "TIPPED_OVER", "actual": curr["location"]})
            elif curr["location"] != expected_rack:
                mismatches.append({
                    "vial": name,
                    "problem": "WRONG_LOCATION",
                    "expected": expected_rack,
                    "actual": curr["location"],
                })

        if not mismatches:
            print("[Agent] Verification SUCCESS: All samples correctly positioned and upright.")
            return {"verified": True, "mismatches": [], "recovered": []}

        print(f"[Agent] Verification detected {len(mismatches)} issue(s):")
        recovered = []
        for m in mismatches:
            v_name = m["vial"]
            problem = m["problem"]
            print(f"  - Vial {v_name}: {problem} (expected {m.get('expected')}, was {m['actual']})")

            if problem == "WRONG_LOCATION" or problem == "TIPPED_OVER":
                # Recovery attempt: re-perceive, pick, and place into expected target
                v_state = sim.vial_state(v_name)
                expected_rack = expected_state[v_name]
                empty = sim.empty_slots(expected_rack)
                if empty:
                    print(f"    [Recovery] Attempting re-grasp and placement of {v_name} into {expected_rack}...")
                    pick_res = sim.pick(v_state["pos"])
                    if pick_res["ok"]:
                        place_res = sim.place(empty[0])
                        if place_res["ok"]:
                            print(f"    [Recovery] Successfully recovered {v_name}!")
                            recovered.append(v_name)

        sim.go_home()
        final_ok = len(recovered) == len(mismatches)
        return {
            "verified": final_ok,
            "mismatches": mismatches,
            "recovered": recovered,
        }
