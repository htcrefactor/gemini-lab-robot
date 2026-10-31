"""LabSim: the robot's *body* — MuJoCo physics, cameras, IK and motion primitives.

The design is intentionally simple so it is easy to read during a workshop:

* The arm is **position-controlled**: every control tick we compute joint
  targets (analytic IK) and write them straight into ``qpos``, like a stiff
  servo arm. Physics still runs for everything else (vials fall, tip over,
  collide with racks).
* The gripper is an **idealised vacuum/suction cup**: ``pick`` succeeds if a
  vial cap is within reach of the tip, after which the vial hangs below the
  cup until ``place`` releases it. Real lab arms commonly use suction
  end-effectors; we skip the pneumatics.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from . import scene as S

ARM_JOINTS = ["j1_yaw", "j2_shoulder", "j3_elbow", "j4_wrist"]
SAFE_Z = 0.20  # travel height of the suction-cup tip
HOME_TIP = np.array([0.10, 0.0, 0.32])
CONTROL_DT = 0.02  # seconds of simulated time per control tick
GRASP_RADIUS = 0.032  # tip must be this close to a cap centre to grab it


@dataclass
class Held:
    name: str
    offset0: np.ndarray  # vial origin - tip at grasp time (world frame)
    quat0: np.ndarray
    t0: float


def _slerp(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - a) * th) * q0 + math.sin(a * th) * q1) / math.sin(th)


def _min_jerk(s: float) -> float:
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class LabSim:
    def __init__(
        self,
        seed: int = 0,
        width: int = 800,
        height: int = 600,
        record: bool = True,
        video_size: tuple[int, int] = (640, 480),
        viewer: bool = False,
        chaos: float = 0.0,
    ):
        self.spec = S.random_scene(seed)
        xml, assets = S.build_mjcf(self.spec)
        self.model = mujoco.MjModel.from_xml_string(xml, assets)
        self.data = mujoco.MjData(self.model)
        self.width, self.height = width, height
        self.renderer = mujoco.Renderer(self.model, height, width)
        self.video_renderer = mujoco.Renderer(self.model, video_size[1], video_size[0]) if record else None
        self.record = record
        self.frames: list[np.ndarray] = []
        self._tick = 0
        self.rng = np.random.default_rng(seed + 1000)
        self.chaos = chaos  # probability of losing vacuum mid-transport (step 4 demo)

        self._arm_qadr = [self.model.jnt_qposadr[self.model.joint(j).id] for j in ARM_JOINTS]
        self._arm_dadr = [self.model.jnt_dofadr[self.model.joint(j).id] for j in ARM_JOINTS]
        self._tip_site = self.model.site("tip").id
        self.vial_names = [v.name for v in self.spec.vials]
        self.vial_colors = {v.name: v.color for v in self.spec.vials}
        self.held: Held | None = None
        self.viewer = None

        self.q = self.ik(HOME_TIP)
        self._apply_arm(self.q)
        mujoco.mj_forward(self.model, self.data)
        self.settle(0.5)

        if viewer:
            import importlib

            mj_viewer = importlib.import_module("mujoco.viewer")  # needs `mjpython` on macOS
            self.viewer = mj_viewer.launch_passive(self.model, self.data)

    # ------------------------------------------------------------------ #
    # Cameras: render, intrinsics, pixel <-> world
    # ------------------------------------------------------------------ #
    def render(self, camera: str = "bench_cam") -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def render_depth(self, camera: str = "bench_cam") -> np.ndarray:
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=camera)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        return depth  # metres along the optical axis

    def intrinsics(self, camera: str = "bench_cam") -> tuple[float, float, float]:
        """Pinhole focal length (px) and principal point for the render size."""
        fovy = math.radians(self.model.cam_fovy[self.model.camera(camera).id])
        f = 0.5 * self.height / math.tan(fovy / 2)
        return f, self.width / 2, self.height / 2

    def _cam_pose(self, camera: str) -> tuple[np.ndarray, np.ndarray]:
        cid = self.model.camera(camera).id
        return self.data.cam_xpos[cid].copy(), self.data.cam_xmat[cid].reshape(3, 3).copy()

    def pixel_to_world(self, u: float, v: float, depth: np.ndarray, camera: str = "bench_cam", win: int = 2) -> np.ndarray:
        """Back-project pixel (u, v) using the depth image -> world xyz.

        MuJoCo cameras look down their local -Z axis with +Y up, +X right.
        """
        f, cx, cy = self.intrinsics(camera)
        ui, vi = int(round(u)), int(round(v))
        ui, vi = np.clip(ui, 0, self.width - 1), np.clip(vi, 0, self.height - 1)
        patch = depth[max(vi - win, 0) : vi + win + 1, max(ui - win, 0) : ui + win + 1]
        d = float(np.median(patch))
        p_cam = np.array([(u - cx) / f * d, -(v - cy) / f * d, -d])
        pos, R = self._cam_pose(camera)
        return pos + R @ p_cam

    def world_to_pixel(self, p: np.ndarray, camera: str = "bench_cam") -> tuple[float, float]:
        f, cx, cy = self.intrinsics(camera)
        pos, R = self._cam_pose(camera)
        pc = R.T @ (np.asarray(p) - pos)
        return cx + f * pc[0] / -pc[2], cy - f * pc[1] / -pc[2]

    def norm_to_pixel(self, point_yx) -> tuple[float, float]:
        """Gemini points are [y, x] normalised to 0..1000 -> pixel (u, v)."""
        y, x = point_yx
        return x / 1000 * self.width, y / 1000 * self.height

    def pixel_to_norm(self, u: float, v: float) -> list[int]:
        return [int(round(v / self.height * 1000)), int(round(u / self.width * 1000))]

    def point_to_world(self, point_yx, depth: np.ndarray | None = None) -> np.ndarray:
        """The key 'grounding' step: Gemini 2D point -> 3D world position."""
        depth = self.render_depth() if depth is None else depth
        u, v = self.norm_to_pixel(point_yx)
        return self.pixel_to_world(u, v, depth)

    # ------------------------------------------------------------------ #
    # Kinematics
    # ------------------------------------------------------------------ #
    @staticmethod
    def ik(tip: np.ndarray) -> np.ndarray:
        """Analytic IK for a tool pointing straight down.

        Base yaw points the arm at the target; then shoulder + elbow form a
        planar 2-link arm reaching the wrist; the wrist keeps the tool vertical.
        Angles are measured from vertical, positive = leaning away from base.
        """
        x, y, z = tip
        q1 = math.atan2(y, x)
        r = math.hypot(x, y)
        dx, dz = r, (z + S.TOOL) - S.SHOULDER_HEIGHT
        L1, L2 = S.UPPER_ARM, S.FOREARM
        D = (dx * dx + dz * dz - L1 * L1 - L2 * L2) / (2 * L1 * L2)
        q3 = math.acos(np.clip(D, -1.0, 1.0))  # elbow-up solution
        q2 = math.atan2(dx, dz) - math.atan2(L2 * math.sin(q3), L1 + L2 * math.cos(q3))
        q4 = math.pi - q2 - q3
        return np.array([q1, q2, q3, q4])

    def tip_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._tip_site].copy()

    @staticmethod
    def reachable(tip: np.ndarray) -> bool:
        r = math.hypot(tip[0], tip[1])
        dz = tip[2] + S.TOOL - S.SHOULDER_HEIGHT
        return 0.12 < math.hypot(r, dz) < (S.UPPER_ARM + S.FOREARM) * 0.98 and r > 0.10

    # ------------------------------------------------------------------ #
    # Low-level stepping
    # ------------------------------------------------------------------ #
    def _apply_arm(self, q: np.ndarray) -> None:
        for a, d, val in zip(self._arm_qadr, self._arm_dadr, q):
            self.data.qpos[a] = val
            self.data.qvel[d] = 0.0

    def _vial_qadr(self, name: str) -> tuple[int, int]:
        jid = self.model.joint(f"{name}_joint").id
        return self.model.jnt_qposadr[jid], self.model.jnt_dofadr[jid]

    def _apply_held(self) -> None:
        if self.held is None:
            return
        tip = self.data.site_xpos[self._tip_site]
        a = min(1.0, (self.data.time - self.held.t0) / 0.4)
        a = _min_jerk(a)
        offset = (1 - a) * self.held.offset0 + a * np.array([0, 0, -S.VIAL_HEIGHT])
        quat = _slerp(self.held.quat0, np.array([1.0, 0, 0, 0]), a)
        qa, da = self._vial_qadr(self.held.name)
        self.data.qpos[qa : qa + 3] = tip + offset
        self.data.qpos[qa + 3 : qa + 7] = quat
        self.data.qvel[da : da + 6] = 0.0

    def _control_tick(self) -> None:
        n_sub = int(round(CONTROL_DT / self.model.opt.timestep))
        for _ in range(n_sub):
            self._apply_arm(self.q)
            mujoco.mj_step(self.model, self.data)
            self._apply_arm(self.q)
            mujoco.mj_kinematics(self.model, self.data)
            self._apply_held()
        mujoco.mj_forward(self.model, self.data)
        self._tick += 1
        if self.record and self._tick % 2 == 0:  # 25 fps video
            self.video_renderer.update_scene(self.data, camera="bench_cam")
            self.frames.append(self.video_renderer.render().copy())
        if self.viewer is not None:
            self.viewer.sync()
            time.sleep(CONTROL_DT)

    def settle(self, seconds: float = 0.5) -> None:
        for _ in range(int(seconds / CONTROL_DT)):
            self._control_tick()

    # ------------------------------------------------------------------ #
    # Motion primitives
    # ------------------------------------------------------------------ #
    def move_tip(self, target: np.ndarray, speed: float = 0.35, min_time: float = 0.3) -> None:
        """Straight-line Cartesian move of the suction-cup tip (min-jerk timing)."""
        start = self.tip_pos()
        target = np.asarray(target, dtype=float)
        dist = float(np.linalg.norm(target - start))
        T = max(min_time, dist / speed)
        n = max(1, int(T / CONTROL_DT))
        for i in range(1, n + 1):
            p = start + _min_jerk(i / n) * (target - start)
            self.q = self.ik(p)
            self._control_tick()

    def go_home(self) -> None:
        tip = self.tip_pos()
        if tip[2] < SAFE_Z:
            self.move_tip([tip[0], tip[1], SAFE_Z])
        self.move_tip(HOME_TIP)

    def _travel_to(self, xy: np.ndarray) -> None:
        tip = self.tip_pos()
        if tip[2] < SAFE_Z - 1e-3:
            self.move_tip([tip[0], tip[1], SAFE_Z])
        self.move_tip([xy[0], xy[1], SAFE_Z])

    def _cap_center(self, name: str) -> np.ndarray:
        body = self.data.body(name)
        R = body.xmat.reshape(3, 3)
        return body.xpos + R @ np.array([0, 0, S.VIAL_GLASS_HEIGHT + S.CAP_HEIGHT / 2])

    def pick(self, target: np.ndarray) -> dict:
        """Descend on a 3D point (ideally the top of a vial cap) and apply suction."""
        target = np.asarray(target, dtype=float)
        if self.held is not None:
            return {"ok": False, "message": f"Already holding {self.held.name}; place it first."}
        if not self.reachable([target[0], target[1], SAFE_Z]):
            return {"ok": False, "message": f"Target {np.round(target, 3).tolist()} is out of reach."}
        self._travel_to(target[:2])
        z = max(target[2] + 0.004, 0.006)
        self.move_tip([target[0], target[1], z], speed=0.2)
        self.settle(0.1)
        tip = self.tip_pos()
        dists = {n: float(np.linalg.norm(self._cap_center(n) - tip)) for n in self.vial_names}
        name = min(dists, key=dists.get)
        if dists[name] > GRASP_RADIUS:
            self.move_tip([tip[0], tip[1], SAFE_Z])
            return {"ok": False, "message": "Suction failed: no vial cap under the gripper (vacuum sensor reads open air)."}
        body = self.data.body(name)
        self.held = Held(name, body.xpos.copy() - tip, body.xquat.copy(), self.data.time)
        self.move_tip([tip[0], tip[1], SAFE_Z], speed=0.25)
        return {"ok": True, "message": "Vacuum sealed: a vial is attached to the gripper."}

    def place(self, target: np.ndarray) -> dict:
        """Carry the held vial over a 3D point (e.g. an empty rack slot) and release it."""
        target = np.asarray(target, dtype=float)
        if self.held is None:
            return {"ok": False, "message": "Not holding anything; pick a vial first."}
        if not self.reachable([target[0], target[1], SAFE_Z]):
            return {"ok": False, "message": f"Target {np.round(target, 3).tolist()} is out of reach."}
        tip = self.tip_pos()
        if self.chaos > 0 and self.rng.random() < self.chaos:
            # Step-4 failure injection: lose vacuum half-way through the move.
            mid = (tip[:2] + target[:2]) / 2
            self._travel_to(mid)
            self.held = None
            self.settle(0.8)
            self.go_home()
            return {"ok": False, "message": "WARNING: vacuum pressure lost during transport - the vial was dropped somewhere along the way."}
        self._travel_to(target[:2])
        z = max(target[2], 0.0) + S.VIAL_HEIGHT + 0.006
        self.move_tip([target[0], target[1], z], speed=0.2)
        self.settle(0.1)
        self.held = None
        self.settle(0.4)
        self.move_tip([target[0], target[1], SAFE_Z], speed=0.3)
        return {"ok": True, "message": "Released the vial."}

    def bump(self, name: str | None = None) -> str:
        """Simulate someone bumping the bench: knock one vial over."""
        name = name or str(self.rng.choice(self.vial_names))
        _, da = self._vial_qadr(name)
        self.data.qvel[da : da + 3] = [0.25, 0.4, 0.3]
        self.data.qvel[da + 3 : da + 6] = [12.0, -6.0, 0.0]
        self.settle(1.5)
        return name

    # ------------------------------------------------------------------ #
    # Ground truth (for scoring & offline mode — the agent never sees this)
    # ------------------------------------------------------------------ #
    def vial_state(self, name: str) -> dict:
        body = self.data.body(name)
        pos = body.xpos.copy()
        upright = float(body.xmat.reshape(3, 3)[2, 2]) > 0.9
        location = "table"
        for rack in self.spec.racks:
            cx, cy = rack.center
            if abs(pos[0] - cx) < S.RACK_HALF[0] and abs(pos[1] - cy) < S.RACK_HALF[1] and pos[2] > S.RACK_TOP - 0.006:
                location = rack.name
        if self.held is not None and self.held.name == name:
            location = "gripper"
        return {
            "name": name,
            "color": self.vial_colors[name],
            "location": location,
            "upright": upright,
            "pos": pos,
        }

    def report(self) -> list[dict]:
        return [self.vial_state(n) for n in self.vial_names]

    def empty_slots(self, rack_name: str) -> list[np.ndarray]:
        rack = next(r for r in self.spec.racks if r.name == rack_name)
        vials = [self.data.body(n).xpos[:2] for n in self.vial_names]
        return [s for s in rack.slots() if all(np.linalg.norm(s[:2] - v) > 0.02 for v in vials)]

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    def save_video(self, path: str | Path, fps: int = 25) -> Path | None:
        if not self.frames:
            return None
        import imageio.v2 as imageio

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(path, self.frames, fps=fps, macro_block_size=8)
        return path

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
        self.renderer.close()
        if self.video_renderer is not None:
            self.video_renderer.close()
