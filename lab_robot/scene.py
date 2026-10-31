"""Procedurally generated MuJoCo lab-bench scene.

Everything (robot, racks, vials, printed labels) is built from primitive shapes
and in-memory textures, so the repo has no binary assets to download.

World frame (metres):
  * origin  = robot base, on the table top (table top is z = 0)
  * +x      = away from the robot, towards the camera
  * +y      = to the robot's left (image right, from the bench camera)
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------- #
# Robot geometry (a 4-DoF desktop arm: base yaw + 3 pitch joints)
# --------------------------------------------------------------------------- #
SHOULDER_HEIGHT = 0.14  # height of the shoulder pitch axis above the table
UPPER_ARM = 0.28  # shoulder -> elbow
FOREARM = 0.26  # elbow -> wrist
TOOL = 0.10  # wrist -> suction-cup tip

# --------------------------------------------------------------------------- #
# Vial geometry (body origin = centre of the vial's bottom face)
# --------------------------------------------------------------------------- #
VIAL_RADIUS = 0.012
VIAL_GLASS_HEIGHT = 0.060
CAP_RADIUS = 0.0135
CAP_HEIGHT = 0.012
VIAL_HEIGHT = VIAL_GLASS_HEIGHT + CAP_HEIGHT  # bottom -> top of cap

CAP_COLORS = {
    "red": (0.85, 0.10, 0.10, 1),
    "blue": (0.10, 0.30, 0.90, 1),
    "green": (0.10, 0.65, 0.20, 1),
    "yellow": (0.95, 0.80, 0.10, 1),
}

# --------------------------------------------------------------------------- #
# Racks
# --------------------------------------------------------------------------- #
RACK_TOP = 0.02  # rack plate thickness (top surface height)
RACK_HALF = (0.075, 0.055)  # half-size of the plate in (x, y)
SLOT_PITCH = 0.045


@dataclass
class Rack:
    name: str
    label: str
    center: tuple[float, float]
    rgba: tuple[float, float, float, float]

    def slots(self) -> list[np.ndarray]:
        """World positions (x, y, z_top) of the 2 x 3 slot grid."""
        cx, cy = self.center
        out = []
        for i in (-1, 1):  # rows along x
            for j in (-1, 0, 1):  # columns along y
                out.append(
                    np.array([cx + i * SLOT_PITCH / 2, cy + j * SLOT_PITCH, RACK_TOP])
                )
        return out


RACKS = [
    Rack("cold_rack", "COLD 4°C", (0.30, -0.25), (0.62, 0.80, 0.95, 1)),
    Rack("incubator_rack", "INCUBATOR 37°C", (0.30, 0.25), (0.93, 0.72, 0.55, 1)),
]


@dataclass
class Vial:
    name: str
    color: str
    xy: tuple[float, float]


@dataclass
class SceneSpec:
    vials: list[Vial] = field(default_factory=list)
    racks: list[Rack] = field(default_factory=lambda: list(RACKS))


def random_scene(seed: int = 0, colors: list[str] | None = None) -> SceneSpec:
    """Scatter vials in the 'intake' area in front of the robot."""
    rng = np.random.default_rng(seed)
    colors = colors or ["red", "red", "blue", "blue", "green", "yellow"]
    vials: list[Vial] = []
    placed: list[np.ndarray] = []
    counts: dict[str, int] = {}
    for color in colors:
        for _ in range(1000):
            xy = np.array([rng.uniform(0.20, 0.42), rng.uniform(-0.11, 0.11)])
            if all(np.linalg.norm(xy - p) > 0.055 for p in placed):
                break
        placed.append(xy)
        counts[color] = counts.get(color, 0) + 1
        vials.append(Vial(f"{color}_{counts[color]}", color, (float(xy[0]), float(xy[1]))))
    return SceneSpec(vials=vials)


# --------------------------------------------------------------------------- #
# Label textures (rendered with Pillow, passed to MuJoCo as in-memory assets)
# --------------------------------------------------------------------------- #
def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _label_png(text: str) -> bytes:
    w, h = 512, 128
    img = Image.new("RGB", (w, h), (250, 250, 245))
    draw = ImageDraw.Draw(img)
    draw.rectangle([4, 4, w - 5, h - 5], outline=(30, 30, 30), width=6)
    size = 52
    while size > 24:
        font = _font(size)
        box = draw.textbbox((0, 0), text, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        if tw <= w - 36 and th <= h - 16:
            break
        size -= 4
    draw.text(((w - tw) / 2 - box[0], (h - th) / 2 - box[1]), text, fill=(20, 20, 20), font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# MJCF generation
# --------------------------------------------------------------------------- #
def _rgba(c) -> str:
    return " ".join(f"{v:.3f}" for v in c)


def build_mjcf(spec: SceneSpec) -> tuple[str, dict[str, bytes]]:
    """Return (xml_string, assets) ready for mujoco.MjModel.from_xml_string."""
    assets: dict[str, bytes] = {}
    textures, materials, rack_xml = [], [], []

    for rack in spec.racks:
        fname = f"{rack.name}_label.png"
        assets[fname] = _label_png(rack.label)
        textures.append(f'<texture name="{rack.name}_tex" type="2d" file="{fname}"/>')
        materials.append(f'<material name="{rack.name}_mat" texture="{rack.name}_tex" texrepeat="1 1"/>')
        cx, cy = rack.center
        slots = "\n".join(
            f'<geom name="{rack.name}_slot{k}" type="cylinder" size="0.017 0.0008" '
            f'pos="{s[0] - cx:.4f} {s[1] - cy:.4f} {RACK_TOP / 2 + 0.0008:.4f}" '
            f'rgba="0.15 0.15 0.18 1" contype="0" conaffinity="0"/>'
            for k, s in enumerate(rack.slots())
        )
        # Label strip on the table, on the camera side of the rack, readable from the camera.
        rack_xml.append(
            f"""
    <body name="{rack.name}" pos="{cx} {cy} {RACK_TOP / 2}">
      <geom name="{rack.name}_plate" type="box" size="{RACK_HALF[0]} {RACK_HALF[1]} {RACK_TOP / 2}"
            rgba="{_rgba(rack.rgba)}" friction="1 0.01 0.001"/>
      {slots}
    </body>
    <geom name="{rack.name}_label" type="plane" size="0.064 0.016 0.01" material="{rack.name}_mat"
          pos="{cx + RACK_HALF[0] + 0.025} {cy} 0.002" euler="0 0 90" contype="0" conaffinity="0"/>"""
        )

    vial_xml = []
    for v in spec.vials:
        vial_xml.append(
            f"""
    <body name="{v.name}" pos="{v.xy[0]:.4f} {v.xy[1]:.4f} 0.0005">
      <freejoint name="{v.name}_joint"/>
      <geom name="{v.name}_glass" type="cylinder" size="{VIAL_RADIUS} {VIAL_GLASS_HEIGHT / 2}"
            pos="0 0 {VIAL_GLASS_HEIGHT / 2}" rgba="0.85 0.92 0.95 1" mass="0.012"
            friction="1 0.01 0.001" condim="4"/>
      <geom name="{v.name}_liquid" type="cylinder" size="{VIAL_RADIUS * 0.85} {VIAL_GLASS_HEIGHT * 0.3}"
            pos="0 0 {VIAL_GLASS_HEIGHT * 0.32}" rgba="{_rgba(CAP_COLORS[v.color][:3] + (0.35,))}"
            contype="0" conaffinity="0" mass="0"/>
      <geom name="{v.name}_cap" type="cylinder" size="{CAP_RADIUS} {CAP_HEIGHT / 2}"
            pos="0 0 {VIAL_GLASS_HEIGHT + CAP_HEIGHT / 2}" rgba="{_rgba(CAP_COLORS[v.color])}"
            mass="0.003" friction="1 0.01 0.001" condim="4"/>
      <site name="{v.name}_cap_top" pos="0 0 {VIAL_HEIGHT}" size="0.003" rgba="0 0 0 0"/>
    </body>"""
        )

    # Robot links have no collision geometry: the arm is position-controlled and
    # the suction gripper is "idealised" (see sim.py). Vials collide with
    # the table, racks and each other.
    link = 'contype="0" conaffinity="0"'
    xml = f"""
<mujoco model="gemini_lab_bench">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <global offwidth="1600" offheight="1200"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.25 0.25 0.25" diffuse="0.35 0.35 0.35" specular="0.05 0.05 0.05"/>
  </visual>
  <asset>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.25 0.27 0.30" rgb2="0.20 0.22 0.25"
             width="512" height="512"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="6 6"/>
    <material name="table_mat" rgba="0.30 0.32 0.35 1" specular="0.15" shininess="0.3"/>
    <material name="robot_white" rgba="0.92 0.92 0.94 1" specular="0.3" shininess="0.5"/>
    <material name="robot_dark" rgba="0.18 0.19 0.22 1" specular="0.3"/>
    <material name="robot_accent" rgba="0.26 0.52 0.96 1" specular="0.4"/>
    {"".join(textures)}
    {"".join(materials)}
  </asset>

  <worldbody>
    <light name="key" pos="0.6 -0.4 1.4" dir="-0.3 0.25 -1" diffuse="0.45 0.45 0.45" castshadow="true"/>
    <light name="fill" pos="0.2 0.6 1.2" dir="0 -0.3 -1" diffuse="0.25 0.25 0.25" castshadow="false"/>
    <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 -0.75" material="floor_mat"/>
    <geom name="table" type="box" size="0.40 0.55 0.02" pos="0.25 0 -0.02" material="table_mat"
          friction="1 0.01 0.001"/>

    <camera name="bench_cam" pos="0.74 0 0.64" xyaxes="0 1 0 -0.8 0 0.6" fovy="58"/>
    <camera name="top_cam" pos="0.28 0 1.10" xyaxes="0 1 0 -1 0 0" fovy="50"/>

    <!-- ===================== 4-DoF desktop robot arm ===================== -->
    <body name="robot_base" pos="0 0 0">
      <geom type="cylinder" size="0.075 0.01" pos="0 0 0.01" material="robot_dark" {link}/>
      <geom type="cylinder" size="0.055 0.035" pos="0 0 0.045" material="robot_white" {link}/>
      <body name="link1" pos="0 0 0.08">
        <joint name="j1_yaw" type="hinge" axis="0 0 1" range="-3.1 3.1"/>
        <geom type="cylinder" size="0.05 0.03" pos="0 0 0.03" material="robot_white" {link}/>
        <geom type="cylinder" size="0.045 0.035" pos="0 0 {SHOULDER_HEIGHT - 0.08}" euler="1.5708 0 0"
              material="robot_accent" {link}/>
        <body name="upper_arm" pos="0 0 {SHOULDER_HEIGHT - 0.08}">
          <joint name="j2_shoulder" type="hinge" axis="0 1 0" range="-1.2 2.4"/>
          <geom type="capsule" fromto="0 0 0 0 0 {UPPER_ARM}" size="0.032" material="robot_white" {link}/>
          <geom type="cylinder" size="0.038 0.033" pos="0 0 {UPPER_ARM}" euler="1.5708 0 0"
                material="robot_accent" {link}/>
          <body name="forearm" pos="0 0 {UPPER_ARM}">
            <joint name="j3_elbow" type="hinge" axis="0 1 0" range="-0.6 2.9"/>
            <geom type="capsule" fromto="0 0 0 0 0 {FOREARM}" size="0.026" material="robot_white" {link}/>
            <geom type="cylinder" size="0.03 0.028" pos="0 0 {FOREARM}" euler="1.5708 0 0"
                  material="robot_accent" {link}/>
            <body name="tool" pos="0 0 {FOREARM}">
              <joint name="j4_wrist" type="hinge" axis="0 1 0" range="-1.0 4.0"/>
              <geom type="capsule" fromto="0 0 0 0 0 {TOOL - 0.035}" size="0.018" material="robot_dark" {link}/>
              <geom type="cylinder" size="0.008 0.012" pos="0 0 {TOOL - 0.023}" material="robot_white" {link}/>
              <geom name="suction_cup" type="cylinder" size="0.013 0.006" pos="0 0 {TOOL - 0.006}"
                    rgba="0.1 0.1 0.1 1" {link}/>
              <site name="tip" pos="0 0 {TOOL}" size="0.004" rgba="1 0 1 0"/>
            </body>
          </body>
        </body>
      </body>
    </body>
    {"".join(rack_xml)}
    {"".join(vial_xml)}
  </worldbody>
</mujoco>
"""
    return xml, assets
