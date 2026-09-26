"""All tunable parameters in one place.

Units: seconds for time; image pixels for positions. Distances and speeds that
must be perspective-robust are expressed in *object sizes*: the object's
sqrt(bbox area), so "speed 1.0" means the object moves one own-size per second.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = Path(os.environ.get("ROADWATCH_WEIGHTS", ROOT / "weights"))
SEED = 1234


@dataclass
class DetectorProfile:
    weights: str
    imgsz: int
    batch: int
    half: bool


PROFILES = {
    # T4-class GPU: larger model and input for small pedestrians far from the camera.
    "gpu": DetectorProfile(weights="yolo11m.pt", imgsz=1280, batch=8, half=True),
    # CPU fallback (demo server, laptops): the nano model at 640.
    "cpu": DetectorProfile(weights="yolo11n.pt", imgsz=640, batch=4, half=False),
}


@dataclass
class PerceptionCfg:
    analysis_fps_gpu: float = 8.0     # Part A detection rate on GPU
    analysis_fps_cpu: float = 5.0     # Part A detection rate on CPU
    conf_min: float = 0.10            # keep low-score boxes for the second ByteTrack stage
    background_samples: int = 120     # frames kept (downscaled) for the background model
    background_width: int = 640
    budget_share: float = 1.6         # Part A may use this many x video duration before stride grows


@dataclass
class TrackerCfg:
    high_thresh: float = 0.45
    new_track_thresh: float = 0.50
    match_iou: float = 0.20           # first-stage minimum IoU
    low_match_iou: float = 0.45       # second-stage (low-score boxes) minimum IoU
    center_gate: float = 1.6          # fallback: centre distance / size (new tracks have no velocity yet)
    center_gate_static: float = 0.5   # fallback gate for an established track that is not moving
    young_hits: int = 6               # tracks with fewer hits use the full centre gate
    max_lost: float = 2.5             # s a moving track survives without detections
    max_lost_static: float = 12.0     # s a stationary track survives (occlusion by passing traffic)
    min_hits: int = 3


@dataclass
class KinematicsCfg:
    smooth_window: float = 1.0        # s, local-linear fit window for position and velocity
    stationary_speed: float = 0.12    # sizes/s
    moving_speed: float = 0.45        # sizes/s
    stitch_gap: float = 2.5           # s, moving fragments
    stitch_gap_static: float = 20.0   # s, stationary fragments at the same place
    min_track_duration: float = 0.8   # s


@dataclass
class SceneCfg:
    grid_w: int = 64
    grid_h: int = 36
    dir_bins: int = 16
    road_min_vehicles: float = 3.0    # distinct vehicles per cell to call it carriageway
    oriented_min_vehicles: float = 8.0
    oriented_min_purity: float = 0.80  # share of votes within +-1 bin of the mode


@dataclass
class EventCfg:
    min_size: float = 0.018           # objects smaller than this share of the image width are too noisy
                                      # for acceleration-based rules (accident, near miss)
    # accident
    acc_contact_gap: float = 0.15     # ground-contact distance / size
    acc_min_prior_speed: float = 0.8  # sizes/s before impact for at least one party
    acc_min_decel: float = 1.2        # sizes/s^2 drop around contact
    acc_rest_speed: float = 0.25
    acc_min_history: float = 2.0      # s a party must be tracked before contact
    acc_max_jump: float = 0.45        # sizes; larger unexplained box jumps are identity switches
    acc_max_len: float = 20.0
    # near miss
    nm_ttc: float = 2.0               # s, predicted time to closest approach when evasion starts
    nm_min_gap: float = 0.45          # predicted closest approach / size (+0.3 tolerance)
    nm_brake: float = 1.6             # sizes/s^2
    nm_swerve_deg: float = 35.0       # heading change within 1 s
    nm_follow_gap: float = 0.6        # same-direction pairs must come this close (sizes)
    # wrong way
    ww_angle: float = 125.0           # deg against the lane direction
    ww_min_duration: float = 1.5
    ww_min_speed: float = 0.6
    # u-turn / illegal turn
    ut_min_turn: float = 150.0        # deg of cumulative heading change
    ut_max_duration: float = 20.0
    it_min_movements: int = 40        # prior movements needed before rare turns are flagged
    it_max_share: float = 0.01
    # stopped vehicle / congestion
    sv_min_duration: float = 10.0
    sv_signal_zone_duration: float = 150.0  # longer than any red phase
    cg_min_duration: float = 40.0
    cg_min_vehicles: int = 5
    cg_slow_share: float = 0.8
    cg_slow_speed: float = 0.35
    # pedestrians
    jw_min_duration: float = 1.2
    jw_road_erode: int = 1            # cells
    fy_min_speed: float = 0.5
    fy_gap: float = 1.5               # pedestrian within this many vehicle sizes (~a lane) of the vehicle
    fy_ped_speed: float = 0.3         # sizes/s: the pedestrian is walking, not waiting
    # lines and signals
    rl_min_speed: float = 0.8
    sl_min_stop: float = 3.0
    slc_min_depth: float = 0.25       # share of the vehicle width past the line
    # obstacle / fire
    ob_min_duration: float = 5.0
    ob_min_area: float = 0.0006       # share of the image
    fire_min_duration: float = 2.0
    # segments
    merge_gap: float = 1.0
    min_duration: float = 0.8


@dataclass
class RiskCfg:
    analysis_fps_gpu: float = 6.0
    analysis_fps_cpu: float = 3.0
    horizon: float = 5.0
    ema: float = 0.5                  # smoothing of the fused score (per analysed frame)
    ttc_scale: float = 1.4            # s; risk halves roughly every ttc_scale beyond the knee
    hold: float = 1.5                 # s a high score is held after its cue disappears
    min_fps: float = 1.5              # the time guard never analyses fewer frames per second
    guard_every: float = 3.0          # s of video between time-guard checks


@dataclass
class Config:
    perception: PerceptionCfg = field(default_factory=PerceptionCfg)
    tracker: TrackerCfg = field(default_factory=TrackerCfg)
    kin: KinematicsCfg = field(default_factory=KinematicsCfg)
    scene: SceneCfg = field(default_factory=SceneCfg)
    events: EventCfg = field(default_factory=EventCfg)
    risk: RiskCfg = field(default_factory=RiskCfg)
    # classes we report; a class predicted but absent from the test set costs a zero in the macro mean
    enabled: tuple = ("accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
                      "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
                      "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke")


CFG = Config()
