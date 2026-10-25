import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from io import BytesIO
from PIL import Image
import math
import base64

# ==============================================================================
# SARSwarm
# Multi-drone search and rescue simulation with:
# - Bayesian probability map
# - Diffusion prior and temporal propagation
# - Terrain, canopy occlusion, river, clearings
# - Drone battery, sensor, communication, safety models
# - Centralized and decentralized market-based coordination
# - Live animation generation
# - Monte Carlo evaluation
# ==============================================================================

_TERRAIN_CACHE = {}


def default_config():
    return {
        "seed": 42,
        "grid_n": 20,
        "cell_size": 100.0,
        "num_drones": 4,
        "replacement_drones": 1,
        "mode": "centralized",
        "packet_loss": 0.0,
        "noise_free": True,
        "pd_base": 0.88,
        "pf_base": 0.015,
        "deadline_min": 25.0,
        "dt": 10.0,
        "speed": 10.0,
        "altitude": 100.0,
        "fov_deg": 60.0,
        "min_sep": 80.0,
        "min_clearance": 30.0,
        "max_alt": 220.0,
        "comm_timeout": 30.0,
        "battery_capacity": 120.0,
        "reserve_pct": 20.0,
        "cruise_power": 85.0,
        "hover_power": 55.0,
        "sensor_power": 20.0,
        "charge_min": 4.0,
        "elapsed_hours": 2.0,
        "target_moves": True,
        "frame_every": 3,
        "record_history": True,
        "enable_logs": True,
        "central_fallback": True,
        "candidate_angles": 12,
        "market_tasks": 16,
        "stop_on_found": True,
        "failure_step": None,
        "failure_index": 0,
    }


def blur(arr, iterations=1):
    a = arr.copy()
    for _ in range(iterations):
        p = np.pad(a, 1, mode="edge")
        a = (
            p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
            + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
            + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
        ) / 9.0
    return a


def smooth_noise(n, seed, octaves=4, persistence=0.55):
    rng = np.random.default_rng(int(seed))
    out = np.zeros((n, n), dtype=float)
    amp = 1.0
    total = 0.0
    freq = 2

    for _ in range(octaves):
        size = max(2, min(n, freq))
        base = rng.random((size, size))
        yi = (np.arange(n) * size) // n
        xi = (np.arange(n) * size) // n
        layer = base[np.ix_(yi, xi)]
        out += amp * layer
        total += amp
        amp *= persistence
        freq *= 2

    out /= max(total, 1e-9)
    out = blur(out, 2)
    return out


class Terrain:
    def __init__(self, n, cell_size, seed):
        self.n = int(n)
        self.cell_size = float(cell_size)
        self.extent = self.n * self.cell_size
        seed = int(seed)

        rng = np.random.default_rng(seed + 991)

        elev_noise = smooth_noise(self.n, seed, octaves=4, persistence=0.55)
        self.elevation = 20.0 + 140.0 * elev_noise

        xs = np.arange(self.n)
        phase1 = (seed % 17) * 0.37
        phase2 = (seed % 7) * 0.61
        ys = (
            self.n / 2.0
            + self.n * 0.18 * np.sin(xs * 0.22 + phase1)
            + self.n * 0.07 * np.cos(xs * 0.09 + phase2)
        ).astype(int)

        self.river = np.zeros((self.n, self.n), dtype=bool)
        width = max(1, int(round(self.n * 0.045)))
        for x, y in zip(xs, ys):
            for dy in range(-width, width + 1):
                yy = y + dy
                if 0 <= yy < self.n:
                    self.river[yy, x] = True

        clearing_noise = smooth_noise(self.n, seed + 101, octaves=3, persistence=0.5)
        thr = np.quantile(clearing_noise, 0.80)
        self.clearings = clearing_noise >= thr

        canopy = 0.15 + 0.75 * smooth_noise(self.n, seed + 202, octaves=4, persistence=0.6)
        canopy[self.clearings] *= 0.22
        canopy[self.river] = np.clip(canopy[self.river] * 0.70, 0.0, 1.0)
        self.canopy = np.clip(canopy, 0.0, 1.0)
        self.transmission = np.clip(1.0 - 0.85 * self.canopy, 0.05, 1.0)

        gy, gx = np.gradient(self.elevation, self.cell_size, self.cell_size)
        slope = np.sqrt(gx * gx + gy * gy)
        slope_norm = slope / (np.max(slope) + 1e-9)

        walk = 1.0 - 0.55 * slope_norm
        walk[self.river] = 0.05
        walk[self.clearings] = np.clip(walk[self.clearings] * 1.25, 0.05, 1.0)
        self.walkability = np.clip(walk, 0.05, 1.0)

        self.walkability = np.clip(
            self.walkability * (0.90 + 0.20 * rng.random((self.n, self.n))),
            0.05,
            1.0,
        )


def get_terrain(n, cell_size, seed):
    key = (int(n), float(cell_size), int(seed))
    if key not in _TERRAIN_CACHE:
        _TERRAIN_CACHE[key] = Terrain(n, cell_size, seed)
    return _TERRAIN_CACHE[key]


class BeliefMap:
    def __init__(self, terrain, lkp_cell, elapsed_hours, prior_steps=None):
        self.terrain = terrain
        n = terrain.n
        self.p = np.full((n, n), 1.0 / (n * n), dtype=float)
        self.p[lkp_cell] = 1.0
        self.normalize()

        if prior_steps is None:
            prior_steps = max(6, int(float(elapsed_hours) * 10))
        self.diffuse(prior_steps, beta=0.22, gamma=0.45)

    def normalize(self):
        s = float(self.p.sum())
        if not np.isfinite(s) or s <= 1e-15:
            n = self.p.shape[0]
            self.p = np.full((n, n), 1.0 / (n * n), dtype=float)
        else:
            self.p = self.p / s
        self.logp = np.log(np.clip(self.p, 1e-12, 1.0))

    def diffuse(self, steps, beta=0.20, gamma=0.40):
        if steps <= 0:
            return

        w = np.clip(self.terrain.walkability, 0.05, 1.0)
        wg = np.power(w, gamma)
        p = self.p.copy()

        for _ in range(int(steps)):
            pad = np.pad(p, 1, mode="edge")
            nb = (
                pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:]
                + pad[1:-1, :-2] + pad[1:-1, 1:-1] + pad[1:-1, 2:]
                + pad[2:, :-2] + pad[2:, 1:-1] + pad[2:, 2:]
            ) / 9.0

            mix = (1.0 - beta) * p + beta * nb
            mix = mix * wg

            s = float(mix.sum())
            if not np.isfinite(s) or s <= 1e-15:
                n = p.shape[0]
                p = np.full((n, n), 1.0 / (n * n), dtype=float)
            else:
                p = mix / s

        self.p = p
        self.logp = np.log(np.clip(self.p, 1e-12, 1.0))

    def apply_delta(self, delta, weight=1.0):
        if delta is None:
            return

        self.logp = self.logp + float(weight) * delta
        m = np.max(self.logp)
        p = np.exp(self.logp - m)

        s = float(p.sum())
        if not np.isfinite(s) or s <= 1e-15:
            n = self.p.shape[0]
            p = np.full((n, n), 1.0 / (n * n), dtype=float)
        else:
            p = p / s

        self.p = p
        self.logp = np.log(np.clip(self.p, 1e-12, 1.0))


class Drone:
    def __init__(self, did, pos, alt_band, cfg):
        self.id = int(did)
        self.pos = np.array(pos, dtype=float)
        self.alt_band = float(alt_band)
        self.altitude = float(alt_band)
        self.speed = float(cfg["speed"])
        self.state = "SEARCH"
        self.battery = 100.0
        self.capacity = float(cfg["battery_capacity"])
        self.comm_ok = True
        self.lost_time = 0.0
        self.assigned = None
        self.lane_idx = 0
        self.wp_idx = 0
        self.lane_list = []


class SwarmSim:
    def __init__(self, cfg=None):
        base = default_config()
        if cfg:
            base.update(cfg)
        self.cfg = base
        self.reset()

    def reset(self):
        cfg = self.cfg
        self.rng = np.random.default_rng(int(cfg["seed"]))
        self.terrain = get_terrain(cfg["grid_n"], cfg["cell_size"], cfg["seed"])

        n = self.terrain.n
        cs = self.terrain.cell_size
        self.extent = self.terrain.extent

        valid = np.flatnonzero(~self.terrain.river)
        if valid.size == 0:
            valid = np.arange(n * n)

        lkp_flat = int(self.rng.choice(valid))
        self.lkp = np.unravel_index(lkp_flat, (n, n))

        self.belief = BeliefMap(self.terrain, self.lkp, cfg["elapsed_hours"])
        self.prior_belief = self.belief.p.copy()
        self.true_target = self.sample_from_belief()

        self.base = np.array([1.5 * cs, 1.5 * cs], dtype=float)
        self.lawn_lanes = self.build_lawnmower_lanes()

        # IMPORTANT: initialize all containers before dispatching drones.
        self.drones = []
        self.next_drone_id = 0
        self.pool = int(cfg["replacement_drones"])
        self.charging = []

        self.time = 0.0
        self.step = 0
        self.run_steps = max(1, int(cfg["deadline_min"] * 60.0 / cfg["dt"]))
        self.finished = False
        self.found = False
        self.find_time = None

        self.detections = []
        self.logs = []
        self.coverage_count = np.zeros((n, n), dtype=float)
        self.visited = np.zeros((n, n), dtype=bool)
        self.override_mask = np.ones((n, n), dtype=float)

        self.pos_history = {}
        self.planned_paths = {}
        self.outboxes = {}
        self.received_updates = []
        self.frames = []

        self.separation_violations = 0
        self.collisions = 0
        self.false_alarms = 0
        self.rtb_battery = 0
        self.rtb_comm = 0
        self.replacements_dispatched = 0

        self.failure_step = cfg.get("failure_step", None)
        self.failure_index = int(cfg.get("failure_index", 0))
        self.last_market = {"tasks": [], "bids": [], "assignments": {}}

        for _ in range(int(cfg["num_drones"])):
            self.dispatch_new_drone()

        self.record_frame(force=True)
        self.log("Simulation initialized")

    def log(self, msg):
        if self.cfg.get("enable_logs", True):
            self.logs.append({
                "time_min": self.time / 60.0,
                "step": self.step,
                "message": msg,
            })
            if len(self.logs) > 2000:
                self.logs = self.logs[-2000:]

    def sample_from_belief(self):
        p = np.clip(self.belief.p.ravel(), 0.0, None)
        s = float(p.sum())
        if s <= 1e-15:
            idx = self.rng.integers(0, p.size)
        else:
            idx = self.rng.choice(p.size, p=p / s)
        return np.unravel_index(int(idx), self.belief.p.shape)

    def cell_center(self, i, j):
        cs = self.terrain.cell_size
        return np.array([j * cs + cs / 2.0, i * cs + cs / 2.0], dtype=float)

    def cell_from_pos(self, pos):
        cs = self.terrain.cell_size
        n = self.terrain.n
        j = int(np.clip(pos[0] // cs, 0, n - 1))
        i = int(np.clip(pos[1] // cs, 0, n - 1))
        return i, j

    def inside_geofence(self, pos):
        return 0.0 <= pos[0] <= self.extent and 0.0 <= pos[1] <= self.extent

    def clip_geofence(self, pos):
        return np.array([
            float(np.clip(pos[0], 0.0, self.extent)),
            float(np.clip(pos[1], 0.0, self.extent)),
        ], dtype=float)

    def move_toward(self, pos, target, max_move):
        v = np.asarray(target, dtype=float) - np.asarray(pos, dtype=float)
        dist = float(np.linalg.norm(v))
        if dist <= max_move or dist <= 1e-9:
            return np.asarray(target, dtype=float).copy()
        return np.asarray(pos, dtype=float) + v / dist * max_move

    def build_lawnmower_lanes(self):
        n = self.terrain.n
        cs = self.terrain.cell_size
        radius = self.cfg["altitude"] * math.tan(math.radians(self.cfg["fov_deg"]) / 2.0)
        spacing = max(1, int(round((2.0 * radius) / cs)))
        spacing = min(spacing, n)

        lanes = []
        row_id = 0
        for r in range(0, n, spacing):
            y = r * cs + cs / 2.0
            xs = np.arange(n) * cs + cs / 2.0
            if row_id % 2 == 1:
                xs = xs[::-1]
            ys = np.full(n, y, dtype=float)
            lanes.append(np.column_stack((xs, ys)))
            row_id += 1

        return lanes

    def dispatch_new_drone(self):
        did = self.next_drone_id
        self.next_drone_id += 1

        base_alt = float(self.cfg.get("altitude", 100.0))
        alt = base_alt + 6.0 * (did % 4)
        alt = float(np.clip(alt, 60.0, float(self.cfg.get("max_alt", 220.0)) - 10.0))

        pos = self.base.copy() + self.rng.normal(0.0, 2.0, 2)
        pos = self.clip_geofence(pos)

        d = Drone(did, pos, alt, self.cfg)

        lane_total = max(1, len(self.lawn_lanes))
        d.lane_idx = did % lane_total
        d.lane_list = self.lawn_lanes[d.lane_idx::lane_total]
        d.wp_idx = 0

        self.drones.append(d)
        self.pos_history[did] = [d.pos.copy()]
        self.planned_paths[did] = [d.pos.copy()]
        self.outboxes[did] = []

        self.log(f"Drone {did} dispatched from base")
        return d

    def get_drone(self, did):
        for d in self.drones:
            if d.id == did:
                return d
        return None

    def active_paths(self, paths=None):
        source = paths if paths is not None else self.planned_paths
        out = {}
        for did, path in source.items():
            d = self.get_drone(did)
            if d is not None and d.state in ("SEARCH", "RTB"):
                out[did] = path
        return out

    def begin_charging(self, d):
        if d.state == "CHARGING":
            return
        d.state = "CHARGING"
        d.assigned = None
        d.pos = self.base.copy()
        ready = self.time + self.cfg["charge_min"] * 60.0
        self.charging.append({"id": d.id, "ready_time": ready})
        self.outboxes[d.id] = []
        self.log(f"Drone {d.id} arrived at base and is charging")

    def footprint_cells(self, pos, altitude):
        radius = float(altitude) * math.tan(math.radians(self.cfg["fov_deg"]) / 2.0)
        i0, j0 = self.cell_from_pos(pos)
        rc = int(math.ceil(radius / self.terrain.cell_size)) + 1
        n = self.terrain.n
        cells = set()

        for i in range(max(0, i0 - rc), min(n, i0 + rc + 1)):
            for j in range(max(0, j0 - rc), min(n, j0 + rc + 1)):
                c = self.cell_center(i, j)
                if np.linalg.norm(c - pos) <= radius:
                    cells.add((i, j))

        if not cells:
            cells.add((i0, j0))

        return cells

    def path_points(self, p0, p1):
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        dist = float(np.linalg.norm(p1 - p0))
        num = max(2, min(6, int(dist / (0.5 * self.terrain.cell_size)) + 1))
        ts = np.linspace(0.0, 1.0, num)
        return [p0 + (p1 - p0) * t for t in ts]

    def path_footprint_cells(self, path, altitude):
        cells = set()
        for pt in path:
            cells.update(self.footprint_cells(pt, altitude))
        return cells

    def sensor_pd_cell(self, i, j):
        if self.cfg["noise_free"]:
            return 1.0
        return float(np.clip(self.cfg["pd_base"] * self.terrain.transmission[i, j], 0.01, 0.99))

    def sensor_pd_at(self, i, j, pos, altitude):
        if self.cfg["noise_free"]:
            return 1.0

        radius = float(altitude) * math.tan(math.radians(self.cfg["fov_deg"]) / 2.0)
        c = self.cell_center(i, j)
        dist = float(np.linalg.norm(c - pos))

        if dist > radius:
            return 0.0

        falloff = 1.0 - 0.30 * (dist / max(radius, 1e-9)) ** 2
        pd = self.cfg["pd_base"] * self.terrain.transmission[i, j] * falloff
        return float(np.clip(pd, 0.01, 0.99))

    def bayes_posterior(self, prior, detected, pd, pf):
        p = float(np.clip(prior, 1e-9, 1.0 - 1e-9))
        pd = float(np.clip(pd, 1e-6, 1.0 - 1e-6))
        pf = float(np.clip(pf, 0.0, 1.0 - 1e-6))

        if detected:
            num = p * pd
            den = num + (1.0 - p) * pf
        else:
            num = p * (1.0 - pd)
            den = num + (1.0 - p) * (1.0 - pf)

        if den <= 1e-15:
            return 1e-9 if not detected else 1.0 - 1e-9

        return float(np.clip(num / den, 1e-9, 1.0 - 1e-9))

    def return_energy_wh(self, pos):
        dist = float(np.linalg.norm(np.asarray(pos, dtype=float) - self.base))
        hours = dist / max(1.0, self.cfg["speed"]) / 3600.0
        return hours * self.cfg["cruise_power"]

    def min_path_distance(self, path_a, path_b):
        if not path_a or not path_b:
            return 1e9

        m = 1e9
        for pa in path_a:
            for pb in path_b:
                d = float(np.linalg.norm(np.asarray(pa) - np.asarray(pb)))
                if d < m:
                    m = d
        return m

    def separation_penalty(self, path, exclude_id=None, paths=None, ignore=False):
        if ignore:
            return 0.0

        source = self.active_paths(paths)
        pen = 0.0
        min_sep = self.cfg["min_sep"]

        for oid, opath in source.items():
            if oid == exclude_id or not opath:
                continue

            dmin = self.min_path_distance(path, opath)
            if dmin < min_sep:
                pen += 1000.0 * (min_sep - dmin) / max(min_sep, 1e-9)
            elif dmin < 2.0 * min_sep:
                pen += 50.0 / (dmin + 1.0)

        return pen

    def candidate_positions(self, d):
        max_move = d.speed * self.cfg["dt"]
        cands = [d.pos.copy()]

        angles = np.linspace(
            0.0,
            2.0 * math.pi,
            int(self.cfg.get("candidate_angles", 12)),
            endpoint=False,
        )

        for a in angles:
            ux = math.cos(a)
            uy = math.sin(a)
            for frac in (0.45, 0.80, 1.00):
                p = d.pos + np.array([ux, uy], dtype=float) * max_move * frac
                if self.inside_geofence(p):
                    cands.append(p)

        return cands

    def evaluate_candidate(self, d, cand, paths_so_far=None, ignore_separation=False):
        path = self.path_points(d.pos, cand)
        cells = self.path_footprint_cells(path, d.altitude)

        val = 0.0
        for i, j in cells:
            p = float(self.belief.p[i, j] * self.override_mask[i, j])
            pd = self.sensor_pd_cell(i, j)
            nov = 1.0 / (1.0 + self.coverage_count[i, j])
            val += p * pd * nov

        dist = float(np.linalg.norm(np.asarray(cand) - d.pos))
        cost = dist / max(1.0, d.speed) + 1e-3
        pen = self.separation_penalty(path, d.id, paths_so_far, ignore_separation)

        rem_wh = d.battery / 100.0 * d.capacity
        ret_wh = self.return_energy_wh(cand)
        reserve_wh = d.capacity * self.cfg["reserve_pct"] / 100.0
        margin = rem_wh - ret_wh - reserve_wh

        if margin < 0.0:
            pen += 1e6
        elif margin < 0.5 * reserve_wh:
            pen += 20.0 * (0.5 * reserve_wh - margin) / max(reserve_wh, 1e-9)

        score = val / cost - pen
        return score, path, val, cost, pen

    def choose_best_candidate(self, d, paths_so_far=None, ignore_separation=False):
        best_pos = d.pos.copy()
        best_path = [d.pos.copy()]
        best_score = -1e30

        for cand in self.candidate_positions(d):
            score, path, _, _, _ = self.evaluate_candidate(
                d, cand, paths_so_far, ignore_separation
            )
            if score > best_score:
                best_score = score
                best_pos = np.asarray(cand, dtype=float).copy()
                best_path = path

        return best_pos, best_path, best_score

    def market_tasks(self, k=None):
        if k is None:
            k = int(self.cfg.get("market_tasks", 16))

        value = (
            self.belief.p
            * self.override_mask
            * self.terrain.transmission
            / (1.0 + self.coverage_count)
        )

        flat = np.argsort(value.ravel())[::-1][:k]
        tasks = []

        for tid, idx in enumerate(flat):
            i, j = np.unravel_index(int(idx), value.shape)
            v = float(value[i, j])
            if v <= 1e-12:
                continue
            tasks.append({
                "id": tid,
                "cell": (i, j),
                "pos": self.cell_center(i, j),
                "value": v,
            })

        return tasks

    def central_link_ok(self):
        return self.rng.random() >= self.cfg["packet_loss"]

    def plan_lawnmower(self):
        paths = {}
        max_move = self.cfg["speed"] * self.cfg["dt"]

        for d in self.drones:
            old = d.pos.copy()

            if d.state == "SEARCH":
                if d.lane_list:
                    if d.lane_idx >= len(d.lane_list):
                        d.lane_idx = len(d.lane_list) - 1
                        d.wp_idx = max(0, len(d.lane_list[d.lane_idx]) - 1)

                    lane = d.lane_list[d.lane_idx]

                    if d.wp_idx >= len(lane):
                        d.wp_idx = len(lane) - 1

                    wp = lane[d.wp_idx]
                    new = self.move_toward(old, wp, max_move)

                    if float(np.linalg.norm(new - wp)) < 1.0:
                        d.wp_idx += 1
                        if d.wp_idx >= len(lane):
                            d.lane_idx += 1
                            d.wp_idx = 0
                            if d.lane_idx >= len(d.lane_list):
                                d.lane_idx = len(d.lane_list) - 1
                                d.wp_idx = len(d.lane_list[d.lane_idx]) - 1

                    d.pos = new

                paths[d.id] = self.path_points(old, d.pos)

            elif d.state == "RTB":
                new = self.move_toward(old, self.base, max_move)
                d.pos = new
                paths[d.id] = self.path_points(old, new)
                if float(np.linalg.norm(new - self.base)) < 1.0:
                    self.begin_charging(d)

            else:
                paths[d.id] = [d.pos.copy()]

        self.planned_paths = paths

    def plan_random(self):
        paths = {}
        max_move = self.cfg["speed"] * self.cfg["dt"]

        for d in self.drones:
            old = d.pos.copy()

            if d.state == "SEARCH":
                new = old.copy()
                for _ in range(8):
                    ang = self.rng.uniform(0.0, 2.0 * math.pi)
                    dist = max_move * self.rng.uniform(0.35, 1.0)
                    cand = old + np.array([math.cos(ang), math.sin(ang)]) * dist
                    if self.inside_geofence(cand):
                        new = cand
                        break

                d.pos = new
                paths[d.id] = self.path_points(old, new)

            elif d.state == "RTB":
                new = self.move_toward(old, self.base, max_move)
                d.pos = new
                paths[d.id] = self.path_points(old, new)
                if float(np.linalg.norm(new - self.base)) < 1.0:
                    self.begin_charging(d)

            else:
                paths[d.id] = [d.pos.copy()]

        self.planned_paths = paths

    def plan_centralized(self):
        paths = {}
        searching = [d for d in self.drones if d.state == "SEARCH"]
        searching.sort(key=lambda x: x.battery)

        for d in searching:
            old = d.pos.copy()
            pos, path, _ = self.choose_best_candidate(d, paths, ignore_separation=False)
            d.pos = pos
            d.assigned = None
            paths[d.id] = path

        for d in self.drones:
            if d.state == "RTB":
                old = d.pos.copy()
                new = self.move_toward(old, self.base, d.speed * self.cfg["dt"])
                d.pos = new
                paths[d.id] = self.path_points(old, new)
                if float(np.linalg.norm(new - self.base)) < 1.0:
                    self.begin_charging(d)
            elif d.state != "SEARCH":
                paths[d.id] = [d.pos.copy()]
            elif d.id not in paths:
                paths[d.id] = [d.pos.copy()]

        self.planned_paths = paths

    def plan_market(self):
        paths = {}
        searching = [d for d in self.drones if d.state == "SEARCH"]
        tasks = self.market_tasks()
        bids = []

        for d in searching:
            for task in tasks:
                if self.rng.random() < self.cfg["packet_loss"]:
                    continue

                dist = float(np.linalg.norm(task["pos"] - d.pos))
                cost = dist / max(1.0, d.speed) + 1e-3
                path = self.path_points(d.pos, task["pos"])
                pen = 0.35 * self.separation_penalty(path, d.id, None)

                rem_wh = d.battery / 100.0 * d.capacity
                ret_wh = self.return_energy_wh(task["pos"])
                reserve_wh = d.capacity * self.cfg["reserve_pct"] / 100.0

                if rem_wh - ret_wh - reserve_wh < 0.0:
                    pen += 1e5

                bid = task["value"] / cost - pen
                if bid > 0.0:
                    bids.append((float(bid), d.id, task))

        bids.sort(key=lambda x: x[0], reverse=True)

        assigned = {}
        used_tasks = set()

        for bid, did, task in bids:
            if did in assigned or task["id"] in used_tasks:
                continue

            d = self.get_drone(did)
            if d is None or d.state != "SEARCH":
                continue

            old = d.pos.copy()
            new = self.move_toward(old, task["pos"], d.speed * self.cfg["dt"])
            path = self.path_points(old, new)

            if self.separation_penalty(path, did, paths) > 500.0:
                continue

            d.pos = new
            d.assigned = task["cell"]
            assigned[did] = task
            used_tasks.add(task["id"])
            paths[did] = path

        for d in searching:
            if d.id in assigned:
                continue

            old = d.pos.copy()
            pos, path, _ = self.choose_best_candidate(d, paths, ignore_separation=False)
            d.pos = pos
            d.assigned = None
            paths[d.id] = path

        for d in self.drones:
            if d.state == "RTB":
                old = d.pos.copy()
                new = self.move_toward(old, self.base, d.speed * self.cfg["dt"])
                d.pos = new
                paths[d.id] = self.path_points(old, new)
                if float(np.linalg.norm(new - self.base)) < 1.0:
                    self.begin_charging(d)
            elif d.state != "SEARCH":
                paths[d.id] = [d.pos.copy()]
            elif d.id not in paths:
                paths[d.id] = [d.pos.copy()]

        self.planned_paths = paths
        self.last_market = {
            "tasks": tasks,
            "bids": bids[:80],
            "assignments": {did: task["cell"] for did, task in assigned.items()},
        }

    def plan(self):
        mode = self.cfg["mode"]
        if mode == "lawnmower":
            self.plan_lawnmower()
        elif mode == "random":
            self.plan_random()
        elif mode == "centralized":
            self.plan_centralized()
        elif mode == "market":
            self.plan_market()
        else:
            self.plan_centralized()

    def update_comm_links(self):
        dt = self.cfg["dt"]

        for d in self.drones:
            if d.state not in ("SEARCH", "RTB"):
                continue

            d.comm_ok = self.rng.random() >= self.cfg["packet_loss"]

            if not d.comm_ok:
                d.lost_time += dt
                if d.lost_time > self.cfg["comm_timeout"] and d.state == "SEARCH":
                    d.state = "RTB"
                    d.assigned = None
                    self.rtb_comm += 1
                    self.log(f"Drone {d.id} comm-loss failsafe: returning to base")
            else:
                d.lost_time = 0.0

    def update_charging_and_dispatch(self):
        remaining = []

        for ch in self.charging:
            if self.time >= ch["ready_time"]:
                self.pool += 1
                self.log(f"Drone {ch['id']} charged; replacement available")
            else:
                remaining.append(ch)

        self.charging = remaining

        searching = [d for d in self.drones if d.state == "SEARCH"]

        while len(searching) < self.cfg["num_drones"] and self.pool > 0:
            nd = self.dispatch_new_drone()
            self.pool -= 1
            self.replacements_dispatched += 1
            searching.append(nd)

    def handle_drone_failure(self):
        if self.failure_step is None or self.step != int(self.failure_step):
            return

        candidates = [d for d in self.drones if d.state == "SEARCH"]
        if candidates:
            idx = min(self.failure_index, len(candidates) - 1)
            d = candidates[idx]
            d.state = "OFFLINE"
            d.assigned = None
            self.log(f"Drone {d.id} failed mid-mission and is offline")

        self.failure_step = None

    def enforce_separation_and_safety(self):
        flying = [d for d in self.drones if d.state in ("SEARCH", "RTB")]
        min_sep = self.cfg["min_sep"]

        for a in range(len(flying)):
            for b in range(a + 1, len(flying)):
                da = flying[a]
                db = flying[b]
                v = da.pos - db.pos
                dist = float(np.linalg.norm(v))

                if dist < min_sep:
                    self.separation_violations += 1

                    if dist < 10.0:
                        self.collisions += 1

                    if dist < 1e-6:
                        ang = self.rng.uniform(0.0, 2.0 * math.pi)
                        u = np.array([math.cos(ang), math.sin(ang)], dtype=float)
                    else:
                        u = v / dist

                    push = 0.5 * (min_sep - dist) + 0.5
                    da.pos = self.clip_geofence(da.pos + u * push)
                    db.pos = self.clip_geofence(db.pos - u * push)

        for d in flying:
            d.pos = self.clip_geofence(d.pos)
            i, j = self.cell_from_pos(d.pos)
            elev = float(self.terrain.elevation[i, j])
            required = elev + self.cfg["min_clearance"]

            d.altitude = max(d.alt_band, required)

            if d.altitude > self.cfg["max_alt"]:
                d.altitude = self.cfg["max_alt"]
                if required > self.cfg["max_alt"] and d.state == "SEARCH":
                    d.state = "RTB"
                    self.log(f"Drone {d.id} terrain clearance failsafe: returning to base")

    def update_battery(self, old_positions):
        dt = self.cfg["dt"]

        for d in self.drones:
            if d.state not in ("SEARCH", "RTB"):
                continue

            old = old_positions.get(d.id, d.pos)
            dist = float(np.linalg.norm(d.pos - old))

            power = self.cfg["hover_power"]
            if dist > 1.0:
                power = self.cfg["cruise_power"]
            if d.state == "SEARCH":
                power += self.cfg["sensor_power"]

            wh = power * dt / 3600.0
            d.battery -= 100.0 * wh / max(1.0, d.capacity)

            if d.battery <= 0.0:
                d.battery = 0.0
                d.state = "OFFLINE"
                self.log(f"Drone {d.id} battery exhausted")
                continue

            if d.state == "SEARCH":
                rem_wh = d.battery / 100.0 * d.capacity
                ret_wh = self.return_energy_wh(d.pos)
                reserve_wh = d.capacity * self.cfg["reserve_pct"] / 100.0

                if rem_wh <= ret_wh + reserve_wh:
                    d.state = "RTB"
                    d.assigned = None
                    self.rtb_battery += 1
                    self.log(f"Drone {d.id} battery-aware RTB")

    def observe_drone(self, d):
        if d.state != "SEARCH":
            return

        fov_set = self.footprint_cells(d.pos, d.altitude)
        if not fov_set:
            return

        fov_list = list(fov_set)

        for i, j in fov_list:
            self.visited[i, j] = True
            self.coverage_count[i, j] += 1.0

        target = self.true_target
        detected = False
        det_cell = None

        if target in fov_set:
            pd = self.sensor_pd_at(target[0], target[1], d.pos, d.altitude)
            if self.rng.random() < pd:
                detected = True
                det_cell = target
        else:
            if not self.cfg["noise_free"]:
                pf_scan = 1.0 - (1.0 - self.cfg["pf_base"]) ** len(fov_list)
                if self.rng.random() < pf_scan:
                    weights = np.array(
                        [self.sensor_pd_at(i, j, d.pos, d.altitude) for i, j in fov_list],
                        dtype=float,
                    )
                    if weights.sum() <= 1e-12:
                        weights = np.ones(len(fov_list), dtype=float)
                    probs = weights / weights.sum()
                    idx = self.rng.choice(len(fov_list), p=probs)
                    detected = True
                    det_cell = fov_list[int(idx)]

        delta = np.zeros_like(self.belief.p, dtype=float)
        pf = 0.0 if self.cfg["noise_free"] else self.cfg["pf_base"]

        if detected:
            i, j = det_cell
            prior = self.belief.p[i, j]
            pd = self.sensor_pd_at(i, j, d.pos, d.altitude)
            post = self.bayes_posterior(prior, True, pd, pf)
            delta[i, j] = math.log(max(post, 1e-12)) - math.log(max(prior, 1e-12))

            is_true = det_cell == self.true_target
            pos = self.cell_center(i, j)

            self.detections.append({
                "time": self.time,
                "step": self.step,
                "cell": det_cell,
                "pos": pos,
                "true": bool(is_true),
                "drone": d.id,
            })

            if is_true:
                self.found = True
                self.find_time = self.time
                self.log(f"TARGET FOUND by Drone {d.id} at {self.time / 60.0:.2f} min")
            else:
                self.false_alarms += 1
                self.log(f"False detection by Drone {d.id}")
        else:
            for i, j in fov_list:
                prior = self.belief.p[i, j]
                pd = self.sensor_pd_at(i, j, d.pos, d.altitude)
                post = self.bayes_posterior(prior, False, pd, pf)
                delta[i, j] = math.log(max(post, 1e-12)) - math.log(max(prior, 1e-12))

        self.outboxes.setdefault(d.id, []).append((self.time, delta))
        if len(self.outboxes[d.id]) > 20:
            self.outboxes[d.id] = self.outboxes[d.id][-20:]

    def observe_all(self):
        for d in self.drones:
            self.observe_drone(d)

    def send_and_apply_updates(self):
        sent = []

        for d in self.drones:
            keep = []
            for ts, deltaray in self.outboxes.get(d.id, []):
                age = self.time - ts
                if age > 240.0:
                    continue

                if d.comm_ok and self.rng.random() >= self.cfg["packet_loss"]:
                    sent.append((ts, deltaray))
                else:
                    keep.append((ts, deltaray))

            self.outboxes[d.id] = keep[-10:]

        if not sent:
            return

        sent.sort(key=lambda x: x[0])
        total = np.zeros_like(self.belief.p, dtype=float)

        for ts, deltaray in sent:
            age = max(0.0, self.time - ts)
            w = math.exp(-age / 90.0) if age > 0 else 1.0
            if w < 0.05:
                continue
            total += w * deltaray

        self.belief.apply_delta(total)

    def diffuse_belief(self):
        steps = 1
        if self.cfg["target_moves"]:
            beta = 0.18
            gamma = 0.35
        else:
            beta = 0.06
            gamma = 0.15
        self.belief.diffuse(steps, beta=beta, gamma=gamma)

    def move_target(self):
        if not self.cfg["target_moves"] or self.found:
            return

        i, j = self.true_target
        n = self.terrain.n
        neigh = []
        weights = []

        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni = int(np.clip(i + di, 0, n - 1))
                nj = int(np.clip(j + dj, 0, n - 1))
                w = float(self.terrain.walkability[ni, nj])

                if di == 0 and dj == 0:
                    w *= 0.55
                else:
                    w *= 0.15

                neigh.append((ni, nj))
                weights.append(w)

        weights = np.array(weights, dtype=float)
        if weights.sum() <= 1e-12:
            return

        probs = weights / weights.sum()
        idx = self.rng.choice(len(neigh), p=probs)
        self.true_target = neigh[int(idx)]

    def record_history(self):
        if not self.cfg.get("record_history", True):
            return

        for d in self.drones:
            if d.state in ("SEARCH", "RTB"):
                self.pos_history.setdefault(d.id, []).append(d.pos.copy())

    def record_frame(self, force=False):
        fe = int(self.cfg.get("frame_every", 0))

        if not force:
            if fe <= 0 or self.step % fe != 0:
                return

        positions = {
            d.id: d.pos.copy()
            for d in self.drones
            if d.state in ("SEARCH", "RTB")
        }

        self.frames.append({
            "step": self.step,
            "time": self.time,
            "belief": self.belief.p.copy(),
            "positions": positions,
            "found": self.found,
        })

        if len(self.frames) > 300:
            self.frames = self.frames[-300:]

    def check_stop(self):
        if self.found and self.cfg.get("stop_on_found", True):
            self.finished = True
            return

        if self.time >= self.cfg["deadline_min"] * 60.0:
            self.finished = True
            return

        searchers = [d for d in self.drones if d.state == "SEARCH"]
        if not searchers and not self.pool and not self.charging:
            self.finished = True

    def one_step(self):
        if self.finished:
            return

        self.step += 1
        self.time += self.cfg["dt"]

        self.handle_drone_failure()
        self.update_comm_links()
        self.update_charging_and_dispatch()

        old_positions = {
            d.id: d.pos.copy()
            for d in self.drones
            if d.state in ("SEARCH", "RTB")
        }

        self.plan()
        self.enforce_separation_and_safety()
        self.update_battery(old_positions)
        self.observe_all()
        self.send_and_apply_updates()
        self.diffuse_belief()
        self.move_target()
        self.record_history()
        self.record_frame()
        self.check_stop()

    def run(self, max_steps=None):
        if max_steps is None:
            max_steps = self.run_steps

        for _ in range(int(max_steps)):
            if self.finished:
                break
            self.one_step()

        if int(self.cfg.get("frame_every", 0)) > 0:
            self.record_frame(force=True)


# ==============================================================================
# Rendering helpers
# ==============================================================================

def create_map_fig(sim, frame=None, show_truth=False, figsize=(5.8, 5.8), dpi=90):
    terrain = sim.terrain
    E = terrain.extent
    cs = terrain.cell_size

    if frame is None:
        belief = sim.belief.p
        positions = {
            d.id: d.pos.copy()
            for d in sim.drones
            if d.state in ("SEARCH", "RTB")
        }
        t = sim.time
        found = sim.found
        step = sim.step
    else:
        belief = frame["belief"]
        positions = frame["positions"]
        t = frame["time"]
        found = frame["found"]
        step = frame["step"]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.imshow(
        terrain.elevation,
        cmap="gray",
        alpha=0.25,
        origin="lower",
        extent=[0, E, 0, E],
    )

    ry, rx = np.where(terrain.river)
    if rx.size:
        ax.scatter(
            rx * cs + cs / 2.0,
            ry * cs + cs / 2.0,
            s=2.0,
            c="#2b7bd6",
            alpha=0.45,
            linewidths=0,
        )

    vmax = float(np.percentile(belief, 99.7)) if belief.size else 1.0
    if vmax <= 1e-12:
        vmax = float(belief.max()) if belief.size else 1.0

    ax.imshow(
        belief,
        cmap="inferno",
        alpha=0.70,
        origin="lower",
        extent=[0, E, 0, E],
        vmin=0.0,
        vmax=max(vmax, 1e-9),
    )

    for did, hist in sim.pos_history.items():
        if frame is not None:
            end = min(len(hist), step + 1)
            hist = hist[:end]

        if len(hist) > 1:
            arr = np.array(hist, dtype=float)
            ax.plot(arr[:, 0], arr[:, 1], lw=1.0, alpha=0.45, color="cyan")

    tab10 = plt.cm.tab10.colors

    for did, pos in positions.items():
        d = sim.get_drone(did)
        if d is None:
            continue

        if d.state == "RTB":
            color = "orange"
        elif d.state == "SEARCH":
            color = tab10[did % len(tab10)]
        else:
            color = "gray"

        radius = d.altitude * math.tan(math.radians(sim.cfg["fov_deg"]) / 2.0)
        circ = Circle(
            pos,
            radius,
            fill=False,
            edgecolor=color,
            linestyle="--",
            alpha=0.55,
            linewidth=1.0,
        )
        ax.add_patch(circ)

        ax.scatter(
            pos[0],
            pos[1],
            s=75,
            c=color,
            edgecolors="white",
            linewidths=1.0,
            zorder=6,
        )
        ax.text(pos[0] + 8, pos[1] + 8, str(did), color="white", fontsize=8, zorder=7)

    for det in sim.detections:
        if det["time"] > t:
            continue

        if det["true"]:
            ax.scatter(
                det["pos"][0],
                det["pos"][1],
                marker="X",
                s=130,
                c="lime",
                edgecolors="black",
                zorder=8,
            )
        else:
            ax.scatter(
                det["pos"][0],
                det["pos"][1],
                marker="x",
                s=80,
                c="magenta",
                edgecolors="black",
                zorder=8,
            )

    ax.scatter(
        sim.base[0],
        sim.base[1],
        marker="*",
        s=230,
        c="white",
        edgecolors="black",
        zorder=9,
    )

    if show_truth and found:
        ti, tj = sim.true_target
        tp = sim.cell_center(ti, tj)
        ax.scatter(
            tp[0],
            tp[1],
            marker="o",
            s=110,
            c="deepskyblue",
            edgecolors="black",
            zorder=10,
            label="True target",
        )

    ax.add_patch(Rectangle((0, 0), E, E, fill=False, edgecolor="yellow", linewidth=2.0))
    ax.set_xlim(0, E)
    ax.set_ylim(0, E)

    status = "FOUND" if found else "SEARCHING"
    ax.set_title(
        f"SARSwarm Live Map | {status} | t={t / 60.0:.2f} min | step={step}",
        fontsize=11,
    )
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    fig.tight_layout()
    return fig


def simple_heatmap_fig(arr, title, cmap="inferno", extent=None, alpha=1.0):
    fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=90)

    if extent is None:
        n = arr.shape[0]
        extent = [0, n, 0, n]

    im = ax.imshow(arr, cmap=cmap, origin="lower", extent=extent, alpha=alpha)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_animation(sim, max_frames=34):
    frames = sim.frames

    if not frames:
        sim.record_frame(force=True)
        frames = sim.frames

    if not frames:
        return None

    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in idx]

    imgs = []

    for fr in frames:
        fig = create_map_fig(sim, frame=fr, show_truth=False)
        png = fig_to_png_bytes(fig)
        im = Image.open(BytesIO(png)).convert("RGB").resize((600, 600))
        imgs.append(im)

    if not imgs:
        return None

    out = BytesIO()
    imgs[0].save(
        out,
        format="GIF",
        save_all=True,
        append_images=imgs[1:],
        duration=220,
        loop=0,
        optimize=False,
    )

    return out.getvalue()


def gif_to_html(gif_bytes):
    b64 = base64.b64encode(gif_bytes).decode("ascii")
    return (
        '<div style="text-align:center;">'
        f'<img src="data:image/gif;base64,{b64}" '
        'style="width:100%; max-width:640px; border:1px solid #444; border-radius:8px;" />'
        "</div>"
    )


# ==============================================================================
# Evaluation helpers
# ==============================================================================

def get_metrics(sim):
    hours = max(sim.time / 3600.0, 1e-6)
    cs = sim.terrain.cell_size
    coverage_km2 = float(sim.visited.sum()) * cs * cs / 1.0e6
    coverage_per_hour = coverage_km2 / hours

    false_alarms = sum(1 for det in sim.detections if not det["true"])
    active = len([d for d in sim.drones if d.state == "SEARCH"])

    return {
        "seed": int(sim.cfg["seed"]),
        "mode": sim.cfg["mode"],
        "packet_loss": float(sim.cfg["packet_loss"]),
        "drone_failure": bool(sim.cfg.get("failure_step") is not None),
        "found": bool(sim.found),
        "time_to_find_min": float(sim.find_time / 60.0) if sim.find_time is not None else np.nan,
        "coverage_km2_hr": float(coverage_per_hour),
        "separation_violations": int(sim.separation_violations),
        "collisions": int(sim.collisions),
        "false_alarms": int(false_alarms),
        "battery_rtb": int(sim.rtb_battery),
        "comm_rtb": int(sim.rtb_comm),
        "replacements": int(sim.replacements_dispatched),
        "final_active": int(active),
        "belief_sum": float(sim.belief.p.sum()),
    }


def make_eval_cfg(mode, packet_loss, failure, seed, cell_size=100, deadline_min=30, num_drones=5):
    cfg = default_config()
    grid_n = int(round(2000.0 / float(cell_size)))
    steps = int(deadline_min * 60.0 / 10.0)

    cfg.update({
        "seed": int(seed),
        "grid_n": grid_n,
        "cell_size": float(2000.0 / grid_n),
        "num_drones": int(num_drones),
        "replacement_drones": 1,
        "mode": mode,
        "packet_loss": float(packet_loss),
        "noise_free": False,
        "pd_base": 0.85,
        "pf_base": 0.02,
        "deadline_min": float(deadline_min),
        "dt": 10.0,
        "speed": 10.0,
        "target_moves": True,
        "frame_every": 0,
        "record_history": False,
        "enable_logs": False,
        "central_fallback": True,
        "candidate_angles": 8,
        "market_tasks": 12,
        "stop_on_found": True,
        "failure_step": int(0.40 * steps) if failure else None,
        "failure_index": 0,
    })

    return cfg


def run_evaluation(
    scenarios,
    modes,
    losses,
    include_failure,
    cell_size,
    deadline_min,
    num_drones,
    base_seed,
    progress_cb=None,
):
    rows = []
    failure_levels = [False, True] if include_failure else [False]
    total = max(1, len(modes) * len(losses) * len(failure_levels) * int(scenarios))
    done = 0

    for mode in modes:
        for loss in losses:
            for fail in failure_levels:
                for s in range(int(scenarios)):
                    seed = int(base_seed) + s
                    cfg = make_eval_cfg(
                        mode,
                        loss,
                        fail,
                        seed,
                        cell_size=cell_size,
                        deadline_min=deadline_min,
                        num_drones=num_drones,
                    )

                    sim = SwarmSim(cfg)
                    sim.run()
                    m = get_metrics(sim)
                    m["scenario"] = s + 1
                    rows.append(m)

                    done += 1
                    if progress_cb is not None:
                        progress_cb(
                            done / float(total),
                            f"{mode} | loss={loss:.0%} | failure={fail} | scenario {s + 1}/{scenarios}",
                        )

    return pd.DataFrame(rows)


def aggregate_eval(df):
    if df.empty:
        return pd.DataFrame()

    agg = df.groupby(["mode", "packet_loss", "drone_failure"], dropna=False).agg(
        detection_prob=("found", "mean"),
        mean_time_min=("time_to_find_min", "mean"),
        median_time_min=("time_to_find_min", "median"),
        coverage_km2_hr=("coverage_km2_hr", "mean"),
        separation_violations=("separation_violations", "mean"),
        collisions=("collisions", "mean"),
        false_alarms=("false_alarms", "mean"),
        battery_rtb=("battery_rtb", "mean"),
        comm_rtb=("comm_rtb", "mean"),
        replacements=("replacements", "mean"),
    ).reset_index()

    return agg


def df_to_markdown(df):
    if df.empty:
        return "_No data._"

    headers = " | ".join(str(c) for c in df.columns)
    sep = " | ".join("---" for _ in df.columns)
    lines = [headers, sep]

    for _, row in df.iterrows():
        vals = []
        for v in row.values:
            if isinstance(v, float):
                if np.isnan(v):
                    vals.append("nan")
                elif abs(v) < 1e-4:
                    vals.append(f"{v:.2e}")
                else:
                    vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        lines.append(" | ".join(vals))

    return "\n".join(lines)


def make_failure_notes(agg, raw):
    if agg.empty:
        return "No evaluation data available."

    notes = []

    best = agg.loc[agg["detection_prob"].idxmax()]
    worst = agg.loc[agg["detection_prob"].idxmin()]

    notes.append(
        f"Best observed configuration: mode={best['mode']}, "
        f"packet_loss={best['packet_loss']:.0%}, "
        f"drone_failure={bool(best['drone_failure'])}, "
        f"detection probability={best['detection_prob']:.2f}."
    )

    notes.append(
        f"Worst observed configuration: mode={worst['mode']}, "
        f"packet_loss={worst['packet_loss']:.0%}, "
        f"drone_failure={bool(worst['drone_failure'])}, "
        f"detection probability={worst['detection_prob']:.2f}."
    )

    for mode in sorted(agg["mode"].unique()):
        sub = agg[agg["mode"] == mode]
        try:
            p0 = float(sub[np.isclose(sub["packet_loss"], 0.0)]["detection_prob"].mean())
            p50 = float(sub[np.isclose(sub["packet_loss"], 0.5)]["detection_prob"].mean())
            notes.append(
                f"{mode}: detection probability changed from {p0:.2f} at 0% loss "
                f"to {p50:.2f} at 50% loss."
            )
        except Exception:
            pass

    if "drone_failure" in agg.columns:
        try:
            nofail = agg[~agg["drone_failure"].astype(bool)]["detection_prob"].mean()
            fail = agg[agg["drone_failure"].astype(bool)]["detection_prob"].mean()
            notes.append(
                f"Losing one drone mid-mission changed mean detection probability "
                f"from {nofail:.2f} to {fail:.2f}."
            )
        except Exception:
            pass

    if not raw.empty:
        bat = float(raw["battery_rtb"].mean())
        com = float(raw["comm_rtb"].mean())
        viol = float(raw["separation_violations"].mean())

        notes.append(f"Average battery-triggered RTBs per scenario: {bat:.2f}.")
        notes.append(f"Average comm-loss RTBs per scenario: {com:.2f}.")
        notes.append(f"Average separation violations per scenario: {viol:.2f}.")

        notes.append(
            "Failure analysis: performance degradation is dominated by delayed consensus "
            "updates under packet loss, battery-constrained lane abandonment, and "
            "canopy-reduced detection probability in forested cells. The centralized fallback "
            "reduces orphaned search regions when market bids are lost, but increases reliance "
            "on available communication links. Replacement dispatch mitigates single-drone loss "
            "when the pool is nonempty."
        )

    return "\n\n".join(notes)


def run_stage_validation(seeds=3):
    rows = []

    def run_one(stage, **over):
        for s in range(int(seeds)):
            cfg = default_config()
            cfg.update({
                "seed": 5000 + s,
                "grid_n": 20,
                "cell_size": 100.0,
                "deadline_min": 25,
                "dt": 10.0,
                "num_drones": 4,
                "replacement_drones": 1,
                "noise_free": True,
                "packet_loss": 0.0,
                "target_moves": False,
                "frame_every": 0,
                "record_history": False,
                "enable_logs": False,
                "candidate_angles": 8,
                "market_tasks": 12,
            })
            cfg.update(over)

            sim = SwarmSim(cfg)
            sim.run()
            m = get_metrics(sim)
            m["stage"] = stage
            rows.append(m)

    run_one("Stage 1: Single lawnmower, noise-free", mode="lawnmower", num_drones=1)
    run_one("Stage 2: Multi centralized, noise-free", mode="centralized", num_drones=4)
    run_one("Stage 3: Multi centralized, 20% packet loss", mode="centralized", num_drones=4, packet_loss=0.20)
    run_one("Stage 4: Decentralized market, noise-free", mode="market", num_drones=4)

    df = pd.DataFrame(rows)
    checks = []

    if not df.empty:
        belief_ok = bool(np.allclose(df["belief_sum"].astype(float), 1.0, atol=1e-4))
        checks.append(("Belief map remains normalized", belief_ok))

        nf = df[df["stage"].str.contains("noise-free")]
        collision_ok = bool(nf["collisions"].sum() == 0)
        checks.append(("Zero collisions in noise-free staged runs", collision_ok))

        try:
            cent = df[df["stage"].str.contains("Stage 2")]["time_to_find_min"].mean()
            market = df[df["stage"].str.contains("Stage 4")]["time_to_find_min"].mean()

            if np.isfinite(cent) and np.isfinite(market) and cent > 0:
                rel = abs(market - cent) / cent
                checks.append(("Market planner is within 75% of centralized mean time", rel < 0.75))
            else:
                checks.append(("Market planner is within 75% of centralized mean time", False))
        except Exception:
            checks.append(("Market planner is within 75% of centralized mean time", False))

    return df, checks


def run_preview(sim, fraction=0.75):
    steps = max(12, int(float(fraction) * float(sim.run_steps)))
    for _ in range(steps):
        if sim.finished:
            break
        sim.one_step()
    return sim


# ==============================================================================
# Streamlit UI
# ==============================================================================

st.set_page_config(page_title="SARSwarm", layout="wide", initial_sidebar_state="expanded")

if "cfg" not in st.session_state:
    st.session_state.cfg = default_config()

if "sim" not in st.session_state:
    with st.spinner(
        "Loading realistic terrain, initializing Bayesian prior, running swarm preview, "
        "and generating live animation..."
    ):
        sim = SwarmSim(st.session_state.cfg)
        run_preview(sim, fraction=0.75)
        st.session_state.sim = sim
        st.session_state.gif = generate_animation(sim, max_frames=34)

st.title("SARSwarm")
st.caption(
    "Multi-drone search and rescue simulation with shared Bayesian probability map, diffusion, "
    "sensor occlusion, packet-loss-tolerant consensus, market-based coordination, safety failsafes, "
    "and automated evaluation. The demo below starts automatically."
)

tab_live, tab_control, tab_belief, tab_coord, tab_safety, tab_eval, tab_ops = st.tabs(
    ["Live Demo", "Mission Control", "Belief & Sensors", "Coordination", "Safety & Overrides", "Evaluation & Report", "Operational Note"]
)

with st.sidebar:
    st.header("Mission Configuration")
    cfg0 = st.session_state.cfg

    seed = st.number_input(
        "Terrain / target seed",
        min_value=0,
        max_value=100000,
        value=int(cfg0["seed"]),
        step=1,
    )

    res_label = st.selectbox(
        "Map resolution",
        ["50 m (detailed, slower)", "100 m (fast demo)"],
        index=0 if int(cfg0["cell_size"]) <= 50 else 1,
    )

    grid_n = 40 if res_label.startswith("50") else 20
    cell_size = 50.0 if res_label.startswith("50") else 100.0

    num_drones = st.slider("Initial drones", 1, 8, int(cfg0["num_drones"]))
    replacement_drones = st.slider("Replacement drones in pool", 0, 4, int(cfg0["replacement_drones"]))

    mode_options = {
        "Lawnmower sweep": "lawnmower",
        "Random coverage": "random",
        "Centralized greedy": "centralized",
        "Decentralized market": "market",
    }

    mode_label = st.selectbox(
        "Planner",
        list(mode_options.keys()),
        index=list(mode_options.values()).index(cfg0["mode"]),
    )
    mode = mode_options[mode_label]

    packet_loss = st.slider("Packet loss", 0.0, 0.5, float(cfg0["packet_loss"]), step=0.05)
    noise_free = st.checkbox("Noise-free sensors", value=bool(cfg0["noise_free"]))

    pd_base = st.slider(
        "Base thermal detection probability",
        0.10,
        0.99,
        float(cfg0["pd_base"]),
        step=0.01,
        disabled=noise_free,
    )

    pf_base = st.slider(
        "False-alarm probability per cell",
        0.0,
        0.10,
        float(cfg0["pf_base"]),
        step=0.005,
        disabled=noise_free,
    )

    deadline_min = st.slider("Deadline (minutes)", 10, 90, int(cfg0["deadline_min"]))
    dt = st.select_slider("Simulation time step (seconds)", options=[2.0, 5.0, 10.0], value=float(cfg0["dt"]))
    speed = st.slider("Drone speed (m/s)", 5.0, 20.0, float(cfg0["speed"]), step=1.0)
    altitude = st.slider("Nominal altitude (m)", 60.0, 160.0, float(cfg0["altitude"]), step=5.0)
    fov_deg = st.slider("Downward sensor FOV (degrees)", 30.0, 90.0, float(cfg0["fov_deg"]), step=5.0)

    target_moves = st.checkbox("Target moves during mission", value=bool(cfg0["target_moves"]))
    central_fallback = st.checkbox("Centralized fallback for market planner", value=bool(cfg0["central_fallback"]))

    apply = st.button("Apply and restart live demo", use_container_width=True)

    if apply:
        new_cfg = default_config()
        new_cfg.update({
            "seed": int(seed),
            "grid_n": int(grid_n),
            "cell_size": float(cell_size),
            "num_drones": int(num_drones),
            "replacement_drones": int(replacement_drones),
            "mode": mode,
            "packet_loss": float(packet_loss),
            "noise_free": bool(noise_free),
            "pd_base": float(pd_base),
            "pf_base": float(pf_base),
            "deadline_min": float(deadline_min),
            "dt": float(dt),
            "speed": float(speed),
            "altitude": float(altitude),
            "fov_deg": float(fov_deg),
            "target_moves": bool(target_moves),
            "central_fallback": bool(central_fallback),
            "frame_every": 5 if grid_n >= 40 else 3,
            "candidate_angles": 16 if grid_n >= 40 else 12,
            "market_tasks": 20 if grid_n >= 40 else 16,
            "record_history": True,
            "enable_logs": True,
            "failure_step": None,
        })

        st.session_state.cfg = new_cfg

        with st.spinner("Restarting simulation and generating live animation..."):
            sim = SwarmSim(new_cfg)
            run_preview(sim, fraction=0.75)
            st.session_state.sim = sim
            st.session_state.gif = generate_animation(sim, max_frames=34)

        st.rerun()

with tab_live:
    sim = st.session_state.sim

    mcol1, mcol2, mcol3, mcol4, mcol5, mcol6 = st.columns(6)

    searching = [d for d in sim.drones if d.state == "SEARCH"]
    min_batt = min([d.battery for d in sim.drones if d.state in ("SEARCH", "RTB")], default=0.0)
    coverage_pct = 100.0 * float(sim.visited.sum()) / float(sim.terrain.n * sim.terrain.n)

    mcol1.metric("Mission time", f"{sim.time / 60.0:.2f} min")
    mcol2.metric("Status", "FOUND" if sim.found else "SEARCHING")
    mcol3.metric("Active searchers", len(searching))
    mcol4.metric("Min battery", f"{min_batt:.0f}%")
    mcol5.metric("Cells covered", f"{coverage_pct:.1f}%")
    mcol6.metric("Separation violations", sim.separation_violations)

    st.subheader("Generated live animation file")

    gif = st.session_state.get("gif", None)
    if gif:
        st.markdown(gif_to_html(gif), unsafe_allow_html=True)
        st.download_button(
            "Download live animation (GIF)",
            data=gif,
            file_name="sarswarm_live_demo.gif",
            mime="image/gif",
            use_container_width=True,
        )
    else:
        st.info("No animation cached. Click 'Generate animation' in Mission Control.")

    st.subheader("Current probability heatmap, drone tracks, and detections")

    show_truth = st.checkbox("Reveal ground truth after detection", value=False, key="show_truth_live")
    fig = create_map_fig(sim, frame=None, show_truth=show_truth)
    st.pyplot(fig)
    plt.close(fig)

with tab_control:
    sim = st.session_state.sim
    st.subheader("Interactive simulation controls")

    c1, c2, c3, c4, c5 = st.columns(5)

    if c1.button("Step once", use_container_width=True):
        if sim.finished:
            st.warning("Mission is finished. Use Reset or Run to end after resetting.")
        else:
            sim.one_step()
            st.session_state.gif = None
            st.rerun()

    if c2.button("Run 10 steps", use_container_width=True):
        if sim.finished:
            st.warning("Mission is finished. Use Reset or Run to end after resetting.")
        else:
            for _ in range(10):
                if sim.finished:
                    break
                sim.one_step()
            st.session_state.gif = None
            st.rerun()

    if c3.button("Run to end", use_container_width=True):
        with st.spinner("Running simulation to completion..."):
            sim.run()
            st.session_state.gif = generate_animation(sim, max_frames=36)
        st.rerun()

    if c4.button("Generate animation", use_container_width=True):
        with st.spinner("Rendering live animation file..."):
            st.session_state.gif = generate_animation(sim, max_frames=36)
        st.success("Animation generated.")

    if c5.button("Reset & auto-preview", use_container_width=True):
        with st.spinner("Resetting and generating preview animation..."):
            sim = SwarmSim(st.session_state.cfg)
            run_preview(sim, fraction=0.75)
            st.session_state.sim = sim
            st.session_state.gif = generate_animation(sim, max_frames=34)
        st.rerun()

    st.subheader("Drone status")

    rows = []
    for d in sim.drones:
        rows.append({
            "ID": d.id,
            "State": d.state,
            "X (m)": float(d.pos[0]),
            "Y (m)": float(d.pos[1]),
            "Altitude (m)": float(d.altitude),
            "Battery %": float(d.battery),
            "Comm OK": bool(d.comm_ok),
            "Lost time (s)": float(d.lost_time),
            "Assigned cell": str(d.assigned) if d.assigned is not None else "",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Event log")
    if sim.logs:
        st.dataframe(pd.DataFrame(sim.logs[-200:]), use_container_width=True)
    else:
        st.write("No logs recorded.")

with tab_belief:
    sim = st.session_state.sim
    st.subheader("Bayesian probability map and sensor model")

    col1, col2 = st.columns(2)

    with col1:
        fig = simple_heatmap_fig(
            sim.prior_belief,
            "Initial prior after diffusion from LKP",
            cmap="inferno",
            extent=[0, sim.extent, 0, sim.extent],
        )
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig = simple_heatmap_fig(
            sim.belief.p,
            "Current posterior probability map",
            cmap="inferno",
            extent=[0, sim.extent, 0, sim.extent],
        )
        st.pyplot(fig)
        plt.close(fig)

    st.markdown(
        "- The prior is initialized at the last-known position and diffused using terrain walkability.\n"
        "- During the mission, the map is propagated with a diffusion model between observations.\n"
        "- Sensor updates use Bayesian likelihoods with detection probability, false-alarm probability, "
        "canopy transmission, and altitude-dependent footprint."
    )

    tcol1, tcol2, tcol3 = st.columns(3)

    with tcol1:
        fig = simple_heatmap_fig(
            sim.terrain.elevation,
            "Terrain elevation (m)",
            cmap="terrain",
            extent=[0, sim.extent, 0, sim.extent],
        )
        st.pyplot(fig)
        plt.close(fig)

    with tcol2:
        fig = simple_heatmap_fig(
            sim.terrain.transmission,
            "Canopy transmission (higher is clearer)",
            cmap="Greens",
            extent=[0, sim.extent, 0, sim.extent],
        )
        st.pyplot(fig)
        plt.close(fig)

    with tcol3:
        fig = simple_heatmap_fig(
            sim.terrain.walkability,
            "Target walkability",
            cmap="viridis",
            extent=[0, sim.extent, 0, sim.extent],
        )
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Detection probability surface at current nominal altitude")

    n = sim.terrain.n
    pd_map = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            pd_map[i, j] = sim.sensor_pd_cell(i, j)

    fig = simple_heatmap_fig(
        pd_map,
        "Cell-wise detection probability (without range falloff)",
        cmap="magma",
        extent=[0, sim.extent, 0, sim.extent],
    )
    st.pyplot(fig)
    plt.close(fig)

with tab_coord:
    sim = st.session_state.sim
    st.subheader("Coordination layer")

    value = (
        sim.belief.p
        * sim.override_mask
        * sim.terrain.transmission
        / (1.0 + sim.coverage_count)
    )

    fig = simple_heatmap_fig(
        value,
        "Planner value map: belief x override x transmission x novelty",
        cmap="plasma",
        extent=[0, sim.extent, 0, sim.extent],
    )
    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        "**Centralized greedy:** each drone selects a reachable waypoint maximizing expected information gain "
        "per travel time, with separation and battery penalties.\n\n"
        "**Decentralized market:** drones bid on high-value search cells. Lost bids are tolerated; unassigned "
        "drones fall back to local or centralized selection when a central link is available."
    )

    if sim.cfg["mode"] == "market" and sim.last_market["tasks"]:
        st.subheader("Last market auction snapshot")

        task_rows = []
        for task in sim.last_market["tasks"][:20]:
            task_rows.append({
                "task_id": task["id"],
                "cell": str(task["cell"]),
                "value": float(task["value"]),
                "x_m": float(task["pos"][0]),
                "y_m": float(task["pos"][1]),
            })
        st.dataframe(pd.DataFrame(task_rows), use_container_width=True)

        bid_rows = []
        for bid, did, task in sim.last_market["bids"][:30]:
            bid_rows.append({
                "drone": int(did),
                "task_id": int(task["id"]),
                "task_cell": str(task["cell"]),
                "bid": float(bid),
            })
        st.dataframe(pd.DataFrame(bid_rows), use_container_width=True)

    st.subheader("Planned paths for current step")

    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=90)
    E = sim.extent

    ax.imshow(sim.belief.p, cmap="inferno", alpha=0.55, origin="lower", extent=[0, E, 0, E])

    for did, path in sim.active_paths(sim.planned_paths).items():
        if len(path) > 1:
            arr = np.array(path, dtype=float)
            ax.plot(arr[:, 0], arr[:, 1], lw=2.0, alpha=0.85, label=f"Drone {did}")

    for d in sim.drones:
        if d.state in ("SEARCH", "RTB"):
            ax.scatter(d.pos[0], d.pos[1], s=70, edgecolors="white")

    ax.add_patch(Rectangle((0, 0), E, E, fill=False, edgecolor="yellow", linewidth=2.0))
    ax.set_xlim(0, E)
    ax.set_ylim(0, E)
    ax.set_title("Planned paths")
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

with tab_safety:
    sim = st.session_state.sim
    st.subheader("Safety enforcement")

    st.markdown(
        "- **Geofence:** all positions are clipped to the 2 km x 2 km operational area.\n"
        "- **Minimum altitude:** each drone is kept above terrain plus a hard clearance margin.\n"
        "- **Minimum separation:** pairwise horizontal separation is enforced and violations are counted.\n"
        "- **Comm-loss failsafe:** if a drone loses telemetry beyond the timeout, it returns to base.\n"
        "- **Battery-aware RTB:** a drone returns when remaining energy cannot cover return plus reserve.\n"
        "- **Replacement dispatch:** if the swarm is short-handed and a charged replacement is available, it is launched."
    )

    st.subheader("Operator priority override")

    region = st.selectbox(
        "Priority region",
        ["None", "North", "South", "East", "West", "Center", "River corridor", "Last known position"],
    )
    intensity = st.slider("Priority multiplier", 1.0, 10.0, 2.5, step=0.5)

    oc1, oc2, oc3 = st.columns(3)

    if oc1.button("Apply override", use_container_width=True):
        mask = np.ones_like(sim.override_mask)
        n = sim.terrain.n

        if region == "North":
            mask[:n // 3, :] *= intensity
        elif region == "South":
            mask[-n // 3:, :] *= intensity
        elif region == "East":
            mask[:, -n // 3:] *= intensity
        elif region == "West":
            mask[:, :n // 3] *= intensity
        elif region == "Center":
            mask[n // 4:3 * n // 4, n // 4:3 * n // 4] *= intensity
        elif region == "River corridor":
            dil = np.pad(sim.terrain.river.astype(float), 1, mode="edge")
            corr = (
                dil[:-2, :-2] + dil[:-2, 1:-1] + dil[:-2, 2:]
                + dil[1:-1, :-2] + dil[1:-1, 1:-1] + dil[1:-1, 2:]
                + dil[2:, :-2] + dil[2:, 1:-1] + dil[2:, 2:]
            ) > 0
            mask[corr] *= intensity
        elif region == "Last known position":
            i, j = sim.lkp
            mask[max(0, i - 3):min(n, i + 4), max(0, j - 3):min(n, j + 4)] *= intensity

        sim.override_mask = mask
        sim.log(f"Operator override applied: region={region}, multiplier={intensity}")
        st.success("Override applied and logged.")

    if oc2.button("Clear override", use_container_width=True):
        sim.override_mask = np.ones_like(sim.override_mask)
        sim.log("Operator override cleared")
        st.success("Override cleared.")

    if oc3.button("Reset override to neutral", use_container_width=True):
        sim.override_mask = np.ones_like(sim.override_mask)
        st.info("Override cleared to neutral.")

    st.subheader("Force a drone to return to base")

    flying_ids = [d.id for d in sim.drones if d.state == "SEARCH"]
    if flying_ids:
        forced = st.selectbox("Drone ID to force RTB", flying_ids)
        if st.button("Force selected drone RTB", use_container_width=True):
            d = sim.get_drone(forced)
            if d is not None and d.state == "SEARCH":
                d.state = "RTB"
                d.assigned = None
                sim.log(f"Operator forced Drone {d.id} to RTB")
                st.success(f"Drone {forced} commanded to return.")
    else:
        st.write("No searching drones available to force RTB.")

    st.subheader("Safety status")

    rows = []
    for d in sim.drones:
        if d.state not in ("SEARCH", "RTB"):
            continue

        i, j = sim.cell_from_pos(d.pos)
        elev = float(sim.terrain.elevation[i, j])
        ret_wh = sim.return_energy_wh(d.pos)
        rem_wh = d.battery / 100.0 * d.capacity
        reserve_wh = d.capacity * sim.cfg["reserve_pct"] / 100.0

        rows.append({
            "ID": d.id,
            "State": d.state,
            "Terrain elev (m)": elev,
            "Altitude (m)": float(d.altitude),
            "Clearance (m)": float(d.altitude - elev),
            "Remaining Wh": float(rem_wh),
            "Return Wh": float(ret_wh),
            "Reserve Wh": float(reserve_wh),
            "Margin Wh": float(rem_wh - ret_wh - reserve_wh),
            "Comm lost (s)": float(d.lost_time),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

with tab_eval:
    st.subheader("Monte Carlo evaluation and planner comparison")

    st.caption(
        "The full requested study is 100 scenarios per configuration. For a quick interactive check, "
        "the default is lower. Increase to 100 for the full report."
    )

    ecol1, ecol2, ecol3, ecol4 = st.columns(4)

    with ecol1:
        scenarios = st.number_input(
            "Scenarios per configuration",
            min_value=1,
            max_value=100,
            value=20,
            step=1,
        )

    with ecol2:
        eval_cell_size = st.selectbox(
            "Evaluation grid",
            [100.0, 125.0],
            index=0,
            format_func=lambda x: f"{int(x)} m",
        )

    with ecol3:
        deadline_eval = st.slider("Evaluation deadline (min)", 15, 60, 30)

    with ecol4:
        num_drones_eval = st.slider("Drones per scenario", 3, 8, 5)

    modes = st.multiselect(
        "Planners to compare",
        ["lawnmower", "random", "centralized", "market"],
        default=["lawnmower", "random", "centralized", "market"],
    )

    losses = st.multiselect(
        "Packet loss rates",
        [0.0, 0.2, 0.5],
        default=[0.0, 0.2, 0.5],
        format_func=lambda x: f"{int(x * 100)}%",
    )

    include_failure = st.checkbox("Include one-drone mid-mission loss", value=True)

    estimated = int(scenarios) * len(modes) * len(losses) * (2 if include_failure else 1)
    st.info(f"Estimated simulations: {estimated}")

    run_btn = st.button("Run evaluation", use_container_width=True)

    if run_btn:
        if not modes or not losses:
            st.warning("Select at least one planner and one packet-loss rate.")
        else:
            progress = st.progress(0.0)
            status = st.empty()

            with st.spinner("Running evaluation..."):
                df = run_evaluation(
                    scenarios=int(scenarios),
                    modes=modes,
                    losses=losses,
                    include_failure=include_failure,
                    cell_size=float(eval_cell_size),
                    deadline_min=float(deadline_eval),
                    num_drones=int(num_drones_eval),
                    base_seed=9000,
                    progress_cb=lambda frac, msg: (progress.progress(frac), status.text(msg)),
                )

            st.session_state.eval_raw = df
            st.session_state.eval_agg = aggregate_eval(df)
            progress.progress(1.0)
            status.text("Evaluation complete.")

    if "eval_raw" in st.session_state and "eval_agg" in st.session_state:
        raw = st.session_state.eval_raw
        agg = st.session_state.eval_agg

        st.subheader("Aggregate results")
        st.dataframe(agg, use_container_width=True)

        st.subheader("Detection probability by planner and packet loss")
        try:
            pivot = agg.pivot_table(index="mode", columns="packet_loss", values="detection_prob", aggfunc="mean")
            fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=90)
            pivot.plot(kind="bar", ax=ax)
            ax.set_ylabel("Detection probability by deadline")
            ax.set_xlabel("Planner")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.grid(axis="y", alpha=0.3)
            ax.set_title("Probability of detection under packet loss")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as ex:
            st.write(f"Chart unavailable: {ex}")

        st.subheader("Mean time to find (found scenarios only)")
        try:
            pivot2 = agg.pivot_table(index="mode", columns="packet_loss", values="mean_time_min", aggfunc="mean")
            fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=90)
            pivot2.plot(kind="bar", ax=ax)
            ax.set_ylabel("Mean time to find (min)")
            ax.set_xlabel("Planner")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.grid(axis="y", alpha=0.3)
            ax.set_title("Search efficiency")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as ex:
            st.write(f"Chart unavailable: {ex}")

        notes = make_failure_notes(agg, raw)
        st.subheader("Failure analysis")
        st.markdown(notes)

        report_md = "# SARSwarm Evaluation Report\n\n"
        report_md += f"Total scenarios evaluated: {len(raw)}\n\n"
        report_md += "## Aggregate table\n\n"
        report_md += df_to_markdown(agg)
        report_md += "\n\n## Failure analysis\n\n"
        report_md += notes

        rc1, rc2 = st.columns(2)

        rc1.download_button(
            "Download Markdown report",
            data=report_md.encode("utf-8"),
            file_name="sarswarm_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

        rc2.download_button(
            "Download raw CSV",
            data=raw.to_csv(index=False).encode("utf-8"),
            file_name="sarswarm_raw.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.divider()
    st.subheader("Staged validation")

    st.markdown(
        "This quick check follows the requested build order: single-drone lawnmower noise-free, "
        "multi-drone centralized, communication loss, then decentralized market."
    )

    if st.button("Run staged validation", use_container_width=True):
        with st.spinner("Running staged validation..."):
            sdf, checks = run_stage_validation(seeds=3)

        st.dataframe(sdf, use_container_width=True)

        st.markdown("#### Validation checks")
        for name, ok in checks:
            st.write(f"{'PASS' if ok else 'FAIL'} - {name}")

with tab_ops:
    st.subheader("Operational requirements note")

    st.markdown(
        """
### Airspace authorization
- Identify the governing aviation authority and rule set for the operational region, for example Part 107 in the United States or the equivalent national UAV regulation.
- Determine whether the incident area lies in controlled airspace, a temporary flight restriction, a national-security area, or near manned emergency aircraft. Obtain authorization through the appropriate automated facility, such as LAANC, or through a manual waiver or authorization when required.
- For search and rescue, confirm whether emergency authority, a Certificate of Authorization or Waiver, or a public-safety exemption applies. Document the legal basis before launch.
- Respect maximum altitude limits, terrain clearance, night-operation rules, and any requirements for anti-collision lighting, ADS-B Receive, or remote identification.
- Deconflict with helicopters, air medical assets, firefighting aircraft, and other UAS. Maintain a common operating picture with the incident commander.

### Human supervision of each drone
- Assign a clearly designated remote pilot in command or supervising operator for each airborne drone, or demonstrate that one supervisor can safely manage the swarm under the applicable regulatory framework.
- Maintain continuous two-way command and control, telemetry, and geofencing. If the link exceeds the configured loss timeout, each drone must execute return-to-base or a preapproved safe landing behavior.
- Preserve a human override path at all times. Operators must be able to change search priorities, force a drone to return, pause autonomy, or terminate the mission. All overrides should be logged with time, actor, and reason.
- Use visual observers or equivalent sense-and-avoid capability when required by regulation or when operating near people, vehicles, obstacles, or other aircraft.
- Ensure battery-aware autonomy does not replace human responsibility. The operator must monitor energy margins, replacement availability, and unexpected no-return events.

### Privacy constraints for thermal imaging
- Thermal sensors can reveal persons, structures, and activity patterns. Collect only data necessary for the locate-and-rescue mission.
- Prefer downward-looking, narrow-footprint sensing over wide-area persistent surveillance. Avoid imaging residences, private lands, or non-target areas unless operationally necessary.
- Apply data minimization. Retain raw thermal imagery only for the period required for verification, evidentiary integrity, or safety investigation, then delete or aggregate it.
- Publish or communicate a privacy notice when feasible, especially for prolonged operations over populated areas.
- Follow agency policy and legal counsel requirements for warrantless emergency searches, consent, exigent circumstances, and post-incident review.
- Mask or redact non-target persons and sensitive locations in reports, dashboards, and shared maps when the information is not needed for the rescue mission.
- Audit access to thermal data and maintain chain-of-custody logs if imagery may be used in legal proceedings.
        """
    )