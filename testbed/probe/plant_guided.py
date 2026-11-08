#!/usr/bin/env python3
# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""The few minutes a person is needed for: the answers the robot's stack cannot give.

OBSERVE ONLY. The operator moves the robot by hand; this script never does. It
subscribes, times windows, and asks for typed numbers. It creates no publisher.

Reads the draft written by ``plant_probe.py`` (``--plant OUT/plant.json``), runs:

  a. motion-channel test — rest, hand on the robot (no push), a hand push of ~10 cm,
     rest again. Classifies each odometry topic as ``measured`` (it moved with no
     command) or ``command-derived`` (it did not, while another channel shows the push
     happened). Reports whether each IMU refutes rest during the push. The IMU is never
     reported as a rest confirmer.
  b. scan-difference rest check — frame-to-frame range difference distributions for
     rest / hand-on / push / rest. Reports the numbers and whether they separate. It is
     not a scan-matched velocity and it is not claimed as a confirming channel.
  c. one typed measurement per unknown that a ruler answers (physical width; lidar
     height when neither TF nor URDF gave a pose), recorded ``source: operator``.

and writes the merged plant.json / plant.yaml plus guided_summary.txt next to it.

On a simulated plant (``plant.kind: simulated``) nothing is recorded or asked: every
step is reported ``not-testable-here``, because a hand push has no physical effect on a
simulated robot and inventing one would fill fields with numbers nobody measured.

    python3 plant_guided.py --plant /tmp/probe/out/plant.json
    python3 plant_guided.py --print-script        # the exact words, no ROS needed
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import plant_fields as pf  # noqa: E402
from plant_fields import measured, unknown  # noqa: E402

VERSION = "0.1.0"

# The windows, in order, and the ONE line the operator reads for each.
WINDOWS = (
    ("rest", 5.0, "Hands off. Do not touch the robot. Recording 5 s at rest."),
    ("hand", 5.0, "Put one hand flat on top of the robot. Do NOT push. Hold still 5 s."),
    ("push", 6.0, "PUSH the robot about 10 cm FORWARD by hand now, then let go and step back."),
    ("after", 5.0, "Hands off. Do not touch the robot. Recording 5 s at rest."),
)
COUNTDOWN_S = 3
PLACE_LINE = ("Robot powered, on a clear floor, 30 cm free all round. Stand beside it, "
              "within arm's reach. Nothing will drive it.")
DISTANCE_PROMPT = ("How far did the robot move, in cm? Measure or estimate; press Enter "
                   "to skip: ")
RULER_PROMPTS = {
    "geometry.physical_width_m": (
        "Measure the robot's widest point, side to side (outside of wheel to outside of "
        "wheel), in mm, and type the number: "),
    "geometry.lidar_height_m": (
        "Measure from the floor to the centre of the lidar's window, in mm, and type the "
        "number: "),
}

# Classification thresholds. They decide a label, they are printed beside it, and they
# are properties of this TEST (a 10 cm push), not of the plant.
ODOM_MOVED_M = 0.03        # >= 30 % of the requested push: odometry saw the motion
ODOM_STILL_M = 0.005       # < 5 mm: odometry did not see it
IMU_DEV_SD = 6.0           # push-window deviation of |a| over rest sd ...
IMU_DEV_FLOOR = 0.15       # ... and at least this, m/s^2, to count as refuting rest
OPERATOR_PUSHED_CM = 3.0   # an operator-reported move this large is a push witness

NOT_TESTABLE_SIM = ("not-testable-here: the plant is simulated; a hand push has no "
                    "physical effect on it, so nothing was recorded")
UNLOCK_SIM = "run plant_guided.py on the physical robot"


# ── pure analysis ───────────────────────────────────────────────────────────


def split_windows(recs: list, bounds: dict) -> dict:
    """{window: [records]} by arrival time; ``bounds`` is {window: (t0, t1)} monotonic."""
    out = {w: [] for w in bounds}
    for r in recs:
        for w, (t0, t1) in bounds.items():
            if t0 <= r["recv"] < t1:
                out[w].append(r)
                break
    return out


def _median_xy(recs):
    return (statistics.median(r["x"] for r in recs), statistics.median(r["y"] for r in recs))


def odom_motion(by_win: dict) -> dict | None:
    """Displacement across the push, and the rest noise it is compared with."""
    need = ("rest", "hand", "after")
    if any(len(by_win.get(w, [])) < 2 for w in need):
        return None
    rest, hand, after = (_median_xy(by_win[w]) for w in need)
    vmax = {w: max((math.hypot(r["vx"], r["vy"]) for r in by_win.get(w, [])), default=None)
            for w in ("rest", "hand", "push", "after")}
    return {
        "push_displacement_m": round(math.hypot(after[0] - hand[0], after[1] - hand[1]), 5),
        "rest_to_hand_displacement_m": round(math.hypot(hand[0] - rest[0], hand[1] - rest[1]), 5),
        "max_speed_m_s": {w: (round(v, 5) if v is not None else None) for w, v in vmax.items()},
        "n": {w: len(by_win.get(w, [])) for w in by_win},
    }


def imu_refutation(by_win: dict) -> dict | None:
    rest = by_win.get("rest", [])
    if len(rest) < 10:
        return None
    mags = [math.sqrt(r["ax"] ** 2 + r["ay"] ** 2 + r["az"] ** 2) for r in rest]
    mu0, sd0 = statistics.mean(mags), statistics.pstdev(mags)
    thr = max(IMU_DEV_SD * sd0, IMU_DEV_FLOOR)
    dev = {}
    gyro = {}
    for w, recs in by_win.items():
        if recs:
            dev[w] = round(max(abs(math.sqrt(r["ax"] ** 2 + r["ay"] ** 2 + r["az"] ** 2) - mu0)
                               for r in recs), 5)
            gyro[w] = round(max(math.sqrt(r["wx"] ** 2 + r["wy"] ** 2 + r["wz"] ** 2)
                                for r in recs), 5)
    def over(w):   # None when the window got no IMU message: not "did not refute"
        return None if w not in dev else dev[w] > thr

    return {"rest_mean_abs_a": round(mu0, 5), "rest_sd_abs_a": round(sd0, 5),
            "threshold_m_s2": round(thr, 5), "max_dev_abs_a": dev, "max_gyro_rad_s": gyro,
            "refutes_in_push": over("push"), "refutes_in_hand": over("hand"),
            "refutes_in_rest_after": over("after")}


def scan_distributions(by_win: dict) -> dict | None:
    out = {}
    for w, recs in by_win.items():
        pairs = []
        for a, b in zip(recs, recs[1:]):
            d = pf.scan_pair_diff(a["ranges"], b["ranges"], a["range_min"], a["range_max"])
            if d is not None:
                pairs.append(d)
        out[w] = {"mean_abs_m": pf.summary_stats([p["mean_abs_m"] for p in pairs]),
                  "frac_changed_gt_2cm": pf.summary_stats([p["frac_changed"] for p in pairs])}
    if not out.get("rest", {}).get("mean_abs_m") or not out.get("push", {}).get("mean_abs_m"):
        return None
    rest_max = out["rest"]["mean_abs_m"]["max"]
    push_max = out["push"]["mean_abs_m"]["max"]
    hand = out.get("hand", {}).get("mean_abs_m")
    hand_max = hand["max"] if hand else None
    return {"windows": out,
            "push_max_over_rest_max": (round(push_max / rest_max, 3) if rest_max > 0 else None),
            "push_separates_from_rest": push_max > rest_max,
            "push_separates_from_hand_on": (push_max > hand_max) if hand_max is not None else None}


def push_witnesses(imus: dict, scans: dict, operator_cm) -> list:
    w = []
    for t, r in imus.items():
        if r and r["refutes_in_push"] and r["refutes_in_hand"] is False:
            w.append(f"IMU {t} deviated {r['max_dev_abs_a']['push']} m/s^2 in the push "
                     f"window (threshold {r['threshold_m_s2']}), not in the hand-on window")
    for t, r in scans.items():
        if r and r["push_separates_from_rest"] and r["push_separates_from_hand_on"]:
            w.append(f"scan {t} push-window frame difference exceeded both rest and hand-on")
    if operator_cm is not None and operator_cm >= OPERATOR_PUSHED_CM:
        w.append(f"operator reported the robot moved {operator_cm} cm")
    return w


def classify_odometry(topic: str, motion: dict | None, witnesses: list, commands: list):
    """measured / command-derived, or unknown with the reason. Never a guess."""
    nonzero = [c for c in commands if abs(c["vx"]) + abs(c["vy"]) + abs(c["wz"]) > 0]
    if nonzero:
        return unknown(f"{len(nonzero)} non-zero velocity command(s) were published during "
                       "the test, so motion may have been commanded; test void")
    if motion is None:
        return unknown(f"{topic}: too few messages in the rest/hand/after windows")
    d = motion["push_displacement_m"]
    ev = {"evidence": motion, "thresholds_m": {"moved": ODOM_MOVED_M, "still": ODOM_STILL_M},
          "push_witnesses": witnesses}
    if d >= ODOM_MOVED_M:
        return measured("measured", f"measured: {topic} moved {d} m across a hand push with "
                        "no velocity command published", **ev)
    if d < ODOM_STILL_M and witnesses:
        return measured("command-derived", f"measured: {topic} moved {d} m while "
                        + "; ".join(witnesses) + " — and no velocity command was published",
                        **ev)
    if d < ODOM_STILL_M:
        return unknown(f"{topic} moved {d} m, but no channel shows the push happened (no "
                       "IMU refuted rest, no scan separated from rest and hand-on, no "
                       f"operator-reported move of {OPERATOR_PUSHED_CM} cm or more); the "
                       "test cannot tell command-derived from not-pushed", **ev)
    return unknown(f"inconclusive: {topic} moved {d} m, between {ODOM_STILL_M} and "
                   f"{ODOM_MOVED_M} m", **ev)


def imu_field(topic, r):
    if r is None:
        return unknown(f"{topic}: fewer than 10 messages in the rest window")
    if r["refutes_in_push"] is None:
        return unknown(f"{topic}: no message in the push window", evidence=r)
    verdict = ("refutes rest during the push" if r["refutes_in_push"]
               else "did not refute rest during the push")
    return measured(verdict, f"measured: {topic} |a| deviation from the rest mean, per "
                    "window; an IMU can refute rest, it cannot confirm it", evidence=r,
                    refutes=r["refutes_in_push"])


def scan_field(topic, r):
    if r is None:
        return unknown(f"{topic}: too few consecutive frame pairs in the rest or push window")
    if r["push_separates_from_rest"] and r["push_separates_from_hand_on"]:
        v = "push separates from rest and from hand-on"
    elif r["push_separates_from_rest"]:
        v = ("push separates from rest but NOT from hand-on: the operator's body in the "
             "scan may explain it")
    else:
        v = "push does not separate from rest"
    return measured(v, f"measured: {topic} consecutive-frame range difference (beams valid "
                    "in both frames), per window — a difference statistic, not a "
                    "scan-matched velocity, and not claimed as a rest-confirming channel",
                    evidence=r)


def not_testable(reason: str, unlock: str) -> dict:
    return unknown(reason, status="not-testable-here", unlock=unlock)


def simulated_result(doc: dict) -> dict:
    """Every guided field on a simulated plant: not-testable-here, nothing invented."""
    mc = doc.get("motion_channel", {})
    for key in ("odometry", "imu_refutes_rest", "scan_difference"):
        cur = mc.get(key)
        if isinstance(cur, dict) and not pf.is_field(cur):
            mc[key] = {t: not_testable(NOT_TESTABLE_SIM, UNLOCK_SIM) for t in cur}
        else:
            mc[key] = not_testable(NOT_TESTABLE_SIM, UNLOCK_SIM)
    doc["motion_channel"] = mc
    geo = doc.setdefault("geometry", {})
    geo["physical_width_m"] = not_testable(
        "not-testable-here: a simulated plant has no physical width to measure",
        "measure the physical robot")
    if _lidar_pose_unknown(doc):
        geo["lidar_height_m"] = not_testable(
            "not-testable-here: a simulated plant has no physical lidar to measure",
            "measure the physical robot, or publish its URDF / TF")
    return doc


def parse_mm(text: str):
    """A typed millimetre value -> metres, or None. Refuses zero, negatives, and words."""
    t = (text or "").strip().lower().replace("mm", "").strip()
    if not t:
        return None
    try:
        v = float(t)
    except ValueError:
        return None
    if not math.isfinite(v) or v <= 0 or v > 5000:
        return None
    return round(v / 1000.0, 4)


# ── the run ─────────────────────────────────────────────────────────────────


def print_script(out=sys.stdout) -> None:
    print(PLACE_LINE, file=out)
    for name, secs, line in WINDOWS:
        print(f"[{name}, {secs:g} s, after a {COUNTDOWN_S}-2-1 countdown] {line}", file=out)
    print(f"[after the windows] {DISTANCE_PROMPT}", file=out)
    for k, p in RULER_PROMPTS.items():
        print(f"[only if {k} is unknown] {p}", file=out)


def _ask(prompt: str, preset: dict, key: str):
    if key in preset:
        print(prompt + preset[key] + "   (from --answer)")
        return preset[key]
    if not sys.stdin.isatty():
        print(prompt + "(no terminal: skipped)")
        return None
    try:
        return input(prompt)
    except EOFError:
        return None


T_ODOM, T_IMU, T_SCAN, T_TWIST = ("nav_msgs/msg/Odometry", "sensor_msgs/msg/Imu",
                                  "sensor_msgs/msg/LaserScan", "geometry_msgs/msg/Twist")


def live_topics(doc: dict, type_name: str) -> list:
    """Topics of ``type_name`` the probe measured a rate on (i.e. that delivered)."""
    return [n for n, f in doc.get("rates", {}).items()
            if not pf.is_unknown(f) and f.get("type") == type_name]


def simulator_markers(doc: dict) -> list:
    """What the probe read that says the plant is simulated, whatever kind ended up as.

    ``kind`` can be ``unknown`` because an operator declared hardware over a marker; the
    marker still means a hand push would be performed on nothing.
    """
    f = doc.get("plant", {}).get("simulator_markers")
    if isinstance(f, dict) and pf.is_field(f) and not pf.is_unknown(f):
        return list(f["value"])
    kind = doc.get("plant", {}).get("kind", {})
    return [kind.get("source", "plant.kind: simulated")] if kind.get("value") == "simulated" \
        else []


def record(plan: dict, say=print) -> tuple:
    """Subscribe, show each window's line, time it. Returns (got, bounds, shown, own)."""
    from rclpy.qos import qos_profile_sensor_data
    from rosidl_runtime_py.utilities import get_message

    import plant_probe
    reader = plant_probe.RclpyReader("swarmada_plant_guided")
    got = {t: [] for t in plan}
    subs = []
    for topic, tn in plan.items():
        def cb(msg, _t=topic, _n=tn):
            rec = plant_probe.summarize(_n, msg)
            rec["recv"] = time.monotonic()
            got[_t].append(rec)
        subs.append(reader.node.create_subscription(get_message(tn), topic, cb,
                                                    qos_profile_sensor_data))
    bounds = {}
    shown = []
    try:
        say("")
        say(PLACE_LINE)
        reader.spin_for(2.0)   # discovery; nothing recorded here counts
        for name, secs, line in WINDOWS:
            say("")
            say(">>> " + line)
            shown.append(line)
            for i in range(COUNTDOWN_S, 0, -1):
                say(f"    {i} ...")
                reader.spin_for(1.0)
            say("    RECORDING")
            t0 = time.monotonic()
            reader.spin_for(secs)
            bounds[name] = (t0, time.monotonic())
            say(f"    done ({name})")
        own = reader.own_publishers()
    finally:
        for s in subs:
            reader.node.destroy_subscription(s)
        reader.close()
    return got, bounds, shown, own


def assemble(doc: dict, got: dict, bounds: dict, ask) -> dict:
    """Merge the recorded windows and the typed answers into ``doc``. Pure: no ROS.

    ``ask(prompt, key)`` returns what the operator typed, or None when nobody could.
    """
    odoms, imus, scans = (live_topics(doc, t) for t in (T_ODOM, T_IMU, T_SCAN))
    typed = ask(DISTANCE_PROMPT, "distance_cm")
    op_cm, op_field = None, None
    if typed is not None and typed.strip():
        try:
            op_cm = float(typed.strip().lower().replace("cm", "").strip())
            op_field = pf.operator(op_cm, DISTANCE_PROMPT.strip(), units="cm")
        except ValueError:
            op_field = unknown(f"operator typed {typed.strip()!r}, not a distance in cm")
    elif typed is not None:
        op_field = unknown("operator skipped")
    else:
        op_field = unknown("no operator at a terminal")

    imu_r = {t: imu_refutation(split_windows(got.get(t, []), bounds)) for t in imus}
    scan_r = {t: scan_distributions(split_windows(got.get(t, []), bounds)) for t in scans}
    witnesses = push_witnesses(imu_r, scan_r, op_cm)
    cmds = got.get("/cmd_vel", [])
    doc["motion_channel"] = {
        "odometry": ({t: classify_odometry(t, odom_motion(split_windows(got.get(t, []),
                                                                        bounds)),
                                           witnesses, cmds) for t in odoms}
                     or unknown("no odometry topic delivered messages during the probe")),
        "imu_refutes_rest": ({t: imu_field(t, imu_r[t]) for t in imus}
                             or unknown("no IMU topic delivered messages during the probe")),
        "scan_difference": ({t: scan_field(t, scan_r[t]) for t in scans}
                            or unknown("no LaserScan topic delivered messages during the probe")),
        "operator_push_distance_cm": op_field,
    }

    geo = doc.setdefault("geometry", {})
    for key, prompt in RULER_PROMPTS.items():
        field = key.split(".", 1)[1]
        if key != "geometry.physical_width_m" and not _lidar_pose_unknown(doc):
            continue
        if field in geo and not pf.is_unknown(geo[field]):
            continue
        typed = ask(prompt, field)
        m = parse_mm(typed) if typed is not None else None
        if m is not None:
            geo[field] = pf.operator(m, prompt.strip(), units="m", typed=typed.strip())
        elif typed is None:
            geo[field] = unknown("no operator at a terminal")
        elif not typed.strip():
            geo[field] = unknown("operator skipped")
        else:
            geo[field] = unknown(f"operator typed {typed.strip()!r}, not a length in mm")
    return doc


def run_live(doc: dict, opts) -> dict:
    odoms, imus, scans = (live_topics(doc, t) for t in (T_ODOM, T_IMU, T_SCAN))
    plan = {t: T_ODOM for t in odoms}
    plan.update({t: T_IMU for t in imus})
    plan.update({t: T_SCAN for t in scans})
    plan["/cmd_vel"] = T_TWIST
    t_start = time.time()
    got, bounds, shown, own = record(plan, say=lambda s: print(s, flush=True))
    print()
    doc = assemble(doc, got, bounds, lambda prompt, key: _ask(prompt, opts.answers, key))
    doc["guided"] = {"tool": "plant_guided.py", "version": VERSION,
                     "started_wall_utc": pf_utc(t_start),
                     "windows_s": {n: round(b[1] - b[0], 3) for n, b in bounds.items()},
                     "lines_shown": shown,
                     "own_publishers": own,
                     "cmd_vel_messages_during_test": len(got.get("/cmd_vel", []))}
    return doc


def _lidar_pose_unknown(doc) -> bool:
    for sect in ("tf", "urdf"):
        lp = doc.get(sect, {}).get("lidar_pose")
        if isinstance(lp, dict) and not pf.is_field(lp):
            if any(not pf.is_unknown(f) for f in lp.values()):
                return False
    return True


def pf_utc(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def guided_summary(doc: dict) -> str:
    L = ["GUIDED STEPS"]
    kind = doc["plant"]["kind"]
    L.append(f"plant kind: {kind['value']}")
    mc = doc["motion_channel"]
    for key in ("odometry", "imu_refutes_rest", "scan_difference"):
        v = mc[key]
        items = [(key, v)] if pf.is_field(v) else [(f"{key} {t}", f) for t, f in v.items()]
        for label, f in items:
            if pf.is_unknown(f):
                tag = f" [{f['status']}; unlock: {f['unlock']}]" if "status" in f else ""
                L.append(f"  {label}: unknown — {f['reason']}{tag}")
            else:
                L.append(f"  {label}: {f['value']}")
                L.append(f"      source: {f['source']}")
                ev = f.get("evidence")
                if ev:
                    L.append("      evidence: " + json.dumps(ev, sort_keys=True))
    for k, f in doc.get("geometry", {}).items():
        if pf.is_unknown(f):
            L.append(f"  geometry.{k}: unknown — {f['reason']}")
        else:
            L.append(f"  geometry.{k}: {f['value']} m  (source: {f['source']})")
    g = doc.get("guided")
    if g and "own_publishers" in g:
        L.append(f"  observe-only check: own publishers {g['own_publishers']}; /cmd_vel "
                 f"messages during the test {g['cmd_vel_messages_during_test']}")
    return "\n".join(L) + "\n"


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plant", help="plant.json written by plant_probe.py")
    ap.add_argument("--print-script", action="store_true",
                    help="print the operator's lines and exit (no ROS)")
    ap.add_argument("--answer", action="append", default=[], metavar="KEY=VALUE",
                    help="pre-typed answer: distance_cm=, physical_width_m= (in mm), "
                         "lidar_height_m= (in mm). Recorded as source: operator")
    a = ap.parse_args(argv)
    if not a.print_script and not a.plant:
        ap.error("--plant is required")
    a.answers = {}
    for kv in a.answer:
        k, sep, v = kv.partition("=")
        if not sep:
            ap.error(f"--answer {kv!r}: expected KEY=VALUE")
        a.answers[k.strip()] = v.strip()
    return a


def main(argv=None) -> int:
    opts = parse_args(sys.argv[1:] if argv is None else argv)
    if opts.print_script:
        print_script()
        return 0
    with open(opts.plant, encoding="utf-8") as f:
        doc = json.load(f)
    markers = simulator_markers(doc)
    if markers:
        print("The plant is SIMULATED (" + "; ".join(markers) + ").")
        print("Every guided step is not-testable-here; nothing will be recorded or asked.")
        doc = simulated_result(doc)
        doc["guided"] = {"tool": "plant_guided.py", "version": VERSION,
                         "started_wall_utc": pf_utc(time.time()),
                         "ran": "no — simulated plant"}
    else:
        doc = run_live(doc, opts)
    out_dir = os.path.dirname(os.path.abspath(opts.plant))
    try:
        paths = pf.write_plant(doc, out_dir)
    except pf.FieldError as e:
        print(f"REFUSED TO WRITE: {e}", file=sys.stderr)
        return 1
    text = guided_summary(doc)
    with open(os.path.join(out_dir, "guided_summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print()
    sys.stdout.write(text)
    print(f"\nwrote {paths['yaml']}, {paths['json']}, {os.path.join(out_dir, 'guided_summary.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
