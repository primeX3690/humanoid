"""
demo/run_walk_demo.py

Generates an 8-step walk using the tuned defaults (see docs/BUGS_FOUND.md
for why these specific defaults), plots the CoM trajectory, the tracked
ZMP vs. its reference, and the footstep/support-polygon layout, and
prints the real stability metric (ZMP-to-support-polygon margin) for the
whole walk - not just "it ran without crashing."

Run: python3 -m demo.run_walk_demo
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from planning.footstep_planner import GaitParams, support_polygon_at
from simulation.walk_simulator import simulate_walk
from simulation.joint_trajectory_generator import generate_joint_trajectory
from simulation.torque_analysis import analyze_torque_feasibility
from actuators.motor_specs import DYNAMIXEL_MX106, DYNAMIXEL_H54P_200


def main():
    gait = GaitParams(n_steps=8)
    result = simulate_walk(gait)
    jt = generate_joint_trajectory(result, gait)
    torque_mx106 = analyze_torque_feasibility(result, jt, gait, motor=DYNAMIXEL_MX106)
    torque_h54p = analyze_torque_feasibility(result, jt, gait, motor=DYNAMIXEL_H54P_200)

    n_violations = int(np.sum(~result.zmp_in_support))
    margins = []
    for k in range(len(result.t)):
        xmin, xmax, ymin, ymax = support_polygon_at(result.t[k], gait, result.footsteps)
        mx = min(result.zmp_x[k] - xmin, xmax - result.zmp_x[k])
        my = min(result.zmp_y[k] - ymin, ymax - result.zmp_y[k])
        margins.append(min(mx, my))
    min_margin_mm = min(margins) * 1000

    from planning.footstep_planner import foot_target_trajectories
    _, left_target, right_target = foot_target_trajectories(gait, result.footsteps)
    left_track_err_mm = np.linalg.norm(jt.left_foot_xyz - left_target, axis=1).max() * 1000
    right_track_err_mm = np.linalg.norm(jt.right_foot_xyz - right_target, axis=1).max() * 1000

    print("=" * 70)
    print("NEXUS Humanoid Locomotion v1 - LIPM + ZMP Preview Control walk demo")
    print("=" * 70)
    print(f"Steps planned: {gait.n_steps}, step length: {gait.step_length_m}m, "
          f"stance width: {gait.step_width_m}m")
    print(f"Total walk duration: {result.t[-1]:.2f}s, {len(result.t)} timesteps @ {gait.dt*1000:.0f}ms")
    print(f"Final CoM position: x={result.com_x[-1]:.3f}m, y={result.com_y[-1]:.3f}m")
    print(f"Support-polygon violations: {n_violations}/{len(result.t)}")
    print(f"Minimum ZMP-to-support-polygon-edge margin: {min_margin_mm:.2f}mm "
          f"({'STABLE - real, non-zero margin' if min_margin_mm > 0 else 'UNSTABLE - would fall'})")
    print(f"Leg IK: left converged {np.sum(jt.left_ik_converged)}/{len(jt.t)}, "
          f"right converged {np.sum(jt.right_ik_converged)}/{len(jt.t)}")
    print(f"Max foot tracking error vs. plan: left={left_track_err_mm:.3f}mm, "
          f"right={right_track_err_mm:.3f}mm")
    print("-" * 70)
    print("Actuator feasibility (Jacobian-transpose torque, 25kg robot assumption):")
    for name, tr in [("Dynamixel MX-106 (hobby servo)", torque_mx106),
                      ("Dynamixel PRO Plus H54P-200 (research-grade)", torque_h54p)]:
        print(f"  {name}:")
        print(f"    hip pitch:  peak {tr.hip_feasibility.peak_torque_nm:6.1f} Nm  "
              f"(continuous margin {tr.hip_feasibility.peak_torque_margin_vs_continuous:.2f}x, "
              f"{'OK' if tr.hip_feasibility.torque_ok_continuous else 'EXCEEDS RATING'})")
        print(f"    knee:       peak {tr.knee_feasibility.peak_torque_nm:6.1f} Nm  "
              f"(continuous margin {tr.knee_feasibility.peak_torque_margin_vs_continuous:.2f}x, "
              f"{'OK' if tr.knee_feasibility.torque_ok_continuous else 'EXCEEDS RATING'})")
    print("=" * 70)

    fig, axes = plt.subplots(4, 1, figsize=(10, 16))

    ax = axes[0]
    ax.plot(result.com_x, result.com_y, label="CoM trajectory", color="tab:blue", linewidth=2)
    ax.plot(result.zmp_x, result.zmp_y, label="ZMP (tracked)", color="tab:red", linewidth=1, alpha=0.7)
    for step in result.footsteps:
        color = "tab:orange" if step.side == "right" else "tab:green"
        ax.add_patch(Rectangle((step.x - 0.12, step.y - 0.05), 0.24, 0.10,
                                fill=False, edgecolor=color, linewidth=2))
        ax.text(step.x, step.y, step.side[0].upper(), ha="center", va="center", fontsize=8)
    ax.set_xlabel("x (m, forward)")
    ax.set_ylabel("y (m, lateral)")
    ax.set_title("CoM + ZMP trajectory over footstep plan (top-down view)")
    ax.legend()
    ax.axis("equal")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(result.t, result.zmp_x, label="ZMP x (tracked)", color="tab:red")
    ax.plot(result.t, result.zmp_ref_x, "--", label="ZMP x (reference)", color="tab:red", alpha=0.5)
    ax.plot(result.t, result.com_x, label="CoM x", color="tab:blue")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("x position (m)")
    ax.set_title("Sagittal (forward) axis: ZMP tracking + CoM motion")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(result.t, result.zmp_y, label="ZMP y (tracked)", color="tab:red")
    ax.plot(result.t, result.zmp_ref_y, "--", label="ZMP y (reference)", color="tab:red", alpha=0.5)
    ax.plot(result.t, result.com_y, label="CoM y", color="tab:blue")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("y position (m)")
    ax.set_title("Lateral axis: ZMP tracking + CoM motion (the axis the real bugs were in - see docs/BUGS_FOUND.md)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[3]
    ax.plot(jt.t, np.degrees(jt.left_joint_angles[:, 3]), label="left knee", color="tab:green")
    ax.plot(jt.t, np.degrees(jt.right_joint_angles[:, 3]), label="right knee", color="tab:orange")
    ax.plot(jt.t, np.degrees(jt.left_joint_angles[:, 2]), "--", label="left hip pitch", color="tab:green", alpha=0.5)
    ax.plot(jt.t, np.degrees(jt.right_joint_angles[:, 2]), "--", label="right hip pitch", color="tab:orange", alpha=0.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("joint angle (deg)")
    ax.set_title("Real solved leg joint angles (6-DOF IK, kinematics/leg_ik.py) - alternating swing/stance visible")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = "results/walk_demo.png"
    import os
    os.makedirs("results", exist_ok=True)
    plt.savefig(out_path, dpi=120)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()

