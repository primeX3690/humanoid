"""What the humanoid-loco hardware (Dynamixel PH54-200: 44.7 Nm continuous, 33.1 rpm no-load) can and cannot do."""
import numpy as np
from planning.centroidal_jump_to import JumpOptimizer, repo_jump_config


def test_repo_actuators_can_jump_on_torque_alone_but_not_at_their_speed():
    torque_only = JumpOptimizer(repo_jump_config(omega_max=0.0))
    r = torque_only.solve(0.0, T_flight_min=0.3)
    assert r["ok"]
    v = torque_only.verify(r)
    assert v["peak_knee_torque_Nm_both_legs"] <= 2 * 44.7 + 1.0
    assert v["peak_stance_joint_speed_rad_s"] > 3 * 3.47                    # needs >3x the actuator's speed
    both = JumpOptimizer(repo_jump_config())
    assert not both.solve(0.0, T_flight_min=0.3)["ok"]                      # with the speed limit it is infeasible
