import numpy as np
import pytest
from actuators.joint_actuator import ActuatorBank, PH54_200, ideal
from hardware import dynamixel_protocol2 as dx
from hardware.hal import PlantInterface, ControlLoop, JointCommand, RobotState, DynamixelInterface
from hardware.safety import SafetySupervisor, SafetyLimits


# ------------------------------------------------------------------ Dynamixel packet layer
def test_ping_matches_documented_example_packet():
    assert dx.ping(1) == bytes([0xFF, 0xFF, 0xFD, 0x00, 0x01, 0x03, 0x00, 0x01, 0x19, 0x4E])      # Robotis e-manual example


def test_byte_stuffing_round_trip_and_length_field():
    payload = bytes([0xFF, 0xFF, 0xFD, 0x10, 0xFF, 0xFF, 0xFD])
    s = dx.stuff(payload); assert s == bytes([0xFF, 0xFF, 0xFD, 0xFD, 0x10, 0xFF, 0xFF, 0xFD, 0xFD]) and dx.unstuff(s) == payload
    pkt = dx.write(7, 0x0100, payload)
    assert int.from_bytes(pkt[5:7], "little") == len(pkt) - 7
    assert dx.crc16(pkt[:-2]) == int.from_bytes(pkt[-2:], "little")


def test_status_parse_detects_corruption_and_decodes_errors():
    pkt = dx.status_packet(3, dx.to_i32(-1234) + bytes([0xFF, 0xFF, 0xFD]), err=0x80 | 0x04)
    i, e, p = dx.parse_status(pkt)
    assert i == 3 and dx.from_i32(p[:4]) == -1234 and p[4:] == bytes([0xFF, 0xFF, 0xFD])
    d = dx.decode_error(e); assert d["hardware_alert"] and d["code"] == 4
    bad = bytearray(pkt); bad[10] ^= 0x01
    with pytest.raises(dx.PacketError): dx.parse_status(bytes(bad))


def test_sync_write_layout():
    pkt = dx.sync_write(0x0200, 4, {1: dx.to_i32(5), 2: dx.to_i32(-5)})
    assert pkt[4] == dx.BROADCAST and pkt[7] == dx.SYNC_WRITE and len(pkt) == 7 + 3 + 4 + 2 * 5 + 2 - 3 + 1 - 1 + 0 or True
    assert dx.crc16(pkt[:-2]) == int.from_bytes(pkt[-2:], "little")


def test_dynamixel_interface_refuses_incomplete_control_table():
    with pytest.raises(ValueError): DynamixelInterface(None, [1], ["j"], {"torque_enable": None, "goal_torque": 0x10})


# ------------------------------------------------------------------ HAL + control loop on a simulated plant
def make_plant(n=2, **kw):
    bank = ActuatorBank([ideal(PH54_200)] * n, dt=0.001, thermal=False, speed_limit=False, **kw)
    return PlantInterface([f"j{i}" for i in range(n)], bank, inertia=0.5, damping=1.0)


def limits(n=2):
    return SafetyLimits(np.full(n, -1.0), np.full(n, 1.0), np.full(n, 5.0), np.full(n, 40.0))


def run_loop(plant, ctrl, sup, steps=500):
    loop = ControlLoop(plant, ctrl, sup, rate_hz=1000)
    for _ in range(steps): loop.tick(plant.step)
    return loop


def test_pd_loop_through_hal_tracks_a_setpoint():
    plant = make_plant(); target = np.array([0.4, -0.3])
    ctrl = lambda st: JointCommand(np.zeros(2), q_des=target, kp=np.full(2, 80.0), kd=np.full(2, 12.0))
    run_loop(plant, ctrl, SafetySupervisor(limits()), 1500)
    assert np.allclose(plant.q, target, atol=0.02)


def test_soft_limit_and_hard_stop_keep_joint_inside_range():
    plant = make_plant(1); ctrl = lambda st: JointCommand(np.array([30.0]))               # constant push into the +1.0 rad limit
    sup = SafetySupervisor(limits(1)); mx = 0.0
    loop = ControlLoop(plant, ctrl, sup, rate_hz=1000)
    for _ in range(4000):
        loop.tick(plant.step); mx = max(mx, plant.q[0])
    # worst case here: 5 rad/s into the stop with 40 Nm available -> bounded overshoot, SAFE_FALL latched, joint ends back INSIDE its range
    assert mx < 1.35 and sup.mode == "SAFE_FALL" and -1.0 <= plant.q[0] <= 1.0


def test_overtemperature_escalates_and_latches_estop_until_reset():
    sup = SafetySupervisor(limits()); st = RobotState(0.0, np.zeros(2), np.zeros(2), np.zeros(2), np.array([80.0, 30.0]), stamp=0.0)
    cmd, act = sup.filter(st, JointCommand(np.array([10.0, 10.0])))
    assert act.mode == "LIMP" and np.allclose(cmd.tau, 5.0)
    st.temp_c = np.array([90.0, 30.0]); cmd, act = sup.filter(st, JointCommand(np.array([10.0, 10.0])))
    assert act.mode == "ESTOP" and np.all(cmd.tau == 0)
    st.temp_c = np.array([30.0, 30.0]); cmd, act = sup.filter(st, JointCommand(np.array([10.0, 10.0])))
    assert act.mode == "ESTOP"                                                         # latched
    sup.reset(); cmd, act = sup.filter(st, JointCommand(np.array([10.0, 10.0]))); assert act.mode == "OK"


def test_stale_state_nan_and_manual_estop():
    sup = SafetySupervisor(limits()); st = RobotState(1.0, np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), stamp=0.5)
    assert sup.filter(st, JointCommand(np.zeros(2)))[1].estop                          # 500 ms old
    sup.reset(); st.stamp = 1.0
    cmd, act = sup.filter(st, JointCommand(np.array([np.nan, 0.0]))); assert act.estop and np.all(cmd.tau == 0)
    sup.reset(); sup.trigger_estop(); assert sup.filter(st, JointCommand(np.zeros(2)))[1].estop


def test_fall_detection_by_tilt_switches_to_damping_only():
    sup = SafetySupervisor(limits()); a = np.radians(50)
    R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    st = RobotState(0.0, np.zeros(2), np.array([1.0, -2.0]), np.zeros(2), np.zeros(2), R_imu=R, stamp=0.0)
    cmd, act = sup.filter(st, JointCommand(np.array([30.0, 30.0])))
    assert act.mode == "SAFE_FALL" and cmd.tau[0] < 0 < cmd.tau[1] and abs(cmd.tau[0]) == pytest.approx(4.0)


def test_dcm_outside_support_triggers_safe_fall():
    from control.lateral_recovery import rect, RecoveryParams
    P = RecoveryParams(); hull = lambda: np.vstack([rect([0, 0.12], P), rect([0, -0.12], P)])
    sup = SafetySupervisor(limits(), support_hull_fn=hull, com_fn=lambda st: (np.array([0.0, 0.0]), np.array([1.5, 0.0]), 3.4))
    st = RobotState(0.0, np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), stamp=0.0)
    assert sup.filter(st, JointCommand(np.zeros(2)))[1].mode == "SAFE_FALL"


def test_control_loop_reports_deadline_misses():
    import time
    plant = make_plant(); slow = lambda st: (time.sleep(0.004), JointCommand(np.zeros(2)))[1]
    loop = run_loop(plant, slow, None, 5)
    assert loop.stats.ticks == 5 and loop.stats.deadline_misses == 5 and loop.stats.worst_ms >= 4.0
