from tasks.mission_manager import MissionManager, EXTENDED_OPS
from tasks.symbolic_planner import plan, DEFAULT_OPS


class Flaky:
    """Fails the first `n` attempts of `action`, otherwise succeeds."""
    def __init__(self, action, n): self.action, self.left, self.calls = action, n, []
    def __call__(self, a, state):
        self.calls.append(a)
        if a == self.action and self.left > 0:
            self.left -= 1; return False
        return True


def test_extended_plan_walks_grasps_carries_and_releases():
    p = plan(frozenset({"object_delivered"}), frozenset(), EXTENDED_OPS, max_depth=16)
    assert p is not None and p.index("WalkToObject") < p.index("MoveToPregrasp") < p.index("LiftObject") < p.index("CarryTo") < p.index("Release")


def test_clean_run_succeeds_without_replanning():
    m = MissionManager(Flaky("none", 0)); r = m.run(frozenset({"object_delivered"}))
    assert r.success and r.replans == 0 and r.executed[-1] == "Release"


def test_dropped_object_triggers_replan_and_mission_still_succeeds():
    ex = Flaky("LiftObject", 2)                        # 1 try + 1 local retry fail -> replan; then succeeds
    r = MissionManager(ex, per_action_retries=1).run(frozenset({"object_delivered"}))
    assert r.success and r.replans == 1 and r.failures == ["LiftObject"]
    assert ex.calls.count("DetectObject") >= 2          # object had to be re-localised after the drop


def test_gives_up_after_max_replans_with_reason():
    r = MissionManager(Flaky("CloseGripper", 99), max_replans=2, per_action_retries=0).run(frozenset({"holding_object"}))
    assert not r.success and r.replans == 3 and "gave up" in r.reason


def test_unplannable_goal_is_reported_not_raised():
    r = MissionManager(Flaky("x", 0), ops=DEFAULT_OPS).run(frozenset({"object_delivered"}))
    assert not r.success and "no plan" in r.reason


def test_local_retry_absorbs_a_single_glitch_without_replanning():
    r = MissionManager(Flaky("CloseGripper", 1), per_action_retries=1).run(frozenset({"holding_object"}))
    assert r.success and r.replans == 0
