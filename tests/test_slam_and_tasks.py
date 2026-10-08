import numpy as np
from simulation.slam_sim import run
from tasks.symbolic_planner import KeywordCommandParser, plan
from tasks.behavior_tree import Sequence, Fallback, Retry, Condition, Node, Status


def test_slam_cuts_drift_by_more_than_10x():
    r, _ = run(seed=7, verbose=False)
    assert r["ate_slam_m"] < 0.2 and r["ate_odometry_m"] > 10 * r["ate_slam_m"]
    assert r["map_f1_slam"][0] > 0.6 and r["map_f1_slam"][0] > r["map_f1_odometry"][0] + 0.3
    assert r["n_landmarks_est"] <= r["n_landmarks_true"] + 2          # association did not explode into duplicates


def test_planner_and_parser():
    g = KeywordCommandParser().parse("pick the red block")
    p = plan(g.predicates)
    assert p == ["LookAtTable", "DetectObject", "OpenGripper", "MoveToPregrasp", "MoveToGrasp", "CloseGripper", "LiftObject"]
    assert plan(frozenset(["flying"])) is None
    assert KeywordCommandParser().parse("neela block uthao").target_label == "blue"


class _Ctx:
    t = 0.0
    def log(self, s): pass


class Cnt(Node):
    def __init__(self, results): self.results, self.i, self.name = results, 0, "cnt"
    def tick(self, ctx):
        r = self.results[min(self.i, len(self.results) - 1)]; self.i += 1; return r


def test_behaviour_tree_semantics():
    ctx = _Ctx()
    assert Sequence([Cnt([Status.SUCCESS]), Cnt([Status.SUCCESS])]).tick(ctx) == Status.SUCCESS
    assert Sequence([Cnt([Status.SUCCESS]), Cnt([Status.FAILURE])]).tick(ctx) == Status.FAILURE
    assert Fallback([Cnt([Status.FAILURE]), Cnt([Status.SUCCESS])]).tick(ctx) == Status.SUCCESS
    r = Retry(Cnt([Status.FAILURE, Status.FAILURE, Status.SUCCESS]), 3)
    assert [r.tick(ctx) for _ in range(3)] == [Status.RUNNING, Status.RUNNING, Status.SUCCESS]
