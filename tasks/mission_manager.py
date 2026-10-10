"""
Mission manager with failure recovery: executes a symbolic plan skill-by-skill, and when a skill fails it applies that skill's
FAILURE EFFECTS to the world state and RE-PLANS toward the same goal (v2: a failed skill just ended the mission).

`EXTENDED_OPS` adds locomotion + delivery to tasks.symbolic_planner.DEFAULT_OPS:  walk to the object, carry to a destination, release.
The executor is any callable (action_name, state) -> bool | (bool, dict)  - a behaviour-tree runner, a sim, or a stub in tests.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, FrozenSet, List
from tasks.symbolic_planner import DEFAULT_OPS, Op, _op, plan

EXTENDED_OPS: List[Op] = [o for o in DEFAULT_OPS if o.name != "MoveToPregrasp"] + [
    _op("WalkToObject", ["object_localised"], ["near_object"]),
    _op("MoveToPregrasp", ["object_localised", "gripper_open", "near_object"], ["at_pregrasp"], ["at_home"]),
    _op("CarryTo", ["holding_object"], ["at_destination"], ["near_object"]),
    _op("Release", ["holding_object", "at_destination"], ["object_delivered", "gripper_open"], ["holding_object", "gripper_closed", "grasped"]),
]

# what is no longer true after a skill FAILS (the world model is pessimistic: it re-observes whatever might have changed)
FAILURE_EFFECTS = {
    "DetectObject": dict(delete={"object_localised"}, add=set()),
    "MoveToPregrasp": dict(delete={"at_pregrasp", "object_localised"}, add=set()),
    "MoveToGrasp": dict(delete={"at_grasp", "at_pregrasp", "object_localised"}, add=set()),
    "CloseGripper": dict(delete={"at_grasp", "at_pregrasp", "grasped", "gripper_closed", "object_localised"}, add={"gripper_open"}),
    "LiftObject": dict(delete={"grasped", "holding_object", "gripper_closed", "at_grasp", "at_pregrasp", "object_localised", "near_object"}, add={"gripper_open"}),
    "CarryTo": dict(delete={"holding_object", "grasped", "gripper_closed", "at_destination", "object_localised", "near_object"}, add={"gripper_open"}),
    "WalkToObject": dict(delete={"near_object"}, add=set()),
}


@dataclass
class MissionReport:
    success: bool
    replans: int
    executed: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    final_state: FrozenSet[str] = frozenset()
    reason: str = ""


class MissionManager:
    def __init__(self, executor: Callable, ops=None, max_replans=3, max_actions=60, per_action_retries=1):
        self.exec, self.ops, self.max_replans, self.max_actions, self.retries = executor, ops or EXTENDED_OPS, max_replans, max_actions, per_action_retries
        self.by_name = {o.name: o for o in self.ops}

    def run(self, goal: FrozenSet[str], state: FrozenSet[str] = frozenset()) -> MissionReport:
        rep = MissionReport(False, 0, final_state=state)
        goal = frozenset(goal); state = frozenset(state)
        p = plan(goal, state, self.ops)
        if p is None:
            rep.reason = "no plan from the initial state"; return rep
        n_actions = 0
        while p:
            act = p.pop(0); n_actions += 1
            if n_actions > self.max_actions: rep.reason = "action budget exhausted"; break
            ok = False
            for _ in range(1 + self.retries):                         # cheap local retry first
                r = self.exec(act, state)
                ok = r[0] if isinstance(r, tuple) else bool(r)
                if ok: break
            if ok:
                op = self.by_name[act]; state = frozenset((state - op.delete) | op.add)
                rep.executed.append(act)
                if goal <= state: rep.success = True; break
                continue
            rep.failures.append(act)
            fx = FAILURE_EFFECTS.get(act, dict(delete=set(), add=set()))
            state = frozenset((state - fx["delete"]) | fx["add"])
            rep.replans += 1
            if rep.replans > self.max_replans: rep.reason = f"gave up after {self.max_replans} re-plans (last failure: {act})"; break
            p = plan(goal, state, self.ops)
            if p is None: rep.reason = f"no plan after failure of {act}"; break
        else:
            rep.success = goal <= state
        if goal <= state: rep.success = True
        rep.final_state = state
        return rep
