"""
STRIPS-style symbolic task planner (breadth-first forward search) + rule-based goal parser.

The planner turns a goal such as  holding(obj)  into an ordered list of skills; `tasks/pick_skills.py` maps each skill name to
a behaviour-tree node. `CommandParser` is the ONE hook where a local LLM (e.g. Ollama) will plug in later: anything that
returns a `Goal` works. It is intentionally a plain keyword parser today -- no LLM is used anywhere in this codebase.
"""
from dataclasses import dataclass, field
from collections import deque
from typing import FrozenSet, List, Optional, Protocol


@dataclass(frozen=True)
class Op:
    name: str
    pre: FrozenSet[str]
    add: FrozenSet[str]
    delete: FrozenSet[str] = frozenset()


def _op(name, pre, add, delete=()):
    return Op(name, frozenset(pre), frozenset(add), frozenset(delete))


DEFAULT_OPS = [
    _op("LookAtTable", [], ["gazing_table"]),
    _op("ScanForObject", ["gazing_table"], ["gazing_table_scanned"]),
    _op("DetectObject", ["gazing_table"], ["object_localised"]),
    _op("OpenGripper", [], ["gripper_open"], ["gripper_closed"]),
    _op("MoveToPregrasp", ["object_localised", "gripper_open"], ["at_pregrasp"], ["at_home"]),
    _op("MoveToGrasp", ["at_pregrasp"], ["at_grasp"], ["at_pregrasp"]),
    _op("CloseGripper", ["at_grasp"], ["gripper_closed", "grasped"], ["gripper_open"]),
    _op("LiftObject", ["grasped"], ["holding_object"]),
    _op("PlaceDown", ["holding_object"], ["object_placed", "gripper_open"], ["holding_object", "gripper_closed", "grasped"]),
]


@dataclass
class Goal:
    predicates: FrozenSet[str]
    target_label: str = "red"
    hand: Optional[str] = None
    text: str = ""


class CommandParser(Protocol):
    def parse(self, text: str) -> Goal: ...


class KeywordCommandParser:
    """Deterministic parser: colour word -> target label, 'pick/grab/lift/uthao' -> holding_object, 'look/dekho' -> gazing."""
    def parse(self, text: str) -> Goal:
        t = text.lower()
        label = "blue" if ("blue" in t or "neela" in t) else "red"
        hand = "L" if ("left" in t or "baya" in t) else ("R" if ("right" in t or "daya" in t) else None)
        if any(w in t for w in ("place", "rakh", "put")):
            goal = frozenset(["object_placed"])
        elif any(w in t for w in ("pick", "grab", "lift", "uthao", "utha", "take")):
            goal = frozenset(["holding_object"])
        elif any(w in t for w in ("look", "dekho", "find", "locate")):
            goal = frozenset(["object_localised"])
        else:
            raise ValueError(f"cannot parse command: {text!r}")
        return Goal(goal, label, hand, text)


def plan(goal: FrozenSet[str], start: FrozenSet[str] = frozenset(), ops: List[Op] = DEFAULT_OPS, max_depth=12) -> Optional[List[str]]:
    q = deque([(start, [])]); seen = {start}
    while q:
        s, p = q.popleft()
        if goal <= s:
            return p
        if len(p) >= max_depth:
            continue
        for op in ops:
            if op.pre <= s:
                ns = frozenset((s - op.delete) | op.add)
                if ns not in seen:
                    seen.add(ns); q.append((ns, p + [op.name]))
    return None
