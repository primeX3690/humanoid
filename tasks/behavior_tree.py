"""Minimal, dependency-free behaviour tree (Sequence / Fallback / Retry / Condition / Action) ticked at control rate."""
from enum import Enum


class Status(Enum):
    SUCCESS = 1
    FAILURE = 2
    RUNNING = 3


class Node:
    name = "node"
    def tick(self, ctx):
        raise NotImplementedError
    def reset(self):
        pass


class Sequence(Node):
    def __init__(self, children, name="seq"):
        self.children, self.name, self.i = children, name, 0
    def tick(self, ctx):
        while self.i < len(self.children):
            s = self.children[self.i].tick(ctx)
            if s == Status.RUNNING:
                return s
            if s == Status.FAILURE:
                ctx.log(f"[BT] {self.name}: child '{self.children[self.i].name}' FAILED")
                self.i = 0
                return s
            ctx.log(f"[BT] {self.children[self.i].name}: SUCCESS (t={ctx.t:.2f}s)")
            self.i += 1
        self.i = 0
        return Status.SUCCESS
    def reset(self):
        self.i = 0
        for c in self.children:
            c.reset()


class Fallback(Node):
    def __init__(self, children, name="fallback"):
        self.children, self.name, self.i = children, name, 0
    def tick(self, ctx):
        while self.i < len(self.children):
            s = self.children[self.i].tick(ctx)
            if s == Status.RUNNING:
                return s
            if s == Status.SUCCESS:
                self.i = 0
                return s
            self.children[self.i].reset()
            self.i += 1
        self.i = 0
        return Status.FAILURE
    def reset(self):
        self.i = 0
        for c in self.children:
            c.reset()


class Retry(Node):
    def __init__(self, child, n, name="retry"):
        self.child, self.n, self.k, self.name = child, n, 0, name
    def tick(self, ctx):
        s = self.child.tick(ctx)
        if s == Status.FAILURE:
            self.k += 1
            self.child.reset()
            return Status.RUNNING if self.k < self.n else Status.FAILURE
        if s == Status.SUCCESS:
            self.k = 0
        return s
    def reset(self):
        self.k = 0; self.child.reset()


class Condition(Node):
    def __init__(self, fn, name="cond"):
        self.fn, self.name = fn, name
    def tick(self, ctx):
        return Status.SUCCESS if self.fn(ctx) else Status.FAILURE
