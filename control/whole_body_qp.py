"""
Whole-Body Control (WBC): strict-priority hierarchical QP (floating-base inverse dynamics).

Decision variables per tick:  x = [ qdd (nv) | tau (n_act) | f (3 * n_contact_points) ]

Hard constraints (always, "level 0"):
  * floating-base equation of motion for every non-gripper dof:  M qdd + h = S^T tau + Jc^T f
  * stance contact points do not accelerate:                      Jc qdd + Jc_dot qd = -kd Jc qd
  * actuator torque limits, joint-limit-aware qdd bounds
  * friction pyramids (mu_cmd < physical mu) and unilateral normal force (fz >= 0)

Prioritised tasks (acceleration-level, lexicographic / strict priority, Kanoun-2011 style):
  level 1: balance  = CoM position (3) + torso orientation (3)
  level 2: manipulation = hand pose(s) (6 each)
  level 3: gaze     = head camera look-at (2)
  level 4: posture  = joint-space regularisation + min effort
Each level is solved as its own QP; every already-solved level is then frozen (within `hold_tol`)
as an inequality band, so a lower-priority task can NEVER degrade a higher-priority one.

The model (M, h, Jacobians) comes from MuJoCo (the same engine as the simulator) evaluated on the
*controller's own* state estimate -- so estimation errors genuinely propagate into control.
"""
from dataclasses import dataclass, field
import numpy as np
import os
import scipy.sparse as sp
try:
    import osqp
except ImportError:          # the pure-NumPy FastQP backend (control/fast_qp.py) works without it
    osqp = None
import mujoco
from control.fast_qp import FastQP

from model.humanoid_model import (ACTUATED, GRIPPERS, make_model, joint_dadr, joint_qadr, ModelParams,
                                   nominal_q_act)


def rot_err(R, Rd):
    """World-frame orientation error vector (small-angle) taking R toward Rd."""
    return 0.5 * (np.cross(R[:, 0], Rd[:, 0]) + np.cross(R[:, 1], Rd[:, 1]) + np.cross(R[:, 2], Rd[:, 2]))


@dataclass
class Task:
    name: str
    level: int
    J: np.ndarray          # (n, nv)  acceleration-level map
    rhs: np.ndarray        # (n,)     desired (J qdd = rhs), bias already subtracted
    weight: float = 1.0
    err: float = 0.0       # reported task-space error norm (for diagnostics)


class WholeBodyController:
    def __init__(self, params: ModelParams = ModelParams(), mu_cmd: float = 0.5, hold_tol: float = 2e-3, leg_bend=None,
                 qp_backend=None):
        # qp_backend: "osqp" (original, new problem every level) | "fast" (cached warm-started dense ADMM, control/fast_qp.py)
        # default: env HUMANOID_QP, else osqp if installed, else fast
        self.qp_backend = qp_backend or os.environ.get("HUMANOID_QP") or ("osqp" if osqp is not None else "fast")
        self._fast = FastQP(eps=1e-6, max_iter=6000)
        self.p = params
        self.m, self.d = make_model(params, table=False)       # controller's private robot-only model
        m = self.m
        self.nv = m.nv
        self.act_dofs = joint_dadr(m, ACTUATED)
        self.act_q = joint_qadr(m, ACTUATED)
        self.grip_dofs = joint_dadr(m, GRIPPERS)
        self.n_act = len(ACTUATED)
        self.tau_max = np.array([m.actuator_forcerange[i, 1] for i in range(self.n_act)])
        jr = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ACTUATED])
        self.q_lo, self.q_hi = jr[:, 0], jr[:, 1]
        self.mu = mu_cmd
        self.hold_tol = hold_tol
        self.q_nom = nominal_q_act(leg_bend)
        self.fmax = 800.0
        sid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.sites = {s: [sid(f"{s}_c{k}") for k in range(4)] for s in "LR"}
        self.tcp = {"L": sid("L_tcp"), "R": sid("R_tcp")}
        self.cam_site = sid("cam_site")
        self.torso = bid("torso")
        self.pelvis = bid("pelvis")
        self.root_body = self.pelvis
        self.sole = {"L": sid("L_sole_site"), "R": sid("R_sole_site")}
        # gains
        self.kp_com, self.kd_com = 120.0, 22.0
        self.kp_rot, self.kd_rot = 150.0, 24.0
        self.kp_hand, self.kd_hand = 220.0, 30.0
        self.kp_hrot, self.kd_hrot = 120.0, 20.0
        self.kp_gaze, self.kd_gaze = 80.0, 14.0
        self.kp_post, self.kd_post = 40.0, 10.0
        self.last = {}
        self.kd_contact = 20.0     # damping on stance-foot velocity; set 0 when the state is an ESTIMATE (feeds velocity error into the constraint)

    # ------------------------------------------------------------------ state / kinematics helpers
    def set_state(self, qpos, qvel):
        self.d.qpos[:] = qpos
        self.d.qvel[:] = qvel
        mujoco.mj_forward(self.m, self.d)

    def _jac_site(self, sid):
        jp = np.zeros((3, self.nv)); jr = np.zeros((3, self.nv))
        mujoco.mj_jacSite(self.m, self.d, jp, jr, sid)
        return jp, jr

    def _jdot_qd(self, fn):
        """Finite-difference  J_dot(q,qd) @ qd  for a Jacobian function fn() evaluated on self.d."""
        d, m = self.d, self.m
        q0 = d.qpos.copy(); qd = d.qvel.copy()
        J0 = fn()
        eps = 1e-6
        mujoco.mj_integratePos(m, d.qpos, qd, eps)
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        J1 = fn()
        d.qpos[:] = q0
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        return ((J1 - J0) / eps) @ qd

    def com(self):
        return self.d.subtree_com[self.root_body].copy()

    def com_vel(self):
        jc = np.zeros((3, self.nv))
        mujoco.mj_jacSubtreeCom(self.m, self.d, jc, self.root_body)
        return jc @ self.d.qvel

    def support_center(self, contacts):
        pts = np.array([self.d.site_xpos[s] for c in contacts for s in self.sites[c]])
        return pts.mean(axis=0)

    # ------------------------------------------------------------------ task builders
    def task_com(self, com_des, vcom_des=None, level=1, weight=1.0, kp=None, kd=None, acc_ff=None):
        kp = self.kp_com if kp is None else kp; kd = self.kd_com if kd is None else kd
        vd = np.zeros(3) if vcom_des is None else vcom_des
        fn = lambda: (lambda J: J)(self._jac_subtree())
        J = fn(); jdqd = self._jdot_qd(fn)
        acc = kp * (com_des - self.com()) + kd * (vd - J @ self.d.qvel)
        if acc_ff is not None:
            acc = acc + acc_ff
        return Task("com", level, J, acc - jdqd, weight, float(np.linalg.norm(com_des - self.com())))

    def _jac_subtree(self):
        jc = np.zeros((3, self.nv))
        mujoco.mj_jacSubtreeCom(self.m, self.d, jc, self.root_body)
        return jc

    def task_torso_orient(self, Rd=np.eye(3), level=1, weight=1.0):
        bid = self.torso
        def fn():
            jp = np.zeros((3, self.nv)); jr = np.zeros((3, self.nv))
            mujoco.mj_jacBody(self.m, self.d, jp, jr, bid)
            return jr
        J = fn(); jdqd = self._jdot_qd(fn)
        R = self.d.xmat[bid].reshape(3, 3)
        e = rot_err(R, Rd)
        # MuJoCo free-joint angular qvel is body-local, but jacBody's jacr maps qvel -> world omega: consistent.
        acc = self.kp_rot * e + self.kd_rot * (-(J @ self.d.qvel))
        return Task("torso_rot", level, J, acc - jdqd, weight, float(np.linalg.norm(e)))

    def task_hand(self, side, pos_des, R_des=None, vel_des=None, level=2, weight=1.0, pos_only=False):
        sid = self.tcp[side]
        def fnp():
            return self._jac_site(sid)[0]
        def fnr():
            return self._jac_site(sid)[1]
        Jp = fnp(); jdp = self._jdot_qd(fnp)
        v = Jp @ self.d.qvel
        vd = np.zeros(3) if vel_des is None else vel_des
        e = pos_des - self.d.site_xpos[sid]
        accp = self.kp_hand * e + self.kd_hand * (vd - v)
        rows_J = [Jp]; rows_r = [accp - jdp]
        err = float(np.linalg.norm(e))
        if R_des is not None and not pos_only:
            Jr = fnr(); jdr = self._jdot_qd(fnr)
            R = self.d.site_xmat[sid].reshape(3, 3)
            er = rot_err(R, R_des)
            accr = self.kp_hrot * er - self.kd_hrot * (Jr @ self.d.qvel)
            rows_J.append(Jr); rows_r.append(accr - jdr)
        return Task(f"hand_{side}", level, np.vstack(rows_J), np.concatenate(rows_r), weight, err)

    def task_foot(self, side, pos_des, vel_des=None, acc_ff=None, R_des=None, level=2, weight=1.0, kp=400.0, kd=40.0):
        """6-D swing-foot task on the sole site (position + flat orientation)."""
        sid = self.sole[side]
        fnp = lambda: self._jac_site(sid)[0]
        fnr = lambda: self._jac_site(sid)[1]
        Jp = fnp(); jdp = self._jdot_qd(fnp); v = Jp @ self.d.qvel
        vd = np.zeros(3) if vel_des is None else vel_des
        e = pos_des - self.d.site_xpos[sid]
        accp = kp * e + kd * (vd - v) + (0 if acc_ff is None else acc_ff)
        Jr = fnr(); jdr = self._jdot_qd(fnr)
        R = self.d.site_xmat[sid].reshape(3, 3)
        er = rot_err(R, np.eye(3) if R_des is None else R_des)
        accr = 200.0 * er - 28.0 * (Jr @ self.d.qvel)
        return Task(f"foot_{side}", level, np.vstack([Jp, Jr]), np.concatenate([accp - jdp, accr - jdr]), weight,
                    float(np.linalg.norm(e)))

    def task_gaze(self, target_world, level=3, weight=1.0):
        sid = self.cam_site
        def fnr():
            return self._jac_site(sid)[1]
        Jr = fnr(); jdr = self._jdot_qd(fnr)
        R = self.d.site_xmat[sid].reshape(3, 3)
        a = R[:, 0]                       # camera forward = head-frame +x
        dvec = target_world - self.d.site_xpos[sid]
        dvec = dvec / np.linalg.norm(dvec)
        e = np.cross(a, dvec)
        P = np.eye(3) - np.outer(a, a)
        acc = self.kp_gaze * e - self.kd_gaze * (Jr @ self.d.qvel)
        return Task("gaze", level, P @ Jr, P @ (acc - jdr), weight, float(np.arcsin(min(1, np.linalg.norm(e)))))

    def task_posture(self, q_des=None, level=4, weight=1.0, joints=None):
        q_des = self.q_nom if q_des is None else q_des
        idx = np.arange(self.n_act) if joints is None else np.asarray(joints)
        J = np.zeros((len(idx), self.nv)); J[np.arange(len(idx)), self.act_dofs[idx]] = 1.0
        q = self.d.qpos[self.act_q[idx]]; qd = self.d.qvel[self.act_dofs[idx]]
        acc = self.kp_post * (q_des[idx] - q) - self.kd_post * qd
        return Task("posture", level, J, acc, weight, float(np.linalg.norm(q_des[idx] - q)))

    # ------------------------------------------------------------------ QP assembly
    def _constraints(self, contacts, relax_limits=False):
        m, d = self.m, self.d
        nv, na = self.nv, self.n_act
        pts = [s for c in contacts for s in self.sites[c]]
        nc = len(pts)
        nx = nv + na + 3 * nc
        M = np.zeros((nv, nv)); mujoco.mj_fullM(m, d, M)
        h = d.qfrc_bias.copy()
        Jc = np.zeros((3 * nc, nv)); jdqd = np.zeros(3 * nc)
        for i, s in enumerate(pts):
            fn = (lambda s=s: self._jac_site(s)[0])
            Jc[3 * i:3 * i + 3] = fn(); jdqd[3 * i:3 * i + 3] = self._jdot_qd(fn)
        free = np.setdiff1d(np.arange(nv), self.grip_dofs)
        S = np.zeros((nv, na)); S[self.act_dofs, np.arange(na)] = 1.0
        # dynamics rows (non-gripper dofs):  M qdd - S tau - Jc^T f = -h
        Aeq1 = np.hstack([M[free], -S[free], -Jc.T[free]]); beq1 = -h[free]
        # Non-slip: each stance FOOT (rigid body) has zero twist acceleration -> 6 independent rows per foot.
        # (Using 3 rows per corner point would be 12 rows/foot for a 6-dof body: linearly dependent rows whose
        #  finite-difference Jdot*qd noise makes the equality set numerically inconsistent -> false "infeasible".)
        rows_J, rows_b = [], []
        for cside in contacts:
            sid = self.sole[cside]
            fn6 = (lambda sid=sid: np.vstack(self._jac_site(sid)))
            J6 = fn6(); jd6 = self._jdot_qd(fn6)
            rows_J.append(J6); rows_b.append(-jd6 - self.kd_contact * (J6 @ d.qvel))
        J6all = np.vstack(rows_J); b6 = np.concatenate(rows_b)
        Aeq2 = np.hstack([J6all, np.zeros((len(b6), na)), np.zeros((len(b6), 3 * nc))])
        beq2 = b6
        Aeq = np.vstack([Aeq1, Aeq2]); beq = np.concatenate([beq1, beq2])
        # friction pyramids + unilateral
        rows = []
        for i in range(nc):
            o = nv + na + 3 * i
            for sgn in (1, -1):
                r = np.zeros(nx); r[o] = sgn; r[o + 2] = -self.mu; rows.append(r)
                r = np.zeros(nx); r[o + 1] = sgn; r[o + 2] = -self.mu; rows.append(r)
        Ain = np.array(rows) if rows else np.zeros((0, nx))
        lin = -np.inf * np.ones(len(Ain)); uin = np.zeros(len(Ain))
        # bounds
        lb = -np.inf * np.ones(nx); ub = np.inf * np.ones(nx)
        h_lim = 0.05
        qj = d.qpos[self.act_q]; qdj = d.qvel[self.act_dofs]
        ub_a = np.clip((self.q_hi - qj - qdj * h_lim) * 2 / h_lim ** 2, -300, 300)
        lb_a = np.clip((self.q_lo - qj - qdj * h_lim) * 2 / h_lim ** 2, -300, 300)
        if relax_limits:
            ub_a = 300.0 * np.ones_like(ub_a); lb_a = -300.0 * np.ones_like(lb_a)
        # never forbid motion *away* from a limit we are sitting on
        lb[self.act_dofs] = np.minimum(lb_a, 0) if False else lb_a
        ub[self.act_dofs] = ub_a
        # keep bounds consistent
        bad = lb[self.act_dofs] > ub[self.act_dofs]
        if bad.any():
            mid = 0.5 * (lb[self.act_dofs] + ub[self.act_dofs])
            lb[self.act_dofs[bad]] = mid[bad]; ub[self.act_dofs[bad]] = mid[bad]
        lb[self.grip_dofs] = 0.0; ub[self.grip_dofs] = 0.0
        lb[nv:nv + na] = -self.tau_max; ub[nv:nv + na] = self.tau_max
        for i in range(nc):
            o = nv + na + 3 * i
            lb[o + 2] = 0.0; ub[o + 2] = self.fmax
        return dict(nx=nx, nc=nc, pts=pts, Aeq=Aeq, beq=beq, Ain=Ain, lin=lin, uin=uin, lb=lb, ub=ub, M=M, h=h)

    def _solve_qp(self, P, q, A, l, u, x0=None):
        """OSQP with a robust retry ladder (tight+polish -> medium -> loose); or the warm-started FastQP backend."""
        if self.qp_backend == "fast":
            r = self._fast.solve(P, q, A, l, u, key=(P.shape[0], A.shape[0]))
            if r.info.status_val in (1, 2) and not np.any(np.isnan(r.x)):
                return r
            return FastQP(eps=1e-4, max_iter=20000).solve(P, q, A, l, u, x0=x0)
        r = None
        for eps, pol, it in ((1e-6, True, 6000), (1e-5, False, 10000), (1e-4, False, 20000)):
            prob = osqp.OSQP()
            prob.setup(sp.triu(sp.csc_matrix(P)).tocsc(), q, sp.csc_matrix(A), l, u, verbose=False,
                       eps_abs=eps, eps_rel=eps, max_iter=it, polish=pol, adaptive_rho=True)
            if x0 is not None:
                prob.warm_start(x=x0)
            r = prob.solve()
            if r.info.status_val in (1, 2) and r.x is not None and not np.any(np.isnan(r.x)):
                return r
        return r

    def solve(self, qpos, qvel, tasks, contacts=("L", "R")):
        """Returns dict(tau (n_act,), qdd, f, resid per level, status). Falls back to relaxed
        joint-limit bounds (physical limits are still enforced by the plant) if the base QP is infeasible."""
        self.set_state(qpos, qvel)
        out = self._solve_once(tasks, contacts, relax_limits=False)
        if out["status"] != "ok":
            out = self._solve_once(tasks, contacts, relax_limits=True)
            out["relaxed_limits"] = True
        return out

    def _solve_once(self, tasks, contacts, relax_limits):
        c = self._constraints(contacts, relax_limits)
        nx, nv, na = c["nx"], self.nv, self.n_act
        # base constraint stack
        A = [c["Aeq"], c["Ain"], np.eye(nx)]
        l = [c["beq"], c["lin"], c["lb"]]
        u = [c["beq"], c["uin"], c["ub"]]
        reg = np.zeros(nx)
        reg[:nv] = 1e-5; reg[nv:nv + na] = 1e-6; reg[nv + na:] = 1e-6
        levels = sorted(set(t.level for t in tasks))
        x = None
        resid = {}
        status = "ok"
        for lv in levels:
            ts = [t for t in tasks if t.level == lv]
            Ak = []; bk = []
            for t in ts:
                Jx = np.zeros((t.J.shape[0], nx)); Jx[:, :nv] = np.sqrt(t.weight) * t.J
                Ak.append(Jx); bk.append(np.sqrt(t.weight) * t.rhs)
            Ak = np.vstack(Ak); bk = np.concatenate(bk)
            reg_k = reg.copy()
            if lv == levels[-1]:       # lowest level also minimises effort / force
                reg_k[nv:nv + na] += 2e-5; reg_k[nv + na:] += 2e-5
            P = 2 * Ak.T @ Ak + np.diag(2 * reg_k)
            q = -2 * Ak.T @ bk
            Afull = np.vstack(A); lfull = np.concatenate(l); ufull = np.concatenate(u)
            r = self._solve_qp(P, q, Afull, lfull, ufull, x)
            if r.info.status_val not in (1, 2) or r.x is None or np.any(np.isnan(r.x)):
                status = "ok" if x is not None else f"level{lv}:{r.info.status}"
                if x is not None:
                    resid[f"level{lv}_dropped"] = 1.0       # keep last feasible higher-priority solution
                break
            x = r.x
            # freeze this level
            val = Ak @ x
            tol = self.hold_tol + 1e-3 * np.abs(val)
            A.append(Ak); l.append(val - tol); u.append(val + tol)
            resid[lv] = float(np.linalg.norm(Ak @ x - bk))
        if x is None:
            return dict(tau=np.zeros(na), qdd=np.zeros(nv), f=np.zeros(0), resid=resid, status=status, contact_pts=c["pts"])
        out = dict(tau=x[nv:nv + na].copy(), qdd=x[:nv].copy(), f=x[nv + na:].copy(), resid=resid,
                   status=status, contact_pts=c["pts"], nc=c["nc"])
        self.last = out
        return out
