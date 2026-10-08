"""
Dynamic / athletic motion: centroidal trajectory optimisation of a jump (stance -> flight -> landing stance).

This replaces the quasi-static "LIPM/ZMP + constant CoM height" assumption: CoM height varies freely, the robot leaves
the ground, and the optimiser decides push-off / landing forces under REAL limits:
  * friction cone (commanded mu, not physical mu)       * unilateral normal force
  * centre-of-pressure inside the foot (toe/heel limits) * leg kinematic reach (hip-ankle distance in [dmin, dmax])
  * LEG TORQUE LIMITS via planar 2-link J^T F (knee/hip) -- the same actuator limits as the MuJoCo model
Sagittal-plane single-rigid-body (zero angular momentum) model, direct multiple shooting + IPOPT (CasADi).
Honest scope: this gives a feasible, dynamically-consistent CENTROIDAL plan (+ required joint torques). Tracking it with
the full 32-dof WBC through the flight phase is the next integration step -- it is NOT claimed here.
"""
from dataclasses import dataclass
import numpy as np
import casadi as ca


@dataclass
class JumpConfig:
    mass: float = 27.4
    g: float = 9.81
    z0: float = 0.916              # standing CoM height
    foot_toe: float = 0.13         # CoP limits relative to foot reference point (ankle x)
    foot_heel: float = -0.07
    ankle_h: float = 0.05
    d_min: float = 0.35            # hip-to-ankle distance limits
    d_max: float = 0.88
    L: float = 0.45                # thigh = shank
    mu: float = 0.5
    tau_knee: float = 2 * 130.0    # two legs share the load
    tau_hip: float = 2 * 130.0
    tau_ankle: float = 2 * 70.0
    omega_max: float = 0.0         # joint speed limit rad/s for the stance phases (0 = unconstrained)
    n1: int = 24
    nf: int = 14
    n2: int = 24


def leg_torques(rx, rz, Fx, Fz, L):
    """Planar 2-link leg (hip->knee->ankle), foot force F applied at the ankle by the ground.
    r = hip - foot (CoM ~ hip). Returns (tau_hip, tau_knee) magnitudes via tau = J^T F (J of ankle wrt hip/knee angle)."""
    d2 = rx * rx + rz * rz
    c2 = (d2 - 2 * L * L) / (2 * L * L)                 # cos(knee flexion)
    s2 = ca.sqrt(ca.fmax(1 - c2 * c2, 1e-6))            # knee bent -> positive flexion
    # ankle position relative to hip in hip frame (x fwd, z down is negative): pos = L*[sin a1 + sin(a1+a2), -cos a1 - cos(a1+a2)]
    # Use geometric torques: knee torque = F_perp-moment about knee; hip torque = moment about hip.
    # hip->ankle vector p = -r. Knee position: choose bend direction (knee forward). Triangle with sides L, L, d.
    d = ca.sqrt(d2)
    h = ca.sqrt(ca.fmax(L * L - 0.25 * d2, 1e-6))
    ux, uz = -rx / d, -rz / d                            # unit hip->ankle
    nx, nz = -uz, ux                                      # perpendicular (rotate +90deg)
    sgn = 1.0
    kx = 0.5 * d * ux + sgn * h * nx                      # knee relative to hip
    kz = 0.5 * d * uz + sgn * h * nz
    # ground force on foot F acts at ankle (relative to hip: ax, az)
    ax, az = -rx, -rz
    tau_hip = (ax * Fz - az * Fx)                         # moment of F about hip
    tau_knee = ((ax - kx) * Fz - (az - kz) * Fx)          # moment of F about knee
    return tau_hip, tau_knee


def leg_angles(rx, rz, L):
    """Planar leg joint angles from hip->foot geometry (r = hip - foot-point). Returns (thigh angle from vertical,
    knee flexion, ankle angle for a flat foot). Same geometry/branch as leg_torques (knee forward)."""
    d2 = rx * rx + rz * rz
    d = ca.sqrt(d2)
    h = ca.sqrt(ca.fmax(L * L - 0.25 * d2, 1e-6))
    ux, uz = -rx / d, -rz / d
    nx, nz = -uz, ux
    kx = 0.5 * d * ux + h * nx
    kz = 0.5 * d * uz + h * nz
    psi = ca.atan2(kx, -kz)                                   # thigh from vertical
    c2 = (d2 - 2 * L * L) / (2 * L * L)
    phi = ca.acos(ca.fmin(ca.fmax(c2, -1 + 1e-9), 1 - 1e-9))  # knee flexion
    return psi, phi, psi - phi


def repo_jump_config(**kw) -> "JumpConfig":
    """Limits of the humanoid-loco hardware choice: 2 legs x PH54-200 (44.7 Nm continuous each, 33.1 rpm no-load)."""
    d = dict(tau_knee=2 * 44.7, tau_hip=2 * 44.7, tau_ankle=2 * 44.7, omega_max=33.1 * 2 * np.pi / 60)
    d.update(kw)
    return JumpConfig(**d)


class JumpOptimizer:
    def __init__(self, cfg: JumpConfig = JumpConfig()):
        self.c = cfg

    def solve(self, dx_target, T_flight_min=0.0, z_apex_min=None, verbose=False):
        c = self.c
        opti = ca.Opti()
        mg = c.mass * c.g
        T1 = opti.variable(); Tf = opti.variable(); T2 = opti.variable()
        opti.subject_to([T1 >= 0.15, T1 <= 0.8, Tf >= max(T_flight_min, 0.05), Tf <= 0.8, T2 >= 0.15, T2 <= 1.0])
        # states [x, z, vx, vz]
        X1 = opti.variable(4, c.n1 + 1); F1 = opti.variable(2, c.n1)
        Xf = opti.variable(4, c.nf + 1)
        X2 = opti.variable(4, c.n2 + 1); F2 = opti.variable(2, c.n2)
        p1 = 0.0                              # take-off foot x (reference)
        p2 = opti.variable()                  # landing foot x

        def f(x, F):
            return ca.vertcat(x[2], x[3], F[0] / c.mass, F[1] / c.mass - c.g)

        def rk4(x, F, dt):
            k1 = f(x, F); k2 = f(x + dt / 2 * k1, F); k3 = f(x + dt / 2 * k2, F); k4 = f(x + dt * k3, F)
            return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        zero = ca.DM([0, 0])
        for k in range(c.n1):
            opti.subject_to(X1[:, k + 1] == rk4(X1[:, k], F1[:, k], T1 / c.n1))
        for k in range(c.nf):
            opti.subject_to(Xf[:, k + 1] == rk4(Xf[:, k], zero, Tf / c.nf))
        for k in range(c.n2):
            opti.subject_to(X2[:, k + 1] == rk4(X2[:, k], F2[:, k], T2 / c.n2))
        # boundary
        opti.subject_to(X1[:, 0] == ca.DM([0.03, c.z0, 0.0, 0.0]))
        opti.subject_to(Xf[:, 0] == X1[:, c.n1]); opti.subject_to(X2[:, 0] == Xf[:, c.nf])
        opti.subject_to(X2[:, c.n2] == ca.DM([0, 0, 0, 0]) + ca.vertcat(p2 + 0.03, c.z0, 0.0, 0.0) * ca.DM([1, 0, 0, 0]) +
                        ca.DM([0, 1, 0, 0]) * c.z0)
        opti.subject_to(p2 == dx_target)
        # lift-off / touchdown: foot leaves/lands at ground  -> leg reach at those instants
        for (X, F, n, p) in ((X1, F1, c.n1, p1), (X2, F2, c.n2, p2)):
            for k in range(n):
                rx = X[0, k] - p; rz = X[1, k] - c.ankle_h
                d2 = rx * rx + rz * rz
                opti.subject_to(opti.bounded(c.d_min ** 2, d2, c.d_max ** 2))
                Fx, Fz = F[0, k], F[1, k]
                opti.subject_to(Fz >= 0); opti.subject_to(Fz <= 4.0 * mg)
                opti.subject_to(Fx <= c.mu * Fz); opti.subject_to(-Fx <= c.mu * Fz)
                # CoP inside the foot: cop = x - z*Fx/Fz  (zero angular momentum)  ->  (x-p-lo)*Fz >= z*Fx ...
                opti.subject_to((rx - c.foot_heel) * Fz >= (X[1, k]) * Fx)
                opti.subject_to((c.foot_toe - rx) * Fz >= -(X[1, k]) * Fx)
                th, tk = leg_torques(rx, rz, Fx, Fz, c.L)
                opti.subject_to(opti.bounded(-c.tau_hip, th, c.tau_hip))
                opti.subject_to(opti.bounded(-c.tau_knee, tk, c.tau_knee))
        # joint-speed limits during the two stance phases (finite differences between shooting nodes)
        if c.omega_max > 0:
            for (X, n, p, T) in ((X1, c.n1, p1, T1), (X2, c.n2, p2, T2)):
                prev = None
                for k in range(n + 1):
                    ps, ph, an = leg_angles(X[0, k] - p, X[1, k] - c.ankle_h, c.L)
                    if prev is not None:
                        dt = T / n
                        for a_now, a_prev in ((ps, prev[0]), (ph, prev[1]), (an, prev[2])):
                            opti.subject_to(opti.bounded(-c.omega_max * dt, a_now - a_prev, c.omega_max * dt))
                    prev = (ps, ph, an)
        # flight: leg must be able to reach touchdown pose; ground clearance of CoM handled by Xf z >= 0.4
        for k in range(c.nf + 1):
            opti.subject_to(Xf[1, k] >= 0.45)
        # touchdown: states at X2[:,0] satisfy reach (already via k=0 loop). take-off velocity upward
        opti.subject_to(X1[3, c.n1] >= 0.2)
        if z_apex_min is not None:
            opti.subject_to(ca.mmax(Xf[1, :]) >= z_apex_min) if False else None
            zmax = opti.variable(); opti.subject_to(zmax <= Xf[1, c.nf // 2]) if False else None
            opti.subject_to(X1[3, c.n1] ** 2 / (2 * c.g) + X1[1, c.n1] >= z_apex_min)
        # smoothness + effort
        J = 0
        for (F, T, n) in ((F1, T1, c.n1), (F2, T2, c.n2)):
            for k in range(n):
                J += (F[0, k] ** 2 + (F[1, k] - mg) ** 2) / mg ** 2 * T / n
            for k in range(n - 1):
                J += 0.3 * ((F[0, k + 1] - F[0, k]) ** 2 + (F[1, k + 1] - F[1, k]) ** 2) / mg ** 2
        opti.minimize(J)
        # warm start: ballistic-ish guess
        T1g, Tfg, T2g = 0.4, 0.25, 0.4
        opti.set_initial(T1, T1g); opti.set_initial(Tf, Tfg); opti.set_initial(T2, T2g)
        opti.set_initial(p2, dx_target)
        for k in range(c.n1 + 1):
            s = k / c.n1; opti.set_initial(X1[:, k], [0.03 + 0.05 * s, c.z0 + 0.05 * np.sin(np.pi * s), 0.1, 0.8 * s])
        for k in range(c.nf + 1):
            s = k / c.nf; opti.set_initial(Xf[:, k], [0.1 + (dx_target - 0.1) * s, c.z0 + 0.1 * np.sin(np.pi * s) + 0.05, dx_target / Tfg, 0.8 - 2 * 0.8 * s])
        for k in range(c.n2 + 1):
            opti.set_initial(X2[:, k], [dx_target + 0.03, c.z0, 0.0, 0.0])
        for k in range(c.n1):
            opti.set_initial(F1[:, k], [30.0, 1.6 * mg])
        for k in range(c.n2):
            opti.set_initial(F2[:, k], [0.0, 1.2 * mg])
        opti.solver("ipopt", {"print_time": False, "ipopt.print_level": 5 if verbose else 0, "ipopt.max_iter": 800,
                              "ipopt.tol": 1e-6, "ipopt.acceptable_tol": 1e-4})
        try:
            sol = opti.solve()
            val = sol.value
            ok = True
        except RuntimeError:
            val = opti.debug.value
            ok = False
        stats = opti.stats()
        ok = ok and stats.get("return_status", "") in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
        out = dict(ok=ok, status=stats.get("return_status"), T1=float(val(T1)), Tf=float(val(Tf)), T2=float(val(T2)),
                   X1=np.array(val(X1)), F1=np.array(val(F1)), Xf=np.array(val(Xf)), X2=np.array(val(X2)), F2=np.array(val(F2)),
                   p2=float(val(p2)), dx=dx_target)
        return out

    # -------- independent verification (pure NumPy, not using the optimiser's integrator)
    def verify(self, r):
        c = self.c
        mg = c.mass * c.g
        def rk4(x, F, dt):
            f = lambda x: np.array([x[2], x[3], F[0] / c.mass, F[1] / c.mass - c.g])
            k1 = f(x); k2 = f(x + dt / 2 * k1); k3 = f(x + dt / 2 * k2); k4 = f(x + dt * k3)
            return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x = np.array([0.03, c.z0, 0.0, 0.0]); zmax = x[1]
        sub = 8
        for k in range(c.n1):
            for _ in range(sub):
                x = rk4(x, r["F1"][:, k], r["T1"] / c.n1 / sub)
        take = x.copy()
        for k in range(c.nf):
            for _ in range(sub):
                x = rk4(x, np.zeros(2), r["Tf"] / c.nf / sub); zmax = max(zmax, x[1])
        land = x.copy()
        for k in range(c.n2):
            for _ in range(sub):
                x = rk4(x, r["F2"][:, k], r["T2"] / c.n2 / sub)
        # torque / friction stats
        def stats(X, F, p):
            peakF = np.max(np.hypot(F[0], F[1])) / mg
            fric = np.max(np.abs(F[0]) / np.maximum(F[1], 1e-6) * (F[1] > 1))
            tk = []; th = []
            for k in range(F.shape[1]):
                rx = X[0, k] - p; rz = X[1, k] - c.ankle_h; d = np.hypot(rx, rz)
                h = np.sqrt(max(c.L ** 2 - 0.25 * d * d, 1e-9)); ux, uz = -rx / d, -rz / d; nx, nz = -uz, ux
                kx = 0.5 * d * ux + h * nx; kz = 0.5 * d * uz + h * nz; ax, az = -rx, -rz
                th.append(ax * F[1, k] - az * F[0, k]); tk.append((ax - kx) * F[1, k] - (az - kz) * F[0, k])
            return peakF, fric, np.max(np.abs(th)), np.max(np.abs(tk))
        def speeds(X, p, T, n):
            ang = []
            for k in range(n + 1):
                rx = X[0, k] - p; rz = X[1, k] - c.ankle_h
                ps, ph, an = (float(v) for v in leg_angles(rx, rz, c.L))
                ang.append((ps, ph, an))
            ang = np.array(ang)
            return float(np.max(np.abs(np.diff(ang, axis=0))) / (T / n))
        sp = max(speeds(r["X1"], 0.0, r["T1"], c.n1), speeds(r["X2"], r["p2"], r["T2"], c.n2))
        s1 = stats(r["X1"], r["F1"], 0.0); s2 = stats(r["X2"], r["F2"], r["p2"])
        return dict(final_state=x, final_err_x=float(x[0] - (r["p2"] + 0.03)), final_err_z=float(x[1] - c.z0),
                    final_speed=float(np.hypot(x[2], x[3])), takeoff_vz=float(take[3]), takeoff_vx=float(take[2]),
                    apex_gain_m=float(zmax - c.z0), flight_time=r["Tf"],
                    peak_GRF_over_mg_takeoff=s1[0], peak_GRF_over_mg_landing=s2[0],
                    max_fric_ratio=max(s1[1], s2[1]),
                    peak_hip_torque_Nm_both_legs=max(s1[2], s2[2]), peak_knee_torque_Nm_both_legs=max(s1[3], s2[3]),
                    peak_stance_joint_speed_rad_s=sp)
