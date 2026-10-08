"""Room + landmarks + 2-D lidar; humanoid walks a loop with drifting leg-odometry. python -m simulation.slam_sim"""
import json, numpy as np
from perception.ekf_slam import EKFSLAM, OccupancyGrid, wrap


def make_world(seed=0):
    rng = np.random.default_rng(seed)
    walls = [((-4, -4), (4, -4)), ((4, -4), (4, 4)), ((4, 4), (-4, 4)), ((-4, 4), (-4, -4)),
             ((-1.5, -1.0), (1.5, -1.0)), ((1.5, 1.2), (1.5, 2.6))]       # room + two interior obstacles
    lms = np.array([[-3.0, -3.0], [3.0, -3.2], [3.2, 3.0], [-3.1, 3.1], [0.0, 2.8], [-2.2, 0.2], [2.4, -0.2], [0.2, -2.5], [-0.8, 1.2], [2.0, 3.4]])
    return walls, lms


def ray_wall(p, d, a, b):
    a, b = np.array(a, float), np.array(b, float)
    e = b - a; den = d[0] * e[1] - d[1] * e[0]
    if abs(den) < 1e-12:
        return np.inf
    w = a - p
    t = (w[0] * e[1] - w[1] * e[0]) / den; s = (w[0] * d[1] - w[1] * d[0]) / den
    return t if (t > 0 and 0 <= s <= 1) else np.inf


def lidar(pose, walls, angles, max_range, rng, sig=0.01):
    p = np.array(pose[:2]); out = []
    for a in angles:
        d = np.array([np.cos(pose[2] + a), np.sin(pose[2] + a)])
        r = min(min(ray_wall(p, d, *w) for w in walls), max_range)
        out.append(min(r + (rng.normal(0, sig) if r < max_range else 0), max_range))
    return np.array(out)


def run(seed=0, loops=2, steps_per_loop=160, verbose=True):
    rng = np.random.default_rng(seed)
    walls, lms = make_world(seed)
    R = 2.4
    traj = []
    for k in range(loops * steps_per_loop + 1):
        s = 2 * np.pi * k / steps_per_loop
        traj.append([R * np.cos(s) * 1.0, R * np.sin(s) * 0.8 - 0.0, 0.0])
    traj = np.array(traj)
    for k in range(len(traj)):
        nxt = traj[min(k + 1, len(traj) - 1)]
        traj[k, 2] = np.arctan2(nxt[1] - traj[k, 1], nxt[0] - traj[k, 0]) if k < len(traj) - 1 else traj[k - 1, 2]
    slam = EKFSLAM(pose0=traj[0].copy())
    odo = traj[0].copy(); odo_hist = [odo.copy()]; est_hist = [slam.mu[:3].copy()]
    grid_est, grid_odo = OccupancyGrid(), OccupancyGrid()
    angles = np.radians(np.arange(-90, 91, 3)); maxr = 6.0
    a = slam.a
    for k in range(1, len(traj)):
        p0, p1 = traj[k - 1], traj[k]
        dx, dy = p1[:2] - p0[:2]
        r1 = wrap(np.arctan2(dy, dx) - p0[2]); tr = np.hypot(dx, dy); r2 = wrap(p1[2] - p0[2] - r1)
        # drifting leg odometry: scale error (systematic 3 %) + heading bias + noise
        n1 = r1 + rng.normal(0, np.sqrt(a[0] * r1 ** 2 + a[1] * tr ** 2)) + 0.004
        nt = tr * 1.03 + rng.normal(0, np.sqrt(a[2] * tr ** 2 + a[3] * (r1 ** 2 + r2 ** 2)))
        n2 = r2 + rng.normal(0, np.sqrt(a[0] * r2 ** 2 + a[1] * tr ** 2)) + 0.004
        odo[0] += nt * np.cos(odo[2] + n1); odo[1] += nt * np.sin(odo[2] + n1); odo[2] = wrap(odo[2] + n1 + n2)
        slam.predict(n1, nt, n2)
        # landmark observations from perception (range <= 4.5 m, FOV +-70 deg)
        obs = []
        for l in lms:
            dx, dy = l - p1[:2]; r = np.hypot(dx, dy); b = wrap(np.arctan2(dy, dx) - p1[2])
            if r < 4.5 and abs(b) < np.radians(70):
                obs.append((r + rng.normal(0, 0.05), wrap(b + rng.normal(0, np.radians(1.5)))))
        rng.shuffle(obs)
        slam.update(obs)
        odo_hist.append(odo.copy()); est_hist.append(slam.mu[:3].copy())
        if k % 4 == 0:
            ranges = lidar(p1, walls, angles, maxr, rng)
            grid_est.integrate_scan(slam.mu[:3], angles, ranges, maxr)
            grid_odo.integrate_scan(odo, angles, ranges, maxr)
    odo_hist, est_hist = np.array(odo_hist), np.array(est_hist)
    ate = lambda h: float(np.sqrt(np.mean(np.sum((h[:, :2] - traj[:, :2]) ** 2, axis=1))))
    # ground-truth occupancy
    gt = OccupancyGrid()
    for w in walls:
        n = int(np.hypot(*(np.array(w[1]) - np.array(w[0]))) / 0.02)
        for t in np.linspace(0, 1, n):
            x, y = np.array(w[0]) * (1 - t) + np.array(w[1]) * t
            i, j = gt._ij(x, y)
            gt.L[i, j] = 5
    def f1(g):
        occ = g.occupied(); tru = gt.L > 0
        from scipy.ndimage import binary_dilation
        tol = binary_dilation(tru, iterations=2)        # 10 cm tolerance
        tp = (occ & tol).sum(); prec = tp / max(occ.sum(), 1)
        rec = (binary_dilation(occ, iterations=2) & tru).sum() / max(tru.sum(), 1)
        return float(2 * prec * rec / max(prec + rec, 1e-9)), float(prec), float(rec)
    lm_est = slam.mu[3:].reshape(-1, 2)
    # match estimated to true landmarks (nearest, after convention that frame == truth start)
    lm_err = [float(np.min(np.linalg.norm(lms - e, axis=1))) for e in lm_est]
    res = dict(ate_odometry_m=ate(odo_hist), ate_slam_m=ate(est_hist),
               final_err_odometry_m=float(np.linalg.norm(odo_hist[-1, :2] - traj[-1, :2])),
               final_err_slam_m=float(np.linalg.norm(est_hist[-1, :2] - traj[-1, :2])),
               n_landmarks_true=len(lms), n_landmarks_est=int(slam.n_lm), landmark_err_mean_m=float(np.mean(lm_err)),
               map_f1_odometry=f1(grid_odo), map_f1_slam=f1(grid_est))
    if verbose:
        print(json.dumps(res, indent=1))
    return res, dict(traj=traj, odo=odo_hist, est=est_hist, lm_true=lms, lm_est=lm_est, grid_est=grid_est, grid_odo=grid_odo, walls=walls)


if __name__ == "__main__":
    allr = []
    for s in range(5):
        r, _ = run(seed=s, verbose=False); allr.append(r)
        print(s, "ATE odo %.3f -> SLAM %.3f | lm %d/%d err %.3f | map F1 odo %.2f -> slam %.2f" % (
            r["ate_odometry_m"], r["ate_slam_m"], r["n_landmarks_est"], r["n_landmarks_true"], r["landmark_err_mean_m"], r["map_f1_odometry"][0], r["map_f1_slam"][0]))
    json.dump(allr, open("results/slam_5seeds.json", "w"), indent=1)
