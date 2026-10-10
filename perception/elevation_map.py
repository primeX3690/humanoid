"""
2.5-D elevation map from depth points + traversability analysis (v2 perception was colour segmentation + 2-D SLAM only).

Fuses world-frame point clouds (from depth cameras/lidar via perception/object_detector.PinholeCamera.backproject + the robot pose)
into a per-cell Kalman height estimate with a distance-dependent sensor model and outlier gating - the same idea as ANYmal's
elevation mapping (Fankhauser et al.), minus the sensor-pose-uncertainty propagation. Outputs: height, slope, step height,
roughness, a traversability mask, and a TerrainProfile adapter so the footstep planner can plan on PERCEIVED terrain.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from terrain.terrain_profile import TerrainProfile


class ElevationMap:
    def __init__(self, size=(6.0, 6.0), res=0.04, origin=(-1.0, -3.0), sigma0=0.02, sigma_per_m=0.004, gate=3.0):
        self.res, self.origin = res, np.asarray(origin, float)
        self.n = (int(round(size[0] / res)), int(round(size[1] / res)))
        self.h = np.full(self.n, np.nan); self.var = np.full(self.n, np.inf); self.cnt = np.zeros(self.n, int)
        self.sigma0, self.sigma_per_m, self.gate = sigma0, sigma_per_m, gate

    # ---------------------------------------------------------------- fusion
    def _ij(self, xy):
        ij = np.floor((xy - self.origin) / self.res).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < self.n[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < self.n[1])
        return ij, ok

    def integrate(self, pts, sensor_pos=(0, 0, 0)):
        pts = np.asarray(pts, float); d = np.linalg.norm(pts - np.asarray(sensor_pos, float), axis=1)
        ij, ok = self._ij(pts[:, :2]); pts, ij, d = pts[ok], ij[ok], d[ok]
        # one pass per point *group* per cell: take the median height of the batch hitting a cell (kills outlier spikes), then fuse
        lin = ij[:, 0] * self.n[1] + ij[:, 1]
        order = np.argsort(lin); lin, z, dd = lin[order], pts[order, 2], d[order]
        edges = np.flatnonzero(np.diff(lin)) + 1
        for zs, ds, l in zip(np.split(z, edges), np.split(dd, edges), np.split(lin, edges)):
            i, j = divmod(int(l[0]), self.n[1])
            zm = float(np.median(zs)); sm2 = (self.sigma0 + self.sigma_per_m * float(ds.mean())) ** 2 / len(zs)
            if self.cnt[i, j] == 0:
                self.h[i, j], self.var[i, j] = zm, sm2
            else:
                innov = zm - self.h[i, j]; S = self.var[i, j] + sm2
                if abs(innov) > self.gate * np.sqrt(S):
                    if innov > 0 or self.cnt[i, j] < 3:       # something new is HIGHER (an obstacle / step) -> trust it; lower -> noise
                        self.h[i, j], self.var[i, j] = zm, sm2
                    continue
                K = self.var[i, j] / S
                self.h[i, j] += K * innov; self.var[i, j] *= (1 - K)
            self.cnt[i, j] += len(zs)

    def decay(self, dt, sigma_drift=0.005):
        m = np.isfinite(self.var); self.var[m] += (sigma_drift ** 2) * dt

    # ---------------------------------------------------------------- products
    def filled(self, default=0.0):
        """Height map with holes inpainted from the nearest observed cell (then lightly smoothed in the holes only)."""
        valid = np.isfinite(self.h)
        if not valid.any(): return np.full(self.n, default)
        idx = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        return self.h[tuple(idx)]

    def height_at(self, x, y, default=0.0):
        H = self.filled(default)
        gx = (x - self.origin[0]) / self.res - 0.5; gy = (y - self.origin[1]) / self.res - 0.5
        return float(ndi.map_coordinates(H, [[gx], [gy]], order=1, mode="nearest")[0])

    def slope_deg(self):
        gx, gy = np.gradient(ndi.gaussian_filter(self.filled(), 1.0), self.res)
        return np.degrees(np.arctan(np.hypot(gx, gy)))

    def step_height(self, radius_cells=1):
        H = self.filled(); k = 2 * radius_cells + 1
        return ndi.maximum_filter(H, size=k) - ndi.minimum_filter(H, size=k)

    def roughness(self):
        H = self.filled(); return np.sqrt(np.maximum(ndi.uniform_filter(H ** 2, 3) - ndi.uniform_filter(H, 3) ** 2, 0.0))

    def traversable(self, max_slope_deg=20.0, max_step_m=0.10, max_rough_m=0.03):
        ok = (self.slope_deg() <= max_slope_deg) & (self.step_height() <= max_step_m) & (self.roughness() <= max_rough_m)
        return ok & np.isfinite(self.h)

    def step_edges(self, min_step_m=0.03):
        """Cells on a height discontinuity (stairs / curbs): where the local step exceeds min_step_m."""
        return self.step_height() >= min_step_m

    def cell_xy(self, ij):
        return self.origin + (np.asarray(ij, float) + 0.5) * self.res


class ElevationMapTerrain(TerrainProfile):
    """Adapter: plan footsteps (planning.footstep_planner.apply_terrain) on the terrain the robot actually PERCEIVED."""
    def __init__(self, emap: ElevationMap): self.emap = emap
    def height_at(self, x, y): return self.emap.height_at(x, y)
