"""tests/test_preview_controller.py — verifies the preview controller's
core theoretical guarantees, not just that it produces some output."""
import numpy as np
import pytest

from dynamics.lipm import LIPM, LIPMParams
from control.zmp_preview_controller import ZMPPreviewController, PreviewControllerConfig


def _make_controller(preview_horizon_s=1.6):
    lipm = LIPM(LIPMParams(com_height_m=0.8, dt=0.01))
    ctrl = ZMPPreviewController(lipm, PreviewControllerConfig(preview_horizon_s=preview_horizon_s))
    return lipm, ctrl


def test_closed_loop_is_stable():
    """The augmented closed-loop system's eigenvalues must all lie strictly
    inside the unit circle - the formal definition of discrete-time
    stability. If this fails, the controller would diverge no matter what
    reference it's given, and nothing downstream would matter."""
    lipm, ctrl = _make_controller()
    A_hat = np.zeros((4, 4))
    A_hat[0, 0] = 1.0
    A_hat[0, 1:4] = (lipm.C @ lipm.A).flatten()
    A_hat[1:4, 1:4] = lipm.A
    B_hat = np.zeros((4, 1))
    B_hat[0, 0] = (lipm.C @ lipm.B).item()
    B_hat[1:4, 0] = lipm.B.flatten()
    K = np.concatenate([[ctrl.Gi], ctrl.Gx])
    eigvals = np.linalg.eigvals(A_hat - B_hat @ K.reshape(1, 4))
    assert np.all(np.abs(eigvals) < 1.0 - 1e-9), f"unstable closed loop, eigenvalues={eigvals}"


def test_tracks_constant_reference_with_zero_steady_state_error():
    """With integral action (the whole point of the augmented-state
    formulation), tracking a constant ZMP reference must converge to
    exactly that reference, not just get close."""
    lipm, ctrl = _make_controller()
    ref_value = 0.15
    zmp_ref = np.full(2000, ref_value)
    x = np.zeros(3)
    N = ctrl.N
    for k in range(len(zmp_ref) - N - 1):
        jerk = ctrl.compute_jerk(x, zmp_ref[k:k + N + 1])
        x = lipm.step(x, jerk)
    final_zmp = lipm.zmp(x)
    assert abs(final_zmp - ref_value) < 1e-4, (
        f"expected convergence to {ref_value}, got {final_zmp} "
        f"(steady-state tracking error should be ~zero with integral action)"
    )


def test_preview_reduces_tracking_lag_vs_zero_preview():
    """The entire point of PREVIEW (vs. plain feedback) control is that the
    CoM starts reacting to a future ZMP step change before it happens. We
    check this directly: with a longer preview horizon, RMS tracking error
    around a step change in the reference should be lower than with almost
    no preview - the anticipation effect must actually be measurable, not
    just asserted in a docstring."""
    lipm_long, ctrl_long = _make_controller(preview_horizon_s=1.6)
    lipm_short, ctrl_short = _make_controller(preview_horizon_s=0.05)

    n = 400
    step_at = 200
    zmp_ref = np.concatenate([np.zeros(step_at), np.full(n - step_at, 0.2)])

    def run(lipm, ctrl, zmp_ref):
        N = ctrl.N
        pad = np.concatenate([zmp_ref, np.full(N + 1, zmp_ref[-1])])
        x = np.zeros(3)
        zmp_out = np.zeros(len(zmp_ref))
        for k in range(len(zmp_ref)):
            zmp_out[k] = lipm.zmp(x)
            jerk = ctrl.compute_jerk(x, pad[k:k + N + 1])
            x = lipm.step(x, jerk)
        return zmp_out

    zmp_long = run(lipm_long, ctrl_long, zmp_ref)
    zmp_short = run(lipm_short, ctrl_short, zmp_ref)

    window = slice(step_at - 30, step_at + 30)
    err_long = np.sqrt(np.mean((zmp_long[window] - zmp_ref[window]) ** 2))
    err_short = np.sqrt(np.mean((zmp_short[window] - zmp_ref[window]) ** 2))
    assert err_long < err_short, (
        f"expected long-preview tracking error ({err_long:.4f}) to beat "
        f"short-preview ({err_short:.4f}) around a reference step change"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
