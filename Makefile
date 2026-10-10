PY ?= python
export PYTHONPATH := .
.PHONY: env test test-fast test-sim test-nopytest report
env:        ; $(PY) tools/check_env.py
test:       ; $(PY) -m pytest
test-fast:  ; $(PY) -m pytest -m "not slow and not mujoco and not casadi and not torch and not sb3"
test-sim:   ; $(PY) -m pytest -m "mujoco or osqp or casadi"
test-nopytest: ; $(PY) tools/minipytest.py tests
report:     ; $(PY) tools/hardware_report.py
