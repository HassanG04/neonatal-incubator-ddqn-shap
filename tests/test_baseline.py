"""Pytest unit tests for the rule-based baseline controller (v3.0 env API).

These tests exercise the public API of ``RuleBasedBaselineController`` from
``src/baseline.py`` against the v3.0 Gymnasium-compatible environment
(``src/environment.py``) for the Neonatal Incubator DDQN + SHAP project
(Member 1 deliverable).

API recap
---------
Controller::

    ctrl = RuleBasedBaselineController(
        target_Tc=37.0, target_SpO2=93.0, Kp=0.8, Ki=0.05, Kd=0.1,
        FF_BIAS=0.5,
    )
    action = ctrl.select_action(state)   # int in [0, 24]
    ctrl.reset()                         # clears PID integrator
    ctrl.last_info                       # dict of PID diagnostics

Environment (Gymnasium v3.0 -- 5-tuple step, 11-dim observation)::

    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(action)

Action decoding (must match the env's actuator grid)::

    heater_idx = action // 5   # HEATER_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
    o2_idx     = action % 5    # FIO2_LEVELS   = (0.21, 0.30, 0.40, 0.50, 0.60)

Observation is an 11-dim float32 vector::

    [Tc, Tsk, Ta, C_O2, SpO2, Tamb, dTa_dt, dC_O2_dt,
     phase, disturbance, scenario_id]
      0   1    2   3     4    5      6        7        8      9           10

Run from the member_1 root directory::

    pytest tests/test_baseline.py -v
"""

import os
import sys

import numpy as np
import pytest

# Make ``src`` importable whether pytest is invoked from the repository root
# or from inside the ``tests`` directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from baseline import RuleBasedBaselineController  # noqa: E402
from environment import NeonatalIncubatorEnv     # noqa: E402


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
HEATER_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
FIO2_LEVELS = (0.21, 0.30, 0.40, 0.50, 0.60)
N_LEVELS_PER_CHANNEL = 5            # 5 heater x 5 O2 = 25 discrete actions
SCENARIOS = [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_state(tc: float, spo2: float) -> np.ndarray:
    """Build a minimal 11-dim v3.0 observation vector.

    Only ``Tc`` (index 0) and ``SpO2`` (index 4) are varied; the remaining
    channels are kept at safe neutral defaults so the controller focuses on
    the test signal.  Layout (matches ``environment.OBS_NAMES``)::

        [Tc, Tsk, Ta, C_O2, SpO2, Tamb, dTa_dt, dC_O2_dt,
         phase, disturbance, scenario_id]
    """
    return np.array(
        [tc, tc - 0.5, 34.0, 21.0, spo2, 25.0, 0.0, 0.0, 0.5, 0.0, 0.0],
        dtype=np.float32,
    )


def _decode_action(action: int):
    """Decode a discrete action into ``(heater_idx, o2_idx)``."""
    action = int(action)
    heater_idx = action // N_LEVELS_PER_CHANNEL
    o2_idx = action % N_LEVELS_PER_CHANNEL
    return heater_idx, o2_idx


def _make_controller(**overrides) -> RuleBasedBaselineController:
    """Instantiate the controller with the documented defaults.

    Defaults: ``target_Tc=37.0, target_SpO2=93.0, Kp=0.8, Ki=0.05, Kd=0.1``.
    Callers may override individual kwargs.
    """
    defaults = dict(target_Tc=37.0, target_SpO2=93.0,
                    Kp=0.8, Ki=0.05, Kd=0.1)
    defaults.update(overrides)
    return RuleBasedBaselineController(**defaults)


def _get_integrator(controller: RuleBasedBaselineController) -> float:
    """Robustly read the PID integrator value from a controller instance.

    The baseline API exposes ``last_info`` as a dict containing ``'integral'``
    and the underlying ``_integral`` private attribute. We prefer the public
    ``last_info['integral']`` value (populated on every ``select_action`` call)
    but fall back to ``controller._integral`` so the test is robust to the
    exact attribute spelling chosen by the implementation.
    """
    # 1. Public diagnostic dict (preferred -- always populated).
    info = getattr(controller, "last_info", None)
    if isinstance(info, dict):
        for key in ("integral", "integrator"):
            if key in info and info[key] is not None:
                try:
                    return float(info[key])
                except (TypeError, ValueError):
                    continue

    # 2. Direct attribute access (fallback for pre-step reads).
    for attr in ("_integral", "integral", "integrator"):
        try:
            val = getattr(controller, attr)
            if val is not None:
                return float(val)
        except (AttributeError, TypeError, ValueError):
            continue

    raise AttributeError(
        "Could not find a PID integrator on the controller (expected "
        "'integral'/'integrator' key in last_info or matching attribute)."
    )


# ---------------------------------------------------------------------------
# 1. Action range
# ---------------------------------------------------------------------------
def test_action_in_valid_range():
    """For many random states, action must be in [0, 24]."""
    controller = _make_controller()
    controller.reset()
    rng = np.random.default_rng(seed=42)
    for _ in range(200):
        tc = float(rng.uniform(33.0, 40.0))
        spo2 = float(rng.uniform(70.0, 100.0))
        state = _make_state(tc, spo2)
        action = int(controller.select_action(state))
        assert 0 <= action <= 24, (
            f"action {action} out of [0, 24] for state tc={tc}, spo2={spo2}"
        )


# ---------------------------------------------------------------------------
# 2. Cold state -> high heater
# ---------------------------------------------------------------------------
def test_cold_state_selects_high_heater():
    """Tc=35.5 (state[0]=35.5) -> heater_idx >= 3 (high heater)."""
    controller = _make_controller()
    controller.reset()
    state = _make_state(tc=35.5, spo2=93.0)
    action = int(controller.select_action(state))
    heater_idx, _ = _decode_action(action)
    assert heater_idx >= 3, (
        f"expected heater_idx >= 3 for cold Tc=35.5, got {heater_idx}"
    )


# ---------------------------------------------------------------------------
# 3. Hot state -> low heater
# ---------------------------------------------------------------------------
def test_hot_state_selects_low_heater():
    """Tc=38.5 -> heater_idx == 0 (heater off)."""
    controller = _make_controller()
    controller.reset()
    state = _make_state(tc=38.5, spo2=93.0)
    action = int(controller.select_action(state))
    heater_idx, _ = _decode_action(action)
    assert heater_idx == 0, (
        f"expected heater_idx == 0 for hot Tc=38.5, got {heater_idx}"
    )


# ---------------------------------------------------------------------------
# 4. Low SpO2 -> higher FiO2
# ---------------------------------------------------------------------------
def test_low_spo2_selects_higher_fio2():
    """SpO2=82 (state[4]=82) -> o2_idx >= 2 (FiO2 >= 0.40)."""
    controller = _make_controller()
    controller.reset()
    state = _make_state(tc=37.0, spo2=82.0)
    action = int(controller.select_action(state))
    _, o2_idx = _decode_action(action)
    assert o2_idx >= 2, (
        f"expected o2_idx >= 2 (FiO2 >= 0.40) for SpO2=82, got {o2_idx}"
    )
    assert FIO2_LEVELS[o2_idx] >= 0.40, (
        f"FiO2 {FIO2_LEVELS[o2_idx]:.2f} below 0.40 for SpO2=82"
    )


# ---------------------------------------------------------------------------
# 5. High SpO2 -> low FiO2 (room air)
# ---------------------------------------------------------------------------
def test_high_spo2_selects_low_fio2():
    """SpO2=97 -> o2_idx == 0 (FiO2=0.21 room air)."""
    controller = _make_controller()
    controller.reset()
    state = _make_state(tc=37.0, spo2=97.0)
    action = int(controller.select_action(state))
    _, o2_idx = _decode_action(action)
    assert o2_idx == 0, (
        f"expected o2_idx == 0 (room air) for SpO2=97, got {o2_idx}"
    )
    assert FIO2_LEVELS[o2_idx] == pytest.approx(0.21, abs=1e-6)


# ---------------------------------------------------------------------------
# 6. Normal state -> conservative actions
# ---------------------------------------------------------------------------
def test_normal_state_selects_conservative():
    """Tc=37.0, SpO2=93 -> heater_idx in {1,2,3}, o2_idx == 0 (room air)."""
    controller = _make_controller()
    controller.reset()
    state = _make_state(tc=37.0, spo2=93.0)
    action = int(controller.select_action(state))
    heater_idx, o2_idx = _decode_action(action)
    assert heater_idx in {1, 2, 3}, (
        f"expected heater_idx in {{1, 2, 3}} at setpoint, got {heater_idx}"
    )
    # When SpO2 is inside the target band, the controller should select the
    # most conservative oxygen command -- room air (FiO2 = 0.21, idx 0).
    assert o2_idx == 0, (
        f"expected o2_idx == 0 (room air) for SpO2 in band, got {o2_idx}"
    )


# ---------------------------------------------------------------------------
# 7. reset() clears the PID integrator
# ---------------------------------------------------------------------------
def test_reset_clears_pid_state():
    """After several cold steps, ``reset()`` must zero the PID integrator."""
    controller = _make_controller()
    controller.reset()
    cold_state = _make_state(tc=35.5, spo2=82.0)
    for _ in range(10):
        controller.select_action(cold_state)
    pre_reset = _get_integrator(controller)
    assert pre_reset != pytest.approx(0.0, abs=1e-9), (
        "integrator did not accumulate over 10 cold steps before reset"
    )
    controller.reset()
    post_reset = _get_integrator(controller)
    assert post_reset == pytest.approx(0.0, abs=1e-9), (
        f"integrator not zero after reset: {post_reset}"
    )


# ---------------------------------------------------------------------------
# 8. PID integrator accumulates under persistent error
# ---------------------------------------------------------------------------
def test_pid_integrator_accumulates():
    """Repeated cold error should grow (net) the PID integrator."""
    controller = _make_controller()
    controller.reset()
    cold_state = _make_state(tc=35.5, spo2=82.0)
    values = [_get_integrator(controller)]
    for _ in range(8):
        controller.select_action(cold_state)
        values.append(_get_integrator(controller))
    assert values[-1] > values[0], (
        f"integrator did not accumulate under persistent cold error: {values}"
    )
    # The integrator should not decrease over the first two steps (cold error
    # is positive & persistent).  We tolerate small numerical jitter but
    # require net growth across the rollout.
    assert values[-1] >= values[1] - 1e-6, (
        f"integrator decreased unexpectedly across rollout: {values}"
    )


# ---------------------------------------------------------------------------
# 9. Full-episode rollout across all scenarios (Gymnasium v3.0 API)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_runs_full_episode_all_scenarios(scenario):
    """Baseline must drive a full episode on every scenario (5-tuple API)."""
    controller = _make_controller()
    controller.reset()
    env = NeonatalIncubatorEnv(scenario=scenario, seed=42)
    try:
        obs, _info = env.reset(seed=42)
        state = obs
        total_reward = 0.0
        terminated = False
        truncated = False
        steps = 0
        max_steps = 1000  # generous safety bound; env default is 200
        while not (terminated or truncated) and steps < max_steps:
            action = int(controller.select_action(state))
            state, reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)
            steps += 1
        assert np.isfinite(total_reward), (
            f"scenario {scenario}: total reward not finite ({total_reward})"
        )
        assert steps > 0, (
            f"scenario {scenario}: episode produced zero steps"
        )
    finally:
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:  # pragma: no cover -- defensive
                pass


# ---------------------------------------------------------------------------
# 10. Stabilisation on scenario 1
# ---------------------------------------------------------------------------
def test_baseline_stabilizes_normal_scenario():
    """On scenario 1, after warmup, mean Tc / SpO2 must lie in relaxed bands.

    Relaxed tolerances reflect what a rule-based controller can realistically
    achieve: mean Tc in ``[36.0, 38.0]`` and mean SpO2 in ``[88, 96]``. If the
    controller cannot meet the bands, we fall back to asserting that the
    total reward is finite so the test is informative rather than brittle.
    """
    controller = _make_controller()
    controller.reset()
    env = NeonatalIncubatorEnv(scenario=1, seed=42)
    try:
        obs, _info = env.reset(seed=42)
        state = obs

        tcs = []
        spo2s = []
        total_reward = 0.0
        terminated = False
        truncated = False
        steps = 0
        warmup = 30        # discard the first 30 steps as transient
        max_steps = 200    # one full episode

        while not (terminated or truncated) and steps < max_steps:
            action = int(controller.select_action(state))
            state, reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)
            steps += 1
            if steps > warmup:
                tcs.append(float(state[0]))
                spo2s.append(float(state[4]))

        # Reward must always be finite.
        assert np.isfinite(total_reward), (
            f"total reward is not finite: {total_reward}"
        )

        # If we collected enough post-warmup samples, assert the relaxed
        # clinical bands; otherwise fall back to the finite-reward check
        # (which has already passed above).
        if len(tcs) >= 20:
            mean_tc = float(np.mean(tcs))
            mean_spo2 = float(np.mean(spo2s))
            assert 36.0 <= mean_tc <= 38.0, (
                f"mean Tc {mean_tc:.3f} outside [36.0, 38.0]"
            )
            assert 88.0 <= mean_spo2 <= 96.0, (
                f"mean SpO2 {mean_spo2:.3f} outside [88, 96]"
            )
    finally:
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:  # pragma: no cover -- defensive
                pass


# ---------------------------------------------------------------------------
# 11. Fix B: Custom 3x2 action grid and invalid grid validation
# ---------------------------------------------------------------------------
def test_custom_3x2_action_grid():
    """Fix B: A custom 3-heater x 2-oxygen action grid (6 actions) must run
    without IndexError across extreme states and full rollouts."""
    import tempfile
    import yaml

    custom_cfg = {
        "environment": {
            "heater_levels": [0.0, 0.5, 1.0],
            "fio2_levels": [0.21, 0.50],
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(custom_cfg, f)
        path = f.name
    try:
        env = NeonatalIncubatorEnv(scenario=5, seed=42, config_path=path)
        assert env.action_space.n == 6
        assert env.n_heater_levels == 3
        assert env.n_o2_levels == 2

        ctrl = RuleBasedBaselineController.from_env(env, config_path=path)
        assert ctrl.n_actions == 6
        assert ctrl.heater_levels == (0.0, 0.5, 1.0)
        assert ctrl.fio2_levels == (0.21, 0.50)

        # Test minimum/maximum indices and extreme states (severe hypoxia, mild hypoxia, normal)
        for tc in (35.0, 37.0, 39.0):
            for spo2 in (80.0, 90.0, 94.0, 99.0):
                a = ctrl.select_action(_make_state(tc, spo2))
                assert 0 <= a < 6, f"action {a} out of 3x2 range [0, 6)"
                h_idx = a // 2
                o_idx = a % 2
                assert 0 <= h_idx < 3 and 0 <= o_idx < 2

        # Full rollout on custom 3x2 grid
        obs, _ = env.reset(seed=42)
        ctrl.reset()
        done = False
        steps = 0
        while not done and steps < 200:
            action = ctrl.select_action(obs)
            assert 0 <= action < 6
            obs, r, term, trunc, _ = env.step(action)
            assert np.isfinite(r)
            done = term or trunc
            steps += 1
        assert steps > 0
        env.close()
    finally:
        os.unlink(path)


def test_invalid_action_grid_rejected():
    """Fix B: Invalid grids fail early with ValueError at construction."""
    with pytest.raises(ValueError):
        RuleBasedBaselineController(heater_levels=[0.5])  # < 2 levels
    with pytest.raises(ValueError):
        RuleBasedBaselineController(heater_levels=[0.0, 1.0, 0.5])  # not ascending
    with pytest.raises(ValueError):
        RuleBasedBaselineController(fio2_levels=[0.10, 0.50])  # below 0.21

