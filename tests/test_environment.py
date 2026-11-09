"""Pytest unit tests for the NeonatalIncubatorEnv (Gymnasium v3.0 API).

These tests validate the API contract, dynamics and review-fixes of the
``NeonatalIncubatorEnv`` class defined in ``src/environment.py`` (v3.0 -- 11-dim
observation, new physics, all V2 review issues resolved).

Run from the member_1 root directory with::

    pytest tests/test_environment.py -v

The environment follows the standard Gymnasium API::

    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(action)

where ``action`` is an integer in ``[0, 25)`` decoded as
``heater_idx * 5 + o2_idx`` (5 heater levels x 5 FiO2 levels).

Observation (11-dim float32, order frozen by ``environment.OBS_NAMES``)::

    [Tc, Tsk, Ta, C_O2, SpO2, Tamb, dTa_dt, dC_O2_dt,
     phase, disturbance, scenario_id]
      0   1    2   3     4    5      6        7        8      9           10
"""

import os
import sys
import tempfile

import numpy as np
import pytest
import yaml

# Make ``src`` importable regardless of the cwd pytest was invoked from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from environment import (  # noqa: E402
    NeonatalIncubatorEnv,
    OBS_NAMES,
    ACTION_TABLE,
    N_SCENARIOS,
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
MAX_STEPS = 200
SCENARIOS = [1, 2, 3, 4, 5]
EXPECTED_OBS_DIM = 11
EXPECTED_N_ACTIONS = 25
# action 6 == heater_idx 1 (u=0.25, Q=90W), o2_idx 1 (FiO2=0.30).
SAFE_ACTION = 6


# ---------------------------------------------------------------------------
# Reusable fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def env():
    """A fresh default (scenario=1) environment."""
    e = NeonatalIncubatorEnv(scenario=1, seed=42)
    yield e
    e.close()


# ---------------------------------------------------------------------------
# 1. reset() returns a (obs, info) 2-tuple
# ---------------------------------------------------------------------------
def test_reset_returns_tuple_obs_info():
    e = NeonatalIncubatorEnv(scenario=1, seed=42)
    try:
        result = e.reset(seed=42)
        assert isinstance(result, tuple), "reset() must return a tuple"
        assert len(result) == 2, (
            f"reset() must return 2-tuple (obs, info), got len={len(result)}"
        )
        obs, info = result
        assert isinstance(obs, np.ndarray), (
            f"obs must be np.ndarray, got {type(obs)}"
        )
        assert isinstance(info, dict), f"info must be dict, got {type(info)}"
    finally:
        e.close()


# ---------------------------------------------------------------------------
# 2. Observation shape and dtype (11-dim float32 -- V3 change from 10)
# ---------------------------------------------------------------------------
def test_observation_shape_and_dtype():
    e = NeonatalIncubatorEnv(scenario=1, seed=42)
    try:
        obs, _ = e.reset(seed=42)
        assert obs.shape == (EXPECTED_OBS_DIM,), (
            f"expected shape ({EXPECTED_OBS_DIM},), got {obs.shape}"
        )
        assert obs.dtype == np.float32, (
            f"expected dtype float32, got {obs.dtype}"
        )
    finally:
        e.close()


# ---------------------------------------------------------------------------
# 3. Observation within observation_space bounds (CRITICAL V2-03 regression)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", [1, 2, 3, 5])
@pytest.mark.parametrize("seed", [0, 42])
def test_observation_within_bounds(scenario, seed):
    """All observations across scenarios / seeds / actions must stay within
    the declared ``observation_space`` bounds (V2-03 regression)."""
    e = NeonatalIncubatorEnv(scenario=scenario, seed=seed)
    try:
        obs, _ = e.reset(seed=seed)
        assert e.observation_space.contains(obs), (
            f"reset obs {obs} not contained in observation_space"
        )
        rng = np.random.default_rng(seed + 1)
        for _ in range(50):
            action = int(rng.integers(0, e.action_space.n))
            obs, reward, terminated, truncated, info = e.step(action)
            assert e.observation_space.contains(obs), (
                f"scenario {scenario} seed {seed} step {info['step']}: "
                f"obs {obs} not in observation_space"
            )
            assert np.isfinite(reward), (
                f"non-finite reward {reward} at step {info['step']}"
            )
            if terminated or truncated:
                break
    finally:
        e.close()


# ---------------------------------------------------------------------------
# 4. Action space size == Discrete(25)
# ---------------------------------------------------------------------------
def test_action_space_size():
    e = NeonatalIncubatorEnv(scenario=1, seed=42)
    try:
        assert e.action_space.n == EXPECTED_N_ACTIONS, (
            f"expected Discrete({EXPECTED_N_ACTIONS}), got n={e.action_space.n}"
        )
    finally:
        e.close()


# ---------------------------------------------------------------------------
# 5. step() returns a 5-tuple with correct types
# ---------------------------------------------------------------------------
def test_step_returns_5_tuple():
    e = NeonatalIncubatorEnv(scenario=1, seed=42)
    try:
        e.reset(seed=42)
        result = e.step(SAFE_ACTION)
        assert isinstance(result, tuple), "step() must return a tuple"
        assert len(result) == 5, (
            f"step() must return 5-tuple, got len={len(result)}"
        )
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert isinstance(reward, (float, np.floating)), (
            f"reward must be a float, got {type(reward)}"
        )
        assert np.isfinite(reward), f"reward must be finite, got {reward}"
        assert isinstance(terminated, (bool, np.bool_)), (
            f"terminated must be bool, got {type(terminated)}"
        )
        assert isinstance(truncated, (bool, np.bool_)), (
            f"truncated must be bool, got {type(truncated)}"
        )
        assert isinstance(info, dict), f"info must be dict, got {type(info)}"
    finally:
        e.close()


# ---------------------------------------------------------------------------
# 6. Invalid actions raise ValueError
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_action", [-1, 25, 100, -5])
def test_invalid_action_raises(bad_action):
    e = NeonatalIncubatorEnv(scenario=1, seed=42)
    try:
        e.reset(seed=42)
        with pytest.raises(ValueError):
            e.step(bad_action)
    finally:
        e.close()


# ---------------------------------------------------------------------------
# 7. Episode truncates at max_steps on scenario 1 with action 6
# ---------------------------------------------------------------------------
def test_episode_truncates_at_max_steps():
    e = NeonatalIncubatorEnv(scenario=1, seed=42)
    try:
        e.reset(seed=42)
        terminated = False
        truncated = False
        steps = 0
        info = {}
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = e.step(SAFE_ACTION)
            steps += 1
            if steps > MAX_STEPS + 5:  # safety valve to avoid infinite loop
                break
        assert truncated, (
            f"Episode should have truncated at step {MAX_STEPS}, "
            f"ran {steps} steps"
        )
        assert not terminated, (
            f"Scenario 1 with action {SAFE_ACTION} should not terminate "
            f"(safety breach), but terminated=True at step {info.get('step')}"
        )
        assert info.get("step") == MAX_STEPS, (
            f"Expected final info['step']=={MAX_STEPS}, got {info.get('step')}"
        )
    finally:
        e.close()


# ---------------------------------------------------------------------------
# 8. All scenarios run without error
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_all_scenarios_run(scenario):
    e = NeonatalIncubatorEnv(scenario=scenario, seed=42)
    try:
        obs, info = e.reset(seed=42)
        assert obs.shape == (EXPECTED_OBS_DIM,), (
            f"scenario {scenario}: bad obs shape {obs.shape}"
        )
        assert obs.dtype == np.float32
        # Take a handful of steps to confirm stepping works in every scenario.
        for _ in range(10):
            obs, reward, terminated, truncated, info = e.step(SAFE_ACTION)
            assert obs.shape == (EXPECTED_OBS_DIM,)
            assert np.isfinite(reward)
            if terminated or truncated:
                break
    finally:
        e.close()


# ---------------------------------------------------------------------------
# 9. Reproducibility: same seed produces identical 10-step trajectories
# ---------------------------------------------------------------------------
def test_reproducibility_with_seed():
    e_a = NeonatalIncubatorEnv(scenario=1, seed=42)
    e_b = NeonatalIncubatorEnv(scenario=1, seed=42)
    try:
        obs_a, _ = e_a.reset(seed=42)
        obs_b, _ = e_b.reset(seed=42)
        np.testing.assert_array_equal(
            obs_a, obs_b, "Initial obs differ for identical seeds"
        )
        traj_a = [obs_a.copy()]
        traj_b = [obs_b.copy()]
        for _ in range(10):
            o_a, *_ = e_a.step(SAFE_ACTION)
            o_b, *_ = e_b.step(SAFE_ACTION)
            traj_a.append(o_a.copy())
            traj_b.append(o_b.copy())
        traj_a = np.array(traj_a)
        traj_b = np.array(traj_b)
        np.testing.assert_allclose(
            traj_a, traj_b, atol=1e-6,
            err_msg="Trajectories differ for identical seeds and actions",
        )
    finally:
        e_a.close()
        e_b.close()


# ---------------------------------------------------------------------------
# 10. Action decoding: action 0 vs action 24 produce different Ta trajectories
# ---------------------------------------------------------------------------
def test_action_decoding():
    # action 0  -> heater_idx=0 (u=0.0, Q=0W),   o2_idx=0 (FiO2=0.21)
    # action 24 -> heater_idx=4 (u=1.0, Q=360W), o2_idx=4 (FiO2=0.60)
    e_off = NeonatalIncubatorEnv(scenario=1, seed=42)
    e_on = NeonatalIncubatorEnv(scenario=1, seed=42)
    try:
        e_off.reset(seed=42)
        e_on.reset(seed=42)
        Ta_off = []
        Ta_on = []
        for _ in range(20):
            o_off, *_ = e_off.step(0)
            o_on, *_ = e_on.step(24)
            Ta_off.append(float(o_off[2]))  # OBS_NAMES[2] == 'Ta'
            Ta_on.append(float(o_on[2]))
        Ta_off = np.array(Ta_off)
        Ta_on = np.array(Ta_on)
        assert not np.allclose(Ta_off, Ta_on, atol=1e-3), (
            f"Ta trajectories identical for action 0 vs action 24: "
            f"off={Ta_off}, on={Ta_on}"
        )
        # Heater-on trajectory should be warmer than heater-off.
        assert Ta_on.mean() > Ta_off.mean(), (
            f"Expected action 24 to heat more than action 0; "
            f"got mean(Ta|a24)={Ta_on.mean():.3f} "
            f"<= mean(Ta|a0)={Ta_off.mean():.3f}"
        )
    finally:
        e_off.close()
        e_on.close()


# ---------------------------------------------------------------------------
# 11. Info dict contains expected keys
# ---------------------------------------------------------------------------
def test_info_dict_keys():
    e = NeonatalIncubatorEnv(scenario=1, seed=42)
    try:
        e.reset(seed=42)
        _, _, _, _, info = e.step(SAFE_ACTION)
        expected_keys = [
            "Tc", "SpO2", "heater_idx", "o2_idx",
            "r_thermal", "r_oxygen", "r_energy", "r_safety",
            "terminated", "truncated", "safe",
            "scenario_id", "w_thermal", "w_oxygen",
            "w_energy", "w_safety", "desat_offset",
        ]
        missing = [k for k in expected_keys if k not in info]
        assert not missing, f"info dict missing required keys: {missing}"
        # Sanity: heater_idx/o2_idx for action 6 are 1 and 1 respectively.
        assert info["heater_idx"] == 1, (
            f"expected heater_idx==1 for action 6, got {info['heater_idx']}"
        )
        assert info["o2_idx"] == 1, (
            f"expected o2_idx==1 for action 6, got {info['o2_idx']}"
        )
    finally:
        e.close()


# ---------------------------------------------------------------------------
# 12. get_state_dict returns dict with scenario_id_norm
# ---------------------------------------------------------------------------
def test_get_state_dict():
    e = NeonatalIncubatorEnv(scenario=1, seed=42)
    try:
        e.reset(seed=42)
        e.step(SAFE_ACTION)
        state = e.get_state_dict()
        assert isinstance(state, dict), (
            f"get_state_dict must return dict, got {type(state)}"
        )
        required_keys = [
            "Tc", "SpO2", "scenario", "step", "scenario_id_norm",
        ]
        missing = [k for k in required_keys if k not in state]
        assert not missing, (
            f"get_state_dict missing required keys: {missing}"
        )
        assert state["scenario"] == 1
        assert state["step"] >= 1, (
            f"step should be >=1 after one step, got {state['step']}"
        )
        # scenario 1 normalizes to 0.0, scenario 5 to 1.0
        assert state["scenario_id_norm"] == pytest.approx(0.0, abs=1e-6), (
            f"scenario 1 should have scenario_id_norm=0.0, "
            f"got {state['scenario_id_norm']}"
        )
    finally:
        e.close()


# ---------------------------------------------------------------------------
# 13. Safety termination on hypothermia (scenario 4 + action 0)
# ---------------------------------------------------------------------------
def test_safety_termination_on_hypothermia():
    e = NeonatalIncubatorEnv(scenario=4, seed=42)
    try:
        e.reset(seed=42)
        terminated = False
        info = {}
        for _ in range(50):
            obs, _, terminated, truncated, info = e.step(0)
            if terminated or truncated:
                break
        assert terminated, (
            "Scenario 4 with action 0 (no heater) should breach "
            "Tc < 35.5 within 50 steps"
        )
        assert info["Tc"] < 35.5, (
            f"Expected Tc < 35.5 at hypothermia termination, "
            f"got Tc={info['Tc']:.3f}"
        )
    finally:
        e.close()


# ---------------------------------------------------------------------------
# 14. Safety termination on hyperthermia/hyperoxia (scenario 1 + action 24)
# ---------------------------------------------------------------------------
def test_safety_termination_on_hyperthermia():
    e = NeonatalIncubatorEnv(scenario=1, seed=42)
    try:
        e.reset(seed=42)
        terminated = False
        info = {}
        for _ in range(50):
            obs, _, terminated, truncated, info = e.step(24)
            if terminated or truncated:
                break
        assert terminated, (
            "Scenario 1 with action 24 (full heater, full FiO2) should "
            "breach Tc > 38 or SpO2 > 96 within 50 steps"
        )
        breached = info["Tc"] > 38.0 or info["SpO2"] > 96.0
        assert breached, (
            f"Expected safety breach (Tc>38 or SpO2>96) at termination, "
            f"got Tc={info['Tc']:.3f}, SpO2={info['SpO2']:.3f}"
        )
    finally:
        e.close()


# ---------------------------------------------------------------------------
# 15. CRITICAL regression test: no first-step SpO2 collapse (old bug)
# ---------------------------------------------------------------------------
def test_no_first_step_collapse():
    """Regression test for the old 'first-step SpO2 collapse' bug.

    On scenario 1 the env resets SpO2 to 92.0 (the C_O2=21% equilibrium).
    After one step with action 6 (heater 0.25, FiO2=0.30), SpO2 should
    NOT suddenly drop to ~80 (the old broken behaviour). It must remain
    within 5 percentage points of 92.0.
    """
    e = NeonatalIncubatorEnv(scenario=1, seed=42)
    try:
        obs, info = e.reset(seed=42)
        spo2_reset = float(obs[4])  # OBS_NAMES[4] == 'SpO2'
        assert abs(spo2_reset - 92.0) < 0.5, (
            f"Expected reset SpO2 ~= 92.0, got {spo2_reset}"
        )
        obs_after, _, _, _, info_after = e.step(SAFE_ACTION)
        spo2_after = float(obs_after[4])
        delta = abs(spo2_after - 92.0)
        assert delta < 5.0, (
            f"First-step SpO2 collapse regression: "
            f"reset={spo2_reset:.3f}, after_step={spo2_after:.3f}, "
            f"|delta|={delta:.3f} >= 5.0"
        )
    finally:
        e.close()


# ---------------------------------------------------------------------------
# 16. CRITICAL V2-04 regression: scenario_id exposed in observation
# ---------------------------------------------------------------------------
def test_markov_property_scenario_id_exposed():
    """obs[10] (scenario_id) is present; scenario 1 -> 0.0, scenario 5 -> 1.0.

    This is the V2-04 fix that makes the environment a fully-observable MDP
    by including the normalized scenario_id in the observation vector.
    """
    e1 = NeonatalIncubatorEnv(scenario=1, seed=42)
    try:
        obs1, _ = e1.reset(seed=42)
        assert obs1.shape[0] == EXPECTED_OBS_DIM, (
            f"obs must be 11-dim for scenario_id exposure, got {obs1.shape}"
        )
        sid1 = float(obs1[10])  # OBS_NAMES[10] == 'scenario_id'
        assert sid1 == pytest.approx(0.0, abs=1e-6), (
            f"scenario 1 should have scenario_id=0.0, got {sid1}"
        )
    finally:
        e1.close()

    e5 = NeonatalIncubatorEnv(scenario=5, seed=42)
    try:
        obs5, _ = e5.reset(seed=42)
        sid5 = float(obs5[10])
        assert sid5 == pytest.approx(1.0, abs=1e-6), (
            f"scenario 5 should have scenario_id=1.0, got {sid5}"
        )
    finally:
        e5.close()


# ---------------------------------------------------------------------------
# 17. Disturbance flag is set during porthole window (scenario 3)
# ---------------------------------------------------------------------------
def test_disturbance_flag_during_porthole():
    """Scenario 3 porthole: info['disturbance'] == 1.0 during the window.

    The env's porthole window is [PORTHOLE_START=50, PORTHOLE_END=80] checked
    on the PRE-increment ``current_step``. Since ``current_step`` is
    incremented AFTER the step body, the disturbance flag is exposed in
    ``info['step']`` in [51, 81] (the post-increment range).
    """
    e = NeonatalIncubatorEnv(scenario=3, seed=42)
    try:
        e.reset(seed=42)
        disturbances = []
        for _ in range(85):
            obs, _, terminated, truncated, info = e.step(SAFE_ACTION)
            disturbances.append((info["step"], info["disturbance"]))
            if terminated or truncated:
                break
        # Post-increment porthole window is info['step'] in [51, 81].
        porthole_entries = [
            d for (s, d) in disturbances if 51 <= s <= 81
        ]
        assert len(porthole_entries) > 0, (
            "Did not reach porthole window (info['step'] in [51, 81]); "
            f"collected steps: {[s for s, _ in disturbances]}"
        )
        bad = [
            (s, d) for (s, d) in disturbances
            if 51 <= s <= 81 and d != 1.0
        ]
        assert not bad, (
            f"disturbance flag should be 1.0 throughout the porthole "
            f"window (info['step'] in [51, 81]); offending entries: {bad}"
        )
    finally:
        e.close()


# ---------------------------------------------------------------------------
# 18. CRITICAL V2-01 regression: scenario 5 desaturation drives a SpO2 dip
# ---------------------------------------------------------------------------
def test_scenario5_desaturation_effect():
    """Scenario 5 desaturation must lower SpO2 vs scenario 1 (same action).

    On scenario 5 with action 1*5+1=6 (heater 0.25, FiO2=0.30), the
    physiological desaturation drive on steps [60, 120] must lower the
    minimum SpO2 observed in that window by at least 2 percentage points
    compared to the same action on scenario 1.
    """
    window_spo2 = {}
    for scn in (1, 5):
        e = NeonatalIncubatorEnv(scenario=scn, seed=42)
        try:
            e.reset(seed=42)
            spo2s = []
            for _ in range(125):
                obs, _, terminated, truncated, info = e.step(SAFE_ACTION)
                # info['step'] is post-increment; the desaturation window
                # is [60, 120] pre-increment -> post-increment [61, 121].
                if 60 <= info["step"] <= 121:
                    spo2s.append(float(obs[4]))
                if terminated or truncated:
                    break
            window_spo2[scn] = min(spo2s) if spo2s else float("nan")
        finally:
            e.close()

    min_s1 = window_spo2[1]
    min_s5 = window_spo2[5]
    drop = min_s1 - min_s5
    assert drop >= 2.0, (
        f"Scenario 5 desaturation effect too small: "
        f"min SpO2 scenario 1 = {min_s1:.3f}, scenario 5 = {min_s5:.3f}, "
        f"drop = {drop:.3f} (required >= 2.0)"
    )


# ---------------------------------------------------------------------------
# 19. render() and close() do not raise
# ---------------------------------------------------------------------------
def test_render_and_close():
    e = NeonatalIncubatorEnv(scenario=1, seed=42)
    try:
        e.reset(seed=42)
        e.step(SAFE_ACTION)
        e.render()
        e.close()
    finally:
        # Calling close() twice should also be safe.
        try:
            e.close()
        except Exception:  # pragma: no cover -- defensive
            pass


# ---------------------------------------------------------------------------
# 20. OBS_NAMES constant has 11 entries (V3 change from 10)
# ---------------------------------------------------------------------------
def test_obs_names_constant():
    assert isinstance(OBS_NAMES, (tuple, list)), (
        f"OBS_NAMES should be tuple or list, got {type(OBS_NAMES)}"
    )
    assert len(OBS_NAMES) == EXPECTED_OBS_DIM, (
        f"OBS_NAMES should have {EXPECTED_OBS_DIM} entries, "
        f"got {len(OBS_NAMES)}"
    )
    e = NeonatalIncubatorEnv(scenario=1, seed=42)
    try:
        obs, _ = e.reset(seed=42)
        assert obs.shape[0] == len(OBS_NAMES), (
            f"obs dim {obs.shape[0]} != len(OBS_NAMES) {len(OBS_NAMES)}"
        )
        for name in OBS_NAMES:
            assert isinstance(name, str) and len(name) > 0, (
                f"OBS_NAMES contains invalid entry: {name!r}"
            )
    finally:
        e.close()


# ---------------------------------------------------------------------------
# 21. ACTION_TABLE has 25 entries with the expected keys and decoding
# ---------------------------------------------------------------------------
def test_action_table():
    assert isinstance(ACTION_TABLE, list), (
        f"ACTION_TABLE should be a list, got {type(ACTION_TABLE)}"
    )
    assert len(ACTION_TABLE) == EXPECTED_N_ACTIONS, (
        f"ACTION_TABLE should have {EXPECTED_N_ACTIONS} entries, "
        f"got {len(ACTION_TABLE)}"
    )
    required_keys = {
        "action", "heater_idx", "o2_idx",
        "heater_level", "fio2_command", "Q_heater_W",
    }
    for i, entry in enumerate(ACTION_TABLE):
        assert isinstance(entry, dict), (
            f"ACTION_TABLE[{i}] should be a dict, got {type(entry)}"
        )
        missing = required_keys - set(entry.keys())
        assert not missing, (
            f"ACTION_TABLE[{i}] missing required keys: {missing}"
        )
        assert entry["action"] == i, (
            f"ACTION_TABLE[{i}]['action'] != {i}: got {entry['action']}"
        )
        assert 0 <= entry["heater_idx"] <= 4, (
            f"ACTION_TABLE[{i}]['heater_idx'] out of [0,4]: "
            f"{entry['heater_idx']}"
        )
        assert 0 <= entry["o2_idx"] <= 4, (
            f"ACTION_TABLE[{i}]['o2_idx'] out of [0,4]: {entry['o2_idx']}"
        )
        # Decoding invariant: action == heater_idx * 5 + o2_idx.
        assert entry["heater_idx"] * 5 + entry["o2_idx"] == i, (
            f"ACTION_TABLE[{i}] decoding broken: "
            f"heater_idx={entry['heater_idx']}, o2_idx={entry['o2_idx']}"
        )


# ---------------------------------------------------------------------------
# 22. V2-03 regression: observed rates match realized state change
# ---------------------------------------------------------------------------
def test_observed_rates_match_realized_change():
    """|dTa_dt - (Ta_new - Ta_old)/dt| < 0.01 and same for dC_O2_dt.

    The env reports realized (post-clipping) per-second rates in obs[6,7].
    These must equal the actual state change divided by dt (within a small
    tolerance for floating-point noise from the defensive clip).
    """
    e = NeonatalIncubatorEnv(scenario=1, seed=42)
    try:
        obs, _ = e.reset(seed=42)
        dt = e.dt
        prev_Ta = float(obs[2])
        prev_C_O2 = float(obs[3])
        for _ in range(8):
            obs, _, _, _, _ = e.step(SAFE_ACTION)
            Ta_new = float(obs[2])
            C_O2_new = float(obs[3])
            dTa_dt_reported = float(obs[6])
            dC_O2_dt_reported = float(obs[7])
            expected_dTa_dt = (Ta_new - prev_Ta) / dt
            expected_dC_O2_dt = (C_O2_new - prev_C_O2) / dt
            assert abs(dTa_dt_reported - expected_dTa_dt) < 0.01, (
                f"dTa_dt mismatch: reported={dTa_dt_reported:.6f}, "
                f"expected={expected_dTa_dt:.6f}, "
                f"Ta {prev_Ta:.4f}->{Ta_new:.4f}"
            )
            assert abs(dC_O2_dt_reported - expected_dC_O2_dt) < 0.01, (
                f"dC_O2_dt mismatch: reported={dC_O2_dt_reported:.6f}, "
                f"expected={expected_dC_O2_dt:.6f}, "
                f"C_O2 {prev_C_O2:.4f}->{C_O2_new:.4f}"
            )
            prev_Ta = Ta_new
            prev_C_O2 = C_O2_new
    finally:
        e.close()


# ---------------------------------------------------------------------------
# 23. V2-05 regression: reward formula independently recomputed from info
# ---------------------------------------------------------------------------
def test_reward_formula_matches_info():
    """R = w_thermal*r_thermal + w_oxygen*r_oxygen + w_energy*r_energy
        + w_safety*r_safety  must match info['reward'] within 1e-6."""
    e = NeonatalIncubatorEnv(scenario=1, seed=42)
    try:
        e.reset(seed=42)
        for _ in range(10):
            obs, reward, terminated, truncated, info = e.step(SAFE_ACTION)
            computed = (
                info["w_thermal"] * info["r_thermal"]
                + info["w_oxygen"] * info["r_oxygen"]
                + info["w_energy"] * info["r_energy"]
                + info["w_safety"] * info["r_safety"]
            )
            assert abs(computed - info["reward"]) < 1e-6, (
                f"reward formula mismatch at step {info['step']}: "
                f"computed={computed:.6f}, info['reward']={info['reward']:.6f}, "
                f"diff={abs(computed - info['reward']):.2e}"
            )
            if terminated or truncated:
                break
    finally:
        e.close()


# ---------------------------------------------------------------------------
# 24. V3-01 regression: custom config overrides propagate to runtime
# ---------------------------------------------------------------------------
def test_config_override():
    """V3-01 regression: custom config overrides must propagate to runtime.

    Verifies that load_config() applies reward weights, safety penalty,
    and q_heater_max, AND that action_table is rebuilt per-instance.
    """
    custom = {
        "reward": {"weights": {"w_energy": 4.0}, "r_safety_penalty": 777.0},
        "environment": {"q_heater_max": 400.0},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(custom, f)
        path = f.name
    try:
        env = NeonatalIncubatorEnv(config_path=path)
        assert env.w_energy == 4.0, f"w_energy={env.w_energy} (expected 4.0)"
        assert env.r_safety_penalty == 777.0, f"r_safety_penalty={env.r_safety_penalty}"
        assert env.q_heater_max == 400.0, f"q_heater_max={env.q_heater_max}"
        max_q = max(row["Q_heater_W"] for row in env.action_table)
        assert max_q == 400.0, f"max Q_heater_W={max_q} (expected 400.0)"
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 25. gymnasium.utils.env_checker.check_env passes
# ---------------------------------------------------------------------------
def test_check_env_passes():
    """The env must pass Gymnasium's official API/dynamics checker."""
    from gymnasium.utils.env_checker import check_env

    e = NeonatalIncubatorEnv(scenario=1, seed=42)
    try:
        # check_env unwraps and resets the env; should not raise.
        check_env(e)
    finally:
        e.close()


# ---------------------------------------------------------------------------
# Module-level sanity: confirm N_SCENARIOS == 5
# ---------------------------------------------------------------------------
def test_n_scenarios_constant():
    assert N_SCENARIOS == 5, (
        f"N_SCENARIOS should be 5, got {N_SCENARIOS}"
    )


# ===========================================================================
# F4 AUDIT -- additional regression / API-contract tests (additive only).
# These tests do NOT modify any existing test; they only append new checks
# discovered during the BLUE-TEAM F4 audit pass.
# ===========================================================================


# ---------------------------------------------------------------------------
# F4-01. step() after terminated=True must not crash mysteriously.
# ---------------------------------------------------------------------------
def test_step_after_terminated():
    """Calling ``step()`` after the episode has terminated must EITHER:

      * succeed and return a valid 5-tuple (the env keeps "stepping"), OR
      * raise a clear, documented exception (e.g. ``RuntimeError``).

    It must NOT crash with a mysterious error such as ``AttributeError``
    or ``IndexError`` that would leak internal implementation details.
    """
    e = NeonatalIncubatorEnv(scenario=4, seed=42)  # cold-start -> fast term.
    try:
        e.reset(seed=42)
        terminated = False
        truncated = False
        info = {}
        # Drive the env to a safety termination (scenario 4 + heater off
        # crosses the hypothermia floor Tc < 35.5 within a few steps).
        for _ in range(50):
            obs, _, terminated, truncated, info = e.step(0)
            if terminated or truncated:
                break
        assert terminated, (
            "test precondition: scenario 4 + action 0 should terminate "
            f"within 50 steps (got terminated={terminated}, "
            f"truncated={truncated}, step={info.get('step')})"
        )

        # Now the actual F4 check: calling step() after termination.
        clear_errors = (
            ValueError, RuntimeError, TypeError, AssertionError, LookupError,
        )
        try:
            result = e.step(SAFE_ACTION)
        except clear_errors as ex:
            # Acceptable: a documented, explicit error. Just ensure it has
            # a non-empty message so a debugger can act on it.
            assert str(ex), (
                f"step-after-terminated raised {type(ex).__name__} with "
                f"empty message"
            )
            return
        # Otherwise: the call must produce a valid Gymnasium 5-tuple.
        assert isinstance(result, tuple), (
            f"step() after terminated must return tuple, got {type(result)}"
        )
        assert len(result) == 5, (
            f"step() after terminated must return 5-tuple, got len={len(result)}"
        )
        obs2, reward2, term2, trunc2, info2 = result
        assert isinstance(obs2, np.ndarray), (
            f"obs must be ndarray after post-term step, got {type(obs2)}"
        )
        assert obs2.shape == (EXPECTED_OBS_DIM,), (
            f"obs shape {obs2.shape} != ({EXPECTED_OBS_DIM},) after post-term step"
        )
        assert obs2.dtype == np.float32
        assert np.isfinite(reward2), (
            f"reward must be finite after post-term step, got {reward2}"
        )
        assert isinstance(term2, (bool, np.bool_))
        assert isinstance(trunc2, (bool, np.bool_))
        assert isinstance(info2, dict)
    finally:
        e.close()


# ---------------------------------------------------------------------------
# F4-02. observation_space.contains(obs) is True immediately after reset.
# ---------------------------------------------------------------------------
def test_observation_contains_after_reset():
    """``env.observation_space.contains(obs)`` must be True on the very first
    observation returned by ``reset()``, BEFORE any ``step()`` is called.

    The existing ``test_observation_within_bounds`` checks this implicitly as a
    precondition to its stepping loop; this test isolates the reset-time
    contract so a future regression to reset() initialisation cannot silently
    push the first observation outside the declared Box bounds.
    """
    for scenario in (1, 2, 3, 4, 5):
        e = NeonatalIncubatorEnv(scenario=scenario, seed=42)
        try:
            obs, info = e.reset(seed=42)
            assert isinstance(obs, np.ndarray), (
                f"scenario {scenario}: obs must be ndarray, got {type(obs)}"
            )
            assert obs.dtype == np.float32, (
                f"scenario {scenario}: obs dtype must be float32, got {obs.dtype}"
            )
            assert e.observation_space.contains(obs), (
                f"scenario {scenario}: reset obs {obs} not contained in "
                f"observation_space {e.observation_space}"
            )
        finally:
            e.close()


# ---------------------------------------------------------------------------
# F4-03. obs[10] (scenario_id) == (scenario - 1) / (N_SCENARIOS - 1).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", [1, 3, 5])
def test_scenario_id_feature(scenario):
    """The normalized scenario_id feature in obs[10] must equal
    ``(scenario - 1) / (N_SCENARIOS - 1)`` for scenarios 1, 3, 5.

    * scenario 1 -> 0.0  (baseline, normal)
    * scenario 3 -> 0.5  (mid-range porthole)
    * scenario 5 -> 1.0  (worst-case desaturation)

    The existing ``test_markov_property_scenario_id_exposed`` only checks the
    two endpoints (scenarios 1 and 5). This test additionally pins the
    intermediate scenario 3 to 0.5 so that a regression in the normalization
    denominator (e.g. dividing by ``N_SCENARIOS`` instead of ``N-1``) cannot
    slip through.
    """
    e = NeonatalIncubatorEnv(scenario=scenario, seed=42)
    try:
        obs, _ = e.reset(seed=42)
        assert obs.shape[0] == EXPECTED_OBS_DIM, (
            f"scenario {scenario}: obs must be {EXPECTED_OBS_DIM}-dim, "
            f"got shape {obs.shape}"
        )
        expected = (scenario - 1) / max(1, N_SCENARIOS - 1)
        sid = float(obs[10])  # OBS_NAMES[10] == 'scenario_id'
        assert sid == pytest.approx(expected, abs=1e-6), (
            f"scenario {scenario}: obs[10]={sid} != expected {expected} "
            f"(normalized scenario_id feature)"
        )
    finally:
        e.close()


# ---------------------------------------------------------------------------
# F4-04. V3-03 regression: desat_offset == 5.0 on the default env.
# ---------------------------------------------------------------------------
def test_desat_offset_is_5():
    """The V3-03 fix reduced ``desat_offset`` from 8.0 to 5.0 to keep the
    scenario-5 desaturation challenge recoverable (max FiO2=0.60 -> C_O2=56.45
    -> SpO2_eq = 92 + 0.15*35.45 - 5 = 92.32%, which is inside the safe band).

    This test pins the resolved instance-level ``desat_offset`` to 5.0 so a
    future accidental revert to 8.0 is caught immediately.
    """
    e = NeonatalIncubatorEnv(scenario=1, seed=42)
    try:
        assert hasattr(e, "desat_offset"), (
            "env must expose a 'desat_offset' instance attribute"
        )
        assert e.desat_offset == pytest.approx(5.0, abs=1e-9), (
            f"desat_offset should be 5.0 (V3-03), got {e.desat_offset}"
        )
        # Also verify the underlying config dict carries the same value so
        # both surfaces (instance attr + resolved config) stay in sync.
        cfg = getattr(e, "_config", None)
        assert isinstance(cfg, dict), "env._config must be a dict"
        assert cfg.get("desat_offset") == pytest.approx(5.0, abs=1e-9), (
            f"_config['desat_offset'] should be 5.0, got {cfg.get('desat_offset')}"
        )
    finally:
        e.close()


# ---------------------------------------------------------------------------
# F4-05. env._config is a resolved dict carrying the expected keys.
# ---------------------------------------------------------------------------
def test_config_resolved_dict():
    """V3-01 fix: ``load_config()`` returns a resolved dict that the env
    stores on ``self._config``. This test pins the contract: ``_config`` is a
    ``dict``, is per-instance, and carries the full set of expected keys.

    The expected key set is drawn from the docstring of ``load_config`` in
    ``src/environment.py`` and from the env constructor's reads.
    """
    e = NeonatalIncubatorEnv(scenario=1, seed=42)
    try:
        assert hasattr(e, "_config"), "env must expose _config"
        assert isinstance(e._config, dict), (
            f"_config must be a dict, got {type(e._config)}"
        )
        expected_keys = {
            # environment / time discretisation
            "dt", "max_steps", "n_substeps", "q_heater_max",
            # physics -- thermal
            "k_loss", "k_baby", "c_air", "c_core",
            "porthole_h_c_factor", "porthole_start", "porthole_end",
            # physics -- oxygen + SpO2
            "k_blend", "k_leak",
            "o2_disturb_start", "o2_disturb_end", "desat_offset",
            "k_spo2", "spo2_base", "spo2_gain", "spo2_min", "spo2_max",
            # reward
            "w_thermal", "w_oxygen", "w_energy", "w_safety",
            "r_safety_penalty", "tc_setpoint",
            "spo2_target_low", "spo2_target_high",
            # safety
            "tc_safe_low", "tc_safe_high",
            "spo2_safe_low", "spo2_safe_high",
        }
        present = set(e._config.keys())
        missing = expected_keys - present
        assert not missing, (
            f"env._config missing expected keys: {sorted(missing)}"
        )
        # Values must be non-None (resolved) for every expected key.
        none_keys = sorted(k for k in expected_keys if e._config[k] is None)
        assert not none_keys, (
            f"env._config has None values for keys: {none_keys}"
        )
        # Sanity: the values on the instance must match the resolved config
        # for at least the reward + desat keys (catches "config ignored"
        # regressions).
        assert e.w_thermal == pytest.approx(float(e._config["w_thermal"]))
        assert e.w_oxygen == pytest.approx(float(e._config["w_oxygen"]))
        assert e.w_energy == pytest.approx(float(e._config["w_energy"]))
        assert e.w_safety == pytest.approx(float(e._config["w_safety"]))
        assert e.desat_offset == pytest.approx(float(e._config["desat_offset"]))
    finally:
        e.close()


# ---------------------------------------------------------------------------
# F4-06. Instance params are not module globals: per-instance isolation.
# ---------------------------------------------------------------------------
def test_instance_params_not_module_globals():
    """V3-01 fix: physics / reward parameters live on the instance, not on
    module globals. Mutating one instance's attribute must NOT leak into
    another instance (this is the regression that the V3-01 load_config() fix
    was designed to prevent).

    Concretely:
      * ``env.k_loss`` is set (not None) and is a float.
      * Setting ``env_a.k_loss = X`` does not change ``env_b.k_loss``.
      * The module-level constant ``environment.K_LOSS`` is also untouched.
    """
    import environment as env_mod

    a = NeonatalIncubatorEnv(scenario=1, seed=42)
    b = NeonatalIncubatorEnv(scenario=1, seed=42)
    try:
        # 1. Instance attribute is populated and is a real number.
        assert a.k_loss is not None, "env.k_loss must be set, not None"
        assert isinstance(a.k_loss, (int, float, np.floating)), (
            f"env.k_loss must be a number, got {type(a.k_loss)}"
        )
        assert b.k_loss is not None
        original_b = float(b.k_loss)
        original_module = float(env_mod.K_LOSS)

        # 2. Mutating instance A's k_loss does NOT affect instance B.
        a.k_loss = 123.456
        assert a.k_loss == pytest.approx(123.456, abs=1e-9), (
            f"assignment to a.k_loss did not stick: got {a.k_loss}"
        )
        assert b.k_loss == pytest.approx(original_b, abs=1e-9), (
            f"mutation of a.k_loss leaked into b.k_loss: "
            f"b.k_loss={b.k_loss} (expected {original_b})"
        )

        # 3. The module-level constant is also untouched (defensive -- if the
        #    module global WERE mutated, downstream code re-importing the
        #    constant would see the wrong value).
        assert env_mod.K_LOSS == pytest.approx(original_module, abs=1e-9), (
            f"module-level K_LOSS changed unexpectedly: "
            f"got {env_mod.K_LOSS} (expected {original_module})"
        )

        # 4. Different config_path instances resolve independently: building a
        #    second env AFTER mutating a.k_loss must still read the original
        #    resolved value, not 123.456.
        c = NeonatalIncubatorEnv(scenario=1, seed=42)
        try:
            assert c.k_loss == pytest.approx(original_b, abs=1e-9), (
                f"new instance c.k_loss={c.k_loss} should match the default "
                f"{original_b}, not the mutated value of a"
            )
        finally:
            c.close()
    finally:
        a.close()
        b.close()
