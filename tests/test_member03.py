#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_member03.py -- Comprehensive Unit and Regression Tests for Member 03 Deliverables
=============================================================================
Tests:
  1. Trained agent loading and Q-value inference fidelity.
  2. Stratified background dataset collection and disturbance/scenario coverage.
  3. Feature ordering consistency with OBS_NAMES.
  4. SHAP explainer efficiency property (|Q - (E[Q] + sum(phi))| < 1e-4).
  5. Single-state SHAP explanation structure and policy-level explanations.
  6. Contrastive action preference attribution.
  7. State-action alignment verification in validation trajectories.
  8. Decision Q-value fidelity across all 25 action heads.
  9. Real simulation state physics consistency (e.g. ambient drop equation).
 10. Integrity of generated artifacts and non-zero file sizes.
=============================================================================
"""

import json
import os
import sys
import numpy as np
import pytest

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_DIR = os.path.abspath(os.path.join(_CUR_DIR, ".."))
_SRC_DIR = os.path.join(_PROJ_DIR, "src")
for p in (_PROJ_DIR, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from environment import NeonatalIncubatorEnv, OBS_NAMES, ACTION_TABLE
from shap_explainer import (
    load_trained_agent,
    collect_background_dataset,
    DDQNShapExplainer,
    extract_real_simulation_states,
)
from validation import evaluate_trajectory
from rl_utils import AgentPolicy, BaselinePolicy


@pytest.fixture(scope="module")
def checkpoint_path():
    ckpt = os.path.join(_PROJ_DIR, "models", "ddqn_tuned_seed0", "best.pt")
    if not os.path.isfile(ckpt):
        pytest.skip(f"Checkpoint not found at {ckpt}")
    return ckpt


@pytest.fixture(scope="module")
def agent(checkpoint_path):
    return load_trained_agent(checkpoint_path, device="cpu")


@pytest.fixture(scope="module")
def background_data(agent):
    raw_bg, norm_bg, metadata = collect_background_dataset(agent=agent, seed=42)
    return raw_bg, norm_bg, metadata


# ---------------------------------------------------------------------------
# Test Group 1: Agent & Model Architecture
# ---------------------------------------------------------------------------

def test_agent_loading(agent):
    """Verify DDQN agent loads in eval mode with correct dimension contracts."""
    assert agent.state_size == 11
    assert agent.action_size == 25
    ref_obs = np.array([37.0, 36.5, 34.0, 21.0, 92.0, 25.0, 0.0, 0.0, 0.5, 0.0, 0.0], dtype=np.float32)
    norm = agent.preprocess(ref_obs)
    q_vals = agent.q_values(norm)
    assert q_vals.shape == (1, 25)
    assert np.all(np.isfinite(q_vals))


def test_feature_ordering_consistency():
    """Verify that feature names and indices strictly match environment contracts."""
    expected_order = [
        "Tc", "Tsk", "Ta", "C_O2", "SpO2", "Tamb",
        "dTa_dt", "dC_O2_dt", "phase", "disturbance", "scenario_id",
    ]
    assert list(OBS_NAMES) == expected_order, f"OBS_NAMES does not match expected: {OBS_NAMES}"
    assert len(OBS_NAMES) == 11


# ---------------------------------------------------------------------------
# Test Group 2: Stratified Background Sampling Coverage
# ---------------------------------------------------------------------------

def test_background_collection_coverage(background_data):
    """Verify stratified background dataset covers all 5 scenarios and all disturbances."""
    raw_bg, norm_bg, metadata = background_data
    assert raw_bg.shape == norm_bg.shape
    assert raw_bg.shape[1] == len(OBS_NAMES)
    assert len(raw_bg) == 100, f"Expected 100 background samples, got {len(raw_bg)}"
    assert np.all(np.isfinite(raw_bg))
    assert np.all(np.isfinite(norm_bg))
    assert "sampling_strategy" in metadata

    # Scenario coverage: scenario_id feature is normalized as (scenario - 1) / 4
    scenario_ids = np.unique(np.round(raw_bg[:, 10] * 4 + 1)).astype(int)
    for scn in [1, 2, 3, 4, 5]:
        assert scn in scenario_ids, f"Scenario {scn} missing from background dataset"

    # Disturbance coverage: disturbance index is 9
    disturbances = np.unique(np.round(raw_bg[:, 9])).astype(int)
    for dist in [0, 1, 2]:
        assert dist in disturbances, f"Disturbance {dist} missing from background dataset"


def test_background_physics_bounds(background_data):
    """Verify that background observations strictly observe physical boundaries."""
    raw_bg, _, _ = background_data
    # Core temperature: 30 to 42 C
    assert np.all(raw_bg[:, 0] >= 30.0) and np.all(raw_bg[:, 0] <= 42.0)
    # SpO2: 70 to 100 %
    assert np.all(raw_bg[:, 4] >= 70.0) and np.all(raw_bg[:, 4] <= 100.0)
    # Ambient temp: 15 to 30 C
    assert np.all(raw_bg[:, 5] >= 15.0) and np.all(raw_bg[:, 5] <= 30.0)
    # Phase: 0 to 1.0
    assert np.all(raw_bg[:, 8] >= 0.0) and np.all(raw_bg[:, 8] <= 1.0)


# ---------------------------------------------------------------------------
# Test Group 3: SHAP Explainer Axioms & Fidelity
# ---------------------------------------------------------------------------

def test_shap_deep_explainer_efficiency(agent, background_data):
    """Verify SHAP efficiency axiom (|sum(phi) + base - Q| < 1e-4) across action heads."""
    _, norm_bg, _ = background_data
    explainer = DDQNShapExplainer(agent=agent, background_norm=norm_bg, method="deep")

    test_state = norm_bg[0:1]
    sv = explainer.explain_norm_batch(test_state)  # (1, 11, 25)
    ev = explainer.expected_values  # (25,)

    q_pred = agent.online.forward_np(test_state)[0]
    for a in [0, 5, 12, 20, 22, 24]:
        phi_sum = float(np.sum(sv[0, :, a]))
        base = float(ev[a])
        pred = float(q_pred[a])
        assert abs(pred - (base + phi_sum)) < 1e-4, f"Action {a} efficiency failed: pred={pred}, base+phi={base+phi_sum}"


def test_shap_explain_state_structure(agent, background_data):
    """Verify structure and completeness of single-state explanation output."""
    _, norm_bg, _ = background_data
    explainer = DDQNShapExplainer(agent=agent, background_norm=norm_bg, method="deep")

    raw_sample = np.array([36.0, 35.5, 32.0, 21.0, 90.0, 20.0, -0.05, 0.0, 0.5, 1.0, 0.25], dtype=np.float32)
    exp = explainer.explain_state(raw_sample)

    assert "greedy_action" in exp
    assert "explained_action" in exp
    assert "shap_values" in exp
    assert len(exp["shap_values"]) == len(OBS_NAMES)
    assert exp["efficiency_error"] < 1e-4


def test_shap_action_difference(agent, background_data):
    """Verify contrastive action preference decomposition."""
    _, norm_bg, _ = background_data
    explainer = DDQNShapExplainer(agent=agent, background_norm=norm_bg, method="deep")

    raw_sample = np.array([36.0, 35.5, 32.0, 21.0, 90.0, 20.0, -0.05, 0.0, 0.5, 1.0, 0.25], dtype=np.float32)
    contrast = explainer.explain_action_difference(raw_sample, action_a=20, action_b=0)

    assert "delta_q" in contrast
    assert "delta_shap" in contrast
    assert len(contrast["delta_shap"]) == 11
    assert contrast["efficiency_error"] < 1e-4


# ---------------------------------------------------------------------------
# Test Group 4: State-Action Alignment in Validation Trajectories
# ---------------------------------------------------------------------------

def test_validation_state_action_alignment(agent):
    """
    CRITICAL AUDIT REGRESSION TEST:
    Verify that for every step t in a recorded trajectory:
      1. selected_action is chosen based on decision_state (decision_obs_*).
      2. decision_q_chosen matches agent's Q-value for selected_action evaluated on decision_state.
      3. next_obs_* matches the state resulting from the transition.
      4. decision_state != next_state for steps where state dynamics advance.
      5. All 25 action Q-values (q_val_0..24) are stored and match agent forward pass on decision_state.
    """
    env = NeonatalIncubatorEnv(scenario=1, max_steps=5)
    policy = AgentPolicy(agent, name="ddqn")

    rows, metrics = evaluate_trajectory(env, policy, seed=901000, explainer=None)
    assert len(rows) == 6  # Step 0 initial row + 5 transitions

    for step_idx in range(1, 6):
        row = rows[step_idx]
        action = row["selected_action"]
        assert 0 <= action < 25, f"Invalid selected action {action} at step {step_idx}"

        # Reconstruct decision state vector from row
        decision_vec = np.array([row[f"decision_obs_{feat}"] for feat in OBS_NAMES], dtype=np.float32)
        next_vec = np.array([row[f"next_obs_{feat}"] for feat in OBS_NAMES], dtype=np.float32)

        # 1. State must evolve: decision_state != next_state
        assert not np.allclose(decision_vec, next_vec), f"Step {step_idx}: decision and next state are identical!"

        # 2. Forward pass on decision state
        norm_decision = agent.preprocess(decision_vec)
        q_values_computed = agent.q_values(norm_decision)[0]

        # 3. Decision Q-value matching
        expected_q_chosen = q_values_computed[action]
        assert abs(row["decision_q_chosen"] - expected_q_chosen) < 1e-4, (
            f"Step {step_idx}: decision_q_chosen {row['decision_q_chosen']} != {expected_q_chosen}"
        )

        # 4. Check all 25 Q-values in CSV row match computed Q-values
        for a_idx in range(25):
            recorded_q = row[f"q_val_{a_idx}"]
            assert abs(recorded_q - q_values_computed[a_idx]) < 1e-4, (
                f"Step {step_idx}: q_val_{a_idx} mismatch ({recorded_q} vs {q_values_computed[a_idx]})"
            )

        # 5. Greedy action consistency
        greedy_act = int(np.argmax(q_values_computed))
        assert action == greedy_act, f"Step {step_idx}: DDQN chose non-greedy action {action} vs {greedy_act}"


def test_validation_action_switching_counter(agent):
    """Verify that action switches and switch ratios are recorded in episode metrics."""
    env = NeonatalIncubatorEnv(scenario=1, max_steps=20)
    policy = AgentPolicy(agent, name="ddqn")

    _, metrics = evaluate_trajectory(env, policy, seed=901000)
    assert "action_switches" in metrics
    assert "action_switch_ratio" in metrics
    assert 0 <= metrics["action_switches"] <= 19
    assert 0.0 <= metrics["action_switch_ratio"] <= 1.0


# ---------------------------------------------------------------------------
# Test Group 5: Real Simulation States & Artifact Verification
# ---------------------------------------------------------------------------

def test_real_simulation_states_physics(agent):
    """Verify extracted simulation states match physical equations (e.g. ambient drop)."""
    states = extract_real_simulation_states(agent=agent)
    amb_key = "Ambient Cooling Ramp (Scn 2, Step 150)"
    assert amb_key in states
    amb_state = states[amb_key]
    step = amb_state["step"]
    # In env.step(), Tamb is updated at the start of transition before step increments
    expected_tamb = 25.0 - 7.0 * ((step - 1) / 200.0)
    actual_tamb = amb_state["raw_obs"][OBS_NAMES.index("Tamb")]
    assert abs(actual_tamb - expected_tamb) < 1e-3, f"Ambient temp mismatch: {actual_tamb} vs {expected_tamb}"

    # Porthole scenario at step 65: disturbance flag must be 1.0 (active porthole)
    port_key = "Porthole Open Event (Scn 3, Step 65)"
    assert port_key in states
    port_state = states[port_key]
    assert port_state["raw_obs"][OBS_NAMES.index("disturbance")] == 1.0

    # Oxygen desaturation at step 90: disturbance flag must be 2.0 (active desaturation)
    o2_key = "Oxygen Desaturation Peak (Scn 5, Step 90)"
    assert o2_key in states
    o2_state = states[o2_key]
    assert o2_state["raw_obs"][OBS_NAMES.index("disturbance")] == 2.0


def test_generated_artifacts_integrity():
    """Verify all required deliverables exist and are non-empty."""
    base_dir = os.path.join(_PROJ_DIR, "outputs", "member03")
    val_dir = os.path.join(base_dir, "validation")
    shap_dir = os.path.join(base_dir, "shap")

    # CSV deliverables
    for csv_file in [
        "normal_scenario_results.csv",
        "abnormal_temperature_results.csv",
        "abnormal_oxygen_results.csv",
        "validation_episodes.csv",
    ]:
        path = os.path.join(val_dir, csv_file)
        assert os.path.isfile(path), f"Missing validation CSV: {path}"
        assert os.path.getsize(path) > 1000, f"Validation CSV {path} is too small"

    # Validation JSON
    val_json_path = os.path.join(val_dir, "validation_summary.json")
    assert os.path.isfile(val_json_path)
    with open(val_json_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    assert val_data["evaluation_design"]["total_evaluation_episodes"] == 50

    # SHAP Plots
    for plot_file in [
        "shap_summary.png",
        "shap_feature_importance.png",
        "shap_policy_feature_importance.png",
        "shap_waterfall.png",
        "shap_action_contrast.png",
        "shap_scenario_comparison.png",
    ]:
        path = os.path.join(shap_dir, plot_file)
        assert os.path.isfile(path), f"Missing SHAP plot: {path}"
        assert os.path.getsize(path) > 5000, f"SHAP plot {path} is too small (<5KB)"

    # SHAP Manifest & Background Metadata
    manifest_path = os.path.join(shap_dir, "shap_manifest.json")
    bg_meta_path = os.path.join(shap_dir, "background_metadata.json")
    assert os.path.isfile(manifest_path)
    assert os.path.isfile(bg_meta_path)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    # Verify machine-readable global feature importance records
    assert "global_feature_importance" in manifest_data
    assert "policy_level" in manifest_data["global_feature_importance"]
    assert "action_specific_action_22" in manifest_data["global_feature_importance"]
    for feat in OBS_NAMES:
        assert feat in manifest_data["global_feature_importance"]["policy_level"]
        assert feat in manifest_data["global_feature_importance"]["action_specific_action_22"]["importance"]

    # Verify machine-readable contrastive explanation records (Action 9 vs Action 5)
    assert "contrastive_explanation" in manifest_data
    contrast = manifest_data["contrastive_explanation"]
    assert contrast["chosen_action"] == 9
    assert contrast["competing_action"] == 5
    assert contrast["delta_q"] > 0.0
    assert contrast["efficiency_error"] < 1e-4
    for feat in OBS_NAMES:
        assert feat in contrast["delta_shap"]


# ---------------------------------------------------------------------------
# Test Group 6: CLI & Workflow Regression Tests
# ---------------------------------------------------------------------------

def test_shap_method_consistency(agent, background_data):
    """Verify that requested SHAP method is recorded and respected."""
    _, norm_bg, _ = background_data
    # 1. Deep requested
    deep_exp = DDQNShapExplainer(agent=agent, background_norm=norm_bg, method="deep")
    assert deep_exp.requested_method == "deep"
    assert deep_exp.explainer_used in ["deep", "kernel"]

    # 2. Kernel requested
    kernel_exp = DDQNShapExplainer(agent=agent, background_norm=norm_bg, method="kernel")
    assert kernel_exp.requested_method == "kernel"
    assert kernel_exp.explainer_used == "kernel"


def test_skip_shap_workflow(agent):
    """Verify that --skip-shap genuinely skips SHAP while still generating valid outputs."""
    env = NeonatalIncubatorEnv(scenario=1, max_steps=5)
    policy = AgentPolicy(agent, name="ddqn")

    # When explainer is None, no SHAP computations should occur
    rows, metrics = evaluate_trajectory(env, policy, seed=901000, explainer=None)
    assert len(rows) == 6
    assert metrics["episode_length"] == 5
    for r in rows[1:]:
        assert "decision_obs_Tc" in r
        assert "next_obs_Tc" in r
        assert "selected_action" in r
        # SHAP fields should not be populated
        assert "shap_base_val" not in r or r.get("shap_base_val") == "" or r.get("shap_base_val") is None

