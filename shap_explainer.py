#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shap_explainer.py -- SHAP Explainability Engine for Pre-Trained DDQN Agents
=============================================================================
Project   : Project 04 - Member 03 (Explainability, Validation & Integration)
Topic     : Neonatal Incubator Temperature and Oxygen Regulation (DDQN + SHAP)
Reference : Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"
            van Hasselt et al. (2016) "Deep Reinforcement Learning with Double Q-learning"

Audit Corrections & Improvements:
---------------------------------
1. Deterministic, stratified background sampling across the full 200-step simulation
   timeline (covering normal, ambient cooling, porthole opening, hypothermia,
   and oxygen desaturation) using both Baseline and DDQN policies.
2. Distinct Policy-Level vs Action-Specific feature importance calculation and plotting.
3. Genuine simulation states extracted from verified rollouts for all local explanations
   (waterfall, contrastive preference, and multi-scenario profiles).
4. Automatic fallback from DeepExplainer to KernelExplainer with comprehensive logging.
5. Numerical reconstruction checks for approximate feature attributions.

>>> SIMULATION-ONLY RESEARCH PROJECT - NOT FOR CLINICAL USE <<<
=============================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from pathlib import Path
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Enforce quiet TF startup and CPU execution by default
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_CUR_DIR, "src")
for p in (_CUR_DIR, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np

try:
    import shap
except ImportError as err:
    raise ImportError(
        "SHAP package is missing. Install with 'pip install shap' before running."
    ) from err

try:
    import tensorflow as tf
except ImportError as err:
    raise ImportError(
        "TensorFlow is missing. Install with 'pip install tensorflow' before running."
    ) from err

from baseline import RuleBasedBaselineController
from ddqn_agent import DDQNAgent, select_device
from environment import ACTION_TABLE, NeonatalIncubatorEnv, OBS_NAMES, SCENARIO_NAMES
from rl_utils import AgentPolicy, BaselinePolicy, test_seed, set_global_seeds


logger = logging.getLogger("shap_explainer")
# Coalition sampling logs one INFO message per internal operation. Keep warnings.
logging.getLogger("shap").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def load_trained_agent(checkpoint_path: str, device: str = "cpu") -> DDQNAgent:
    """Load pre-trained DDQNAgent checkpoint and verify Q-value forward pass."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")

    dev_str = select_device(device)
    agent = DDQNAgent.load(checkpoint_path, device=dev_str)

    env = NeonatalIncubatorEnv()
    if (agent.state_size, agent.action_size) != (len(OBS_NAMES), len(ACTION_TABLE)):
        raise ValueError("Checkpoint dimensions differ from the Member 03 observation/action contract")
    if agent.normalizer.names != list(OBS_NAMES) or env.action_table != ACTION_TABLE:
        raise ValueError("Observation ordering or action table mismatch")
    # Verification: Ensure weights are non-empty and forward pass is finite
    dummy_obs = np.array([37.0, 36.5, 34.0, 21.0, 92.0, 25.0, 0.0, 0.0, 0.5, 0.0, 0.0], dtype=np.float32)
    dummy_norm = agent.preprocess(dummy_obs)
    q_vals = agent.q_values(dummy_norm)

    if q_vals.shape != (1, agent.action_size):
        raise ValueError(
            f"Shape mismatch: expected (1, {agent.action_size}), got {q_vals.shape}"
        )
    if not np.all(np.isfinite(q_vals)):
        raise ValueError("Agent produced non-finite Q-values on reference state.")

    return agent


def collect_background_dataset(
    config_path: Optional[str] = None,
    seed: int = 42,
    agent: Optional[DDQNAgent] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Collect deterministic, stratified on-manifold observations across the full timeline.

    Samples representative states covering:
      - Scenario 1: Normal setpoint tracking across the 200 steps.
      - Scenario 2: Early, mid, and late ambient cooling (Tamb from 25 to 18 C).
      - Scenario 3: Pre-porthole, active porthole opening (hc=2.0, disturbance=1.0), and rewarming.
      - Scenario 4: Acute initial hypothermia (Tc=35.8 C), rewarming transient, and stabilization.
      - Scenario 5: Pre-desat, active peak desaturation (disturbance=2.0, step 90), and recovery.

    Uses both Baseline and DDQN policies over genuine simulation rollouts.

    Returns
    -------
    raw_states : np.ndarray, shape (100, 11)
    norm_states : np.ndarray, shape (100, 11)
    metadata : Dict[str, Any]
    """
    if agent is None:
        agent = load_trained_agent("models/ddqn_tuned_seed0/best.pt", device="cpu")

    stratified_steps = {
        1: [0, 25, 50, 75, 100, 125, 150, 195],                    # 8 steps
        2: [10, 30, 50, 75, 100, 125, 150, 170, 185, 198],        # 10 steps
        3: [15, 35, 48, 55, 65, 75, 79, 85, 100, 130, 180],        # 11 steps
        4: [0, 5, 12, 25, 40, 60, 85, 120, 160, 195],              # 10 steps
        5: [20, 40, 55, 65, 75, 90, 105, 118, 125, 150, 190],      # 11 steps
    }

    raw_states_list: List[np.ndarray] = []
    sample_records: List[Dict[str, Any]] = []

    for scn, step_targets in stratified_steps.items():
        step_set = set(step_targets)
        for pol_name in ["baseline", "ddqn"]:
            env = NeonatalIncubatorEnv(scenario=scn, config_path=config_path, seed=seed + scn * 10)
            policy = BaselinePolicy() if pol_name == "baseline" else AgentPolicy(agent)
            
            # Use deterministic seed for rollout
            ep_seed = seed + scn * 100 + (1 if pol_name == "ddqn" else 0)
            obs, _ = env.reset(seed=ep_seed)
            policy.reset(env)

            step = 0
            if 0 in step_set:
                raw_states_list.append(obs.copy())
                sample_records.append({
                    "scenario": scn,
                    "step": 0,
                    "policy": pol_name,
                    "disturbance": float(obs[9]),
                    "phase": float(obs[8]),
                    "Tamb": float(obs[5]),
                    "Tc": float(obs[0]),
                    "SpO2": float(obs[4]),
                })

            term = trunc = False
            while not (term or trunc):
                action = int(policy.act(obs))
                obs, _, term, trunc, _ = env.step(action)
                step += 1
                if step in step_set:
                    raw_states_list.append(obs.copy())
                    sample_records.append({
                        "scenario": scn,
                        "step": step,
                        "policy": pol_name,
                        "disturbance": float(obs[9]),
                        "phase": float(obs[8]),
                        "Tamb": float(obs[5]),
                        "Tc": float(obs[0]),
                        "SpO2": float(obs[4]),
                    })

    raw_states = np.array(raw_states_list, dtype=np.float32)
    norm_states = np.array([agent.preprocess(s) for s in raw_states], dtype=np.float32)

    # Compile sampling coverage metadata
    metadata = {
        "total_samples": int(len(raw_states)),
        "sampling_strategy": "deterministic_stratified_timeline",
        "records": sample_records,
        "random_seed": seed,
        "scenarios_covered": list(stratified_steps.keys()),
        "scenario_sample_counts": {
            SCENARIO_NAMES[scn]: sum(1 for r in sample_records if r["scenario"] == scn)
            for scn in stratified_steps
        },
        "disturbance_counts": {
            "idle_0": sum(1 for r in sample_records if r["disturbance"] == 0.0),
            "porthole_1": sum(1 for r in sample_records if r["disturbance"] == 1.0),
            "oxygen_desat_2": sum(1 for r in sample_records if r["disturbance"] == 2.0),
        },
        "feature_ranges": {
            OBS_NAMES[i]: {
                "min": float(np.min(raw_states[:, i])),
                "max": float(np.max(raw_states[:, i])),
                "mean": float(np.mean(raw_states[:, i])),
            }
            for i in range(len(OBS_NAMES))
        },
    }

    return raw_states, norm_states, metadata


class DDQNShapExplainer:
    """SHAP Explainer wrapper for DDQN Q-Networks with automatic fallback.

    Supports:
      - DeepExplainer (DeepLIFT backpropagation on piece-wise linear layers)
      - KernelExplainer (Model-agnostic weighted linear regression)
    """

    def __init__(
        self,
        agent: DDQNAgent,
        background_norm: np.ndarray,
        method: str = "deep",
    ) -> None:
        self.agent = agent
        self.requested_method = method.lower()
        self.background_norm = np.asarray(background_norm, dtype=np.float32)
        self.method = self.requested_method
        self.explainer_used = self.method
        self.explainer: Any = None
        self.expected_values: np.ndarray = np.zeros(self.agent.action_size, dtype=np.float32)

        if self.background_norm.ndim != 2 or self.background_norm.shape[1] != agent.state_size or not len(self.background_norm):
            raise ValueError("Expected a nonempty normalized background matrix")
        if not np.isfinite(self.background_norm).all() or self.method not in {"deep", "kernel"}:
            raise ValueError("Invalid background or SHAP method")
        self.fallback_reason = None
        self._kernel_heads = {}
        self._last_batch = None
        self._initialize_explainer()

    def _initialize_explainer(self):
        if self.requested_method == "deep":
            try:
                net = self.agent.online.net
                model = tf.keras.Model(inputs=net.inputs, outputs=net.outputs[0])
                self.explainer = shap.DeepExplainer(model, self.background_norm)
                self._check_expected_values()
                self.explain_norm_batch(self.background_norm[:1])
                return
            except Exception as exc:
                self.fallback_reason = f"{type(exc).__name__}: {exc}"
                logger.warning("Deep SHAP preflight failed. Using Kernel SHAP: %s", self.fallback_reason)
        self.method = self.explainer_used = "kernel"
        self._last_batch = None
        self.explainer = shap.KernelExplainer(self.agent.q_values, self.background_norm)
        self._check_expected_values()

    def _check_expected_values(self):
        self.expected_values = np.asarray(self.explainer.expected_value, dtype=np.float32)
        reference = self.agent.q_values(self.background_norm).mean(axis=0)
        if self.expected_values.shape != (self.agent.action_size,) or not np.isfinite(self.expected_values).all():
            raise ValueError("Invalid expected Q-values")
        np.testing.assert_allclose(self.expected_values, reference, atol=1e-5, rtol=1e-5)

    @staticmethod
    def _kernel_values(engine, x, nsamples):
        # KernelExplainer uses NumPy's legacy RNG for coalition draws. Tie the
        # draws to the decision input so saving a plot cannot change its data.
        state = np.random.get_state()
        try:
            np.random.seed(zlib.crc32(x.tobytes()))
            return engine.shap_values(x, nsamples=nsamples, l1_reg=0, silent=True)
        finally:
            np.random.set_state(state)

    def explain_norm_batch(self, norm_batch, nsamples=100):
        """Return (observations, features, actions), checking additive reconstruction."""
        x = np.atleast_2d(np.asarray(norm_batch, dtype=np.float32))
        if x.ndim != 2 or x.shape[1] != self.agent.state_size or not np.isfinite(x).all():
            raise ValueError("Expected finite normalized observations in checkpoint feature order")
        key = (nsamples, x.shape, x.tobytes())
        if self._last_batch is not None and self._last_batch[0] == key:
            return self._last_batch[1]
        if self.method == "deep":
            raw = self.explainer.shap_values(x)
        else:
            raw = np.concatenate([self._kernel_values(self.explainer, row[None], nsamples) for row in x])
        values = np.asarray(np.stack(raw, axis=-1) if isinstance(raw, list) else raw, dtype=np.float32)
        if values.shape != (len(x), self.agent.state_size, self.agent.action_size):
            raise ValueError(f"Unexpected SHAP axes: {values.shape}")
        np.testing.assert_allclose(values.sum(axis=1) + self.expected_values,
                                   self.agent.q_values(x), atol=1e-4, rtol=0)
        self._last_batch = (key, values)
        return values

    def decision_values(self, norm, action, q):
        """Explain just the executed head; check the head index and its reconstruction."""
        x = norm[None]
        if self.method == "deep":
            # SHAP's built-in ranked-output check indexes unranked heads. Apply
            # the same efficiency test below against the actual selected Q.
            values, ranks = self.explainer.shap_values(x, ranked_outputs=1, check_additivity=False)
            if int(ranks[0, 0]) == action:
                phi = np.asarray(values[0, :, 0], dtype=np.float32)
            else:
                phi = self.explain_norm_batch(x)[0, :, action]
        else:
            if action not in self._kernel_heads:
                self._kernel_heads[action] = shap.KernelExplainer(
                    lambda batch: self.agent.q_values(batch)[:, action], self.background_norm)
            phi = np.asarray(self._kernel_values(self._kernel_heads[action], x, 100)[0], dtype=np.float32)
        base = float(self.expected_values[action])
        residual = abs(float(q) - (base + float(phi.sum())))
        if phi.shape != (self.agent.state_size,) or not np.isfinite(residual) or residual > 1e-4:
            raise ValueError(f"Invalid attributions for action {action}: residual {residual}")
        return base, phi, residual

    def explain_state(
        self, raw_obs: np.ndarray, action_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """Explain an individual state observation."""
        raw_obs = np.asarray(raw_obs, dtype=np.float32).reshape(-1)
        norm_obs = self.agent.preprocess(raw_obs)
        q_vals = self.agent.q_values(norm_obs)[0]
        greedy_act = int(np.argmax(q_vals))

        target_act = greedy_act if action_idx is None else int(action_idx)
        if not 0 <= target_act < self.agent.action_size:
            raise ValueError(f"Invalid action index {target_act}")
        sv = self.explain_norm_batch(norm_obs[np.newaxis, :])  # (1, 11, 25)
        act_sv = sv[0, :, target_act]
        base_val = float(self.expected_values[target_act])

        action_meta = ACTION_TABLE[target_act]

        return {
            "raw_obs": raw_obs,
            "norm_obs": norm_obs,
            "q_values": q_vals,
            "greedy_action": greedy_act,
            "explained_action": target_act,
            "action_meta": action_meta,
            "base_value": base_val,
            "shap_values": act_sv,
            "all_shap_values": sv[0],  # (11, 25)
            "efficiency_error": float(abs(q_vals[target_act] - (base_val + np.sum(act_sv)))),
        }

    def explain_action_difference(
        self, raw_obs: np.ndarray, action_a: int, action_b: int
    ) -> Dict[str, Any]:
        """Contrastive explanation: Why action_a over action_b."""
        exp_a = self.explain_state(raw_obs, action_idx=action_a)
        exp_b = self.explain_state(raw_obs, action_idx=action_b)

        delta_q = float(exp_a["q_values"][action_a] - exp_a["q_values"][action_b])
        delta_base = float(exp_a["base_value"] - exp_b["base_value"])
        delta_shap = exp_a["shap_values"] - exp_b["shap_values"]

        return {
            "raw_obs": raw_obs,
            "action_a": action_a,
            "action_b": action_b,
            "action_a_meta": ACTION_TABLE[action_a],
            "action_b_meta": ACTION_TABLE[action_b],
            "delta_q": delta_q,
            "delta_base": delta_base,
            "delta_shap": delta_shap,
            "efficiency_error": float(abs(delta_q - (delta_base + np.sum(delta_shap)))),
        }

    # ----------------------------------------------------------------------- #
    # Feature Importance Calculations (Distinguishing Action vs Policy)        #
    # ----------------------------------------------------------------------- #
    def compute_action_feature_importance(
        self, norm_states: np.ndarray, action_idx: int
    ) -> np.ndarray:
        """Action-specific feature importance: Mean |SHAP| for fixed action head."""
        sv = self.explain_norm_batch(norm_states)  # (N, 11, 25)
        return np.mean(np.abs(sv[:, :, action_idx]), axis=0)

    def compute_policy_feature_importance(
        self, norm_states: np.ndarray
    ) -> np.ndarray:
        """Policy-level feature importance: Mean |SHAP| for each state's greedy action."""
        sv = self.explain_norm_batch(norm_states)  # (N, 11, 25)
        greedy_actions = [
            int(np.argmax(self.agent.q_values(s[np.newaxis, :])[0])) for s in norm_states
        ]
        policy_sv = np.array([sv[i, :, greedy_actions[i]] for i in range(len(norm_states))])
        return np.mean(np.abs(policy_sv), axis=0)

    # ----------------------------------------------------------------------- #
    # Visualization Generators                                                #
    # ----------------------------------------------------------------------- #
    def plot_summary_beeswarm(
        self,
        norm_states: np.ndarray,
        action_idx: int,
        output_path: str,
        raw_states: Optional[np.ndarray] = None,
        title: Optional[str] = None,
    ) -> None:
        """Generate SHAP summary/beeswarm plot for a designated action."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        sv = self.explain_norm_batch(norm_states)
        act_sv = sv[:, :, action_idx]
        base_val = float(self.expected_values[action_idx])
        base_arr = np.full(len(norm_states), base_val, dtype=np.float32)

        display_data = raw_states if raw_states is not None else norm_states

        exp = shap.Explanation(
            values=act_sv,
            base_values=base_arr,
            data=display_data,
            feature_names=list(OBS_NAMES),
        )

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        act_meta = ACTION_TABLE[action_idx]
        default_title = (
            f"SHAP Summary (Beeswarm) -- Action {action_idx}: "
            f"Heater={act_meta['heater_level']:.2f} ({act_meta['Q_heater_W']:.0f}W), "
            f"FiO2={act_meta['fio2_command']:.2f}"
        )
        plt.title(title or default_title, fontsize=12, pad=12, fontweight="bold")
        shap.plots.beeswarm(exp, show=False, max_display=11)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close("all")

    def plot_feature_importance_bar(
        self,
        norm_states: np.ndarray,
        action_idx: int,
        output_path: str,
        title: Optional[str] = None,
    ) -> None:
        """Generate mean absolute SHAP value feature-importance bar plot for a fixed action."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        mean_abs_shap = self.compute_action_feature_importance(norm_states, action_idx)

        sorted_idx = np.argsort(mean_abs_shap)
        sorted_features = [OBS_NAMES[i] for i in sorted_idx]
        sorted_vals = mean_abs_shap[sorted_idx]

        fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=300)
        bars = ax.barh(sorted_features, sorted_vals, color="#2b5c8f", edgecolor="#142c44", height=0.65)
        ax.set_xlabel("Mean Absolute SHAP Value (Impact on Q-value magnitude)", fontsize=11, fontweight="bold")
        ax.set_ylabel("State Observation Features", fontsize=11, fontweight="bold")

        act_meta = ACTION_TABLE[action_idx]
        default_title = (
            f"Action-Specific Feature Importance -- Action {action_idx}\n"
            f"[Heater: {act_meta['heater_level']:.2f} ({act_meta['Q_heater_W']:.0f}W), "
            f"FiO2: {act_meta['fio2_command']:.2f}]"
        )
        ax.set_title(title or default_title, fontsize=11, fontweight="bold", pad=12)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        ax.margins(x=0.18)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.003 * max(sorted_vals), bar.get_y() + bar.get_height() / 2,
                    f"{w:.4f}", va="center", ha="left", fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_policy_feature_importance_bar(
        self,
        norm_states: np.ndarray,
        output_path: str,
        title: Optional[str] = None,
    ) -> None:
        """Generate Policy-Level feature importance (aggregating across each state's greedy action)."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        mean_abs_shap = self.compute_policy_feature_importance(norm_states)

        sorted_idx = np.argsort(mean_abs_shap)
        sorted_features = [OBS_NAMES[i] for i in sorted_idx]
        sorted_vals = mean_abs_shap[sorted_idx]

        fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=300)
        bars = ax.barh(sorted_features, sorted_vals, color="#1f77b4", edgecolor="#0b3c66", height=0.65)
        ax.set_xlabel("Policy-Level Mean Absolute SHAP Value (Evaluated on Greedy Actions)", fontsize=11, fontweight="bold")
        ax.set_ylabel("State Observation Features", fontsize=11, fontweight="bold")

        default_title = (
            "Policy-Level Global Feature Importance\n"
            "(Aggregated across each observation's greedy action a* = argmax Q)"
        )
        ax.set_title(title or default_title, fontsize=11, fontweight="bold", pad=12)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        ax.margins(x=0.18)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.003 * max(sorted_vals), bar.get_y() + bar.get_height() / 2,
                    f"{w:.4f}", va="center", ha="left", fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_waterfall(
        self,
        raw_obs: np.ndarray,
        action_idx: Optional[int],
        output_path: str,
        title: Optional[str] = None,
    ) -> None:
        """Generate single decision SHAP waterfall plot on a genuine simulation state."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        exp_dict = self.explain_state(raw_obs, action_idx=action_idx)
        act_idx = exp_dict["explained_action"]
        act_meta = exp_dict["action_meta"]

        exp = shap.Explanation(
            values=exp_dict["shap_values"],
            base_values=exp_dict["base_value"],
            data=exp_dict["raw_obs"],
            feature_names=list(OBS_NAMES),
        )

        fig = plt.figure(figsize=(9.5, 6), dpi=300)
        shap.plots.waterfall(exp, show=False, max_display=11)
        default_title = (
            f"SHAP Decision Waterfall -- Real Simulation State: Action {act_idx}\n"
            f"[Q={exp_dict['q_values'][act_idx]:.3f}, E[Q]={exp_dict['base_value']:.3f} | "
            f"Heater: {act_meta['heater_level']:.2f} ({act_meta['Q_heater_W']:.0f}W), "
            f"FiO2: {act_meta['fio2_command']:.2f}]"
        )
        plt.gcf().suptitle(title or default_title, fontsize=11, fontweight="bold", y=1.06)
        plt.subplots_adjust(top=0.88)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close("all")

    def plot_action_contrast(
        self,
        raw_obs: np.ndarray,
        action_winner: int,
        action_loser: int,
        output_path: str,
        title: Optional[str] = None,
    ) -> None:
        """Generate contrastive attribution bar chart: Why action_winner > action_loser."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        contrast = self.explain_action_difference(raw_obs, action_winner, action_loser)

        delta_shap = contrast["delta_shap"]
        features = list(OBS_NAMES)
        sorted_idx = np.argsort(delta_shap)
        sorted_features = [features[i] for i in sorted_idx]
        sorted_deltas = delta_shap[sorted_idx]

        colors = ["#2ca02c" if val >= 0 else "#d62728" for val in sorted_deltas]

        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
        bars = ax.barh(sorted_features, sorted_deltas, color=colors, edgecolor="#222222", height=0.6)
        ax.axvline(0, color="#444444", linewidth=1.0, linestyle="--")

        ax.set_xlabel(
            f"Delta SHAP Value [ phi_j(Action {action_winner}) - phi_j(Action {action_loser}) ]\n"
            f"(Green: Feature favors chosen Action {action_winner}; Red: Feature favors competing Action {action_loser})",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_ylabel("State Observation Features", fontsize=11, fontweight="bold")

        w_meta = contrast["action_a_meta"]
        l_meta = contrast["action_b_meta"]
        default_title = (
            f"Contrastive Decision Attribution: Action {action_winner} vs Action {action_loser}\n"
            f"Chosen: [H={w_meta['heater_level']:.2f}, FiO2={w_meta['fio2_command']:.2f}] vs "
            f"Comp: [H={l_meta['heater_level']:.2f}, FiO2={l_meta['fio2_command']:.2f}] | "
            f"Delta Q = {contrast['delta_q']:+.4f}"
        )
        ax.set_title(title or default_title, fontsize=11, fontweight="bold", pad=12)
        ax.grid(axis="x", linestyle=":", alpha=0.6)

        ax.margins(x=0.18)
        for bar in bars:
            w = bar.get_width()
            offset = 0.002 * (max(sorted_deltas) - min(sorted_deltas))
            if w >= 0:
                ax.text(w + offset, bar.get_y() + bar.get_height() / 2, f"{w:+.4f}",
                        va="center", ha="left", fontsize=9)
            else:
                ax.text(w - offset, bar.get_y() + bar.get_height() / 2, f"{w:+.4f}",
                        va="center", ha="right", fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_scenario_feature_comparison(
        self,
        scenario_explanations: Dict[str, Dict[str, Any]],
        output_path: str,
        title: Optional[str] = None,
    ) -> None:
        """Plot comparative feature contributions across real simulation states."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        scenario_names = list(scenario_explanations.keys())
        features = list(OBS_NAMES)
        n_scenarios = len(scenario_names)
        n_features = len(features)

        fig, ax = plt.subplots(figsize=(13, 6.5), dpi=300)
        x = np.arange(n_features)
        bar_width = 0.8 / n_scenarios
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        for idx, scn_name in enumerate(scenario_names):
            exp = scenario_explanations[scn_name]
            sv = exp["shap_values"]
            offsets = x - 0.4 + (idx + 0.5) * bar_width
            ax.bar(
                offsets,
                sv,
                width=bar_width,
                label=f"{scn_name} (Act {exp['explained_action']})",
                color=palette[idx % len(palette)],
                edgecolor="#333333",
            )

        ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=35, ha="right", fontsize=10, fontweight="bold")
        ax.set_ylabel("SHAP Attribution phi_j to Chosen Q(s, a*)", fontsize=11, fontweight="bold")
        ax.set_title(
            title or "SHAP Attribution Profiles Across Four Simulated Decisions",
            fontsize=12,
            fontweight="bold",
            pad=14,
        )
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False, fontsize=9.5)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def extract_real_simulation_states(
    agent: DDQNAgent,
) -> Dict[str, Dict[str, Any]]:
    """Extract genuine simulation states from actual test episode rollouts."""
    targets = [
        (1, 50, "Normal Condition (Scn 1, Step 50)"),
        (2, 150, "Ambient Cooling Ramp (Scn 2, Step 150)"),
        (3, 65, "Porthole Open Event (Scn 3, Step 65)"),
        (5, 90, "Oxygen Desaturation Peak (Scn 5, Step 90)"),
    ]

    extracted: Dict[str, Dict[str, Any]] = {}
    for scn, step_idx, label in targets:
        env = NeonatalIncubatorEnv(scenario=scn)
        pol = AgentPolicy(agent)
        obs, _ = env.reset(seed=test_seed(scn, 0))
        pol.reset(env)

        for s in range(step_idx):
            a = pol.act(obs)
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                raise ValueError(f"Scenario {scn} ended before decision {step_idx}")

        norm_s = agent.preprocess(obs)
        q_vals = agent.q_values(norm_s)[0]
        chosen_a = int(np.argmax(q_vals))

        extracted[label] = {
            "scenario": scn,
            "step": step_idx,
            "raw_obs": obs.copy(),
            "norm_obs": norm_s.copy(),
            "chosen_action": chosen_a,
            "action_meta": ACTION_TABLE[chosen_a],
            "q_values": q_vals.copy(),
        }

    return extracted


def prepare_explanation(checkpoint, device, seed, method):
    agent = load_trained_agent(checkpoint, device)
    raw, norm, metadata = collect_background_dataset(seed=seed, agent=agent)
    return agent, raw, norm, metadata, DDQNShapExplainer(agent, norm, method)


def run_shap_analysis(
    checkpoint_path: str = "models/ddqn_tuned_seed0/best.pt",
    output_dir: str = "outputs/member03/shap",
    method: str = "deep",
    device: str = "cpu",
    seed: int = 42,
    context=None,
) -> Dict[str, Any]:
    """Execute complete SHAP explainability workflow and export all artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    set_global_seeds(seed)
    agent, raw_bg, norm_bg, bg_meta, explainer = context or prepare_explanation(
        checkpoint_path, device, seed, method)
    logger.info(
        f"Background dataset collected: {norm_bg.shape} ({len(raw_bg)} states). "
        f"Disturbance counts: {bg_meta['disturbance_counts']}"
    )

    # Save background metadata
    bg_meta_path = os.path.join(output_dir, "background_metadata.json")
    with open(bg_meta_path, "w", encoding="utf-8") as fh:
        json.dump(bg_meta, fh, indent=2)

    logger.info("Plotting with %s SHAP (requested %s)", explainer.method, method)

    # Extract genuine simulation states for all local explanations
    logger.info("Extracting genuine simulation states from actual episode rollouts...")
    real_states = extract_real_simulation_states(agent)

    # 1. Beeswarm summary plot for default greedy action on nominal conditions
    nominal_data = real_states["Normal Condition (Scn 1, Step 50)"]
    chosen_act = nominal_data["chosen_action"]

    global_sv = explainer.explain_norm_batch(norm_bg)
    greedy = [int(np.argmax(agent.q_values(row)[0])) for row in norm_bg]
    summary_png = os.path.join(output_dir, "shap_summary.png")
    logger.info(f"Generating summary beeswarm plot -> {summary_png}")
    explainer.plot_summary_beeswarm(
        norm_states=norm_bg,
        action_idx=chosen_act,
        output_path=summary_png,
        raw_states=raw_bg,
        title=f"SHAP on 100 sampled states -- nominal decision head {chosen_act} (Heater={nominal_data['action_meta']['heater_level']}, FiO2={nominal_data['action_meta']['fio2_command']})",
    )

    # 2. Action-specific feature importance bar plot (Action 22)
    feat_imp_png = os.path.join(output_dir, "shap_feature_importance.png")
    logger.info(f"Generating action-specific feature importance bar plot -> {feat_imp_png}")
    action_22_importance = explainer.compute_action_feature_importance(norm_bg, 22)
    explainer.plot_feature_importance_bar(
        norm_states=norm_bg,
        action_idx=22,
        output_path=feat_imp_png,
        title=None,
    )

    # 2b. Policy-Level feature importance bar plot (aggregating greedy actions across background states)
    policy_feat_imp_png = os.path.join(output_dir, "shap_policy_feature_importance.png")
    logger.info(f"Generating policy-level feature importance bar plot -> {policy_feat_imp_png}")
    policy_importance = explainer.compute_policy_feature_importance(norm_bg)
    explainer.plot_policy_feature_importance_bar(
        norm_states=norm_bg,
        output_path=policy_feat_imp_png,
    )

    # 3. Individual decision waterfall plot (Real Ambient Cooling state: Scn 2, Step 150)
    ambient_data = real_states["Ambient Cooling Ramp (Scn 2, Step 150)"]
    waterfall_png = os.path.join(output_dir, "shap_waterfall.png")
    logger.info(f"Generating individual decision waterfall plot -> {waterfall_png}")
    explainer.plot_waterfall(
        raw_obs=ambient_data["raw_obs"],
        action_idx=ambient_data["chosen_action"],
        output_path=waterfall_png,
        title=(
            f"SHAP Waterfall -- Real Ambient Cooling State (Scn 2, Step 150)\n"
            f"[Tamb={ambient_data['raw_obs'][5]:.2f}C, Tc={ambient_data['raw_obs'][0]:.2f}C, phase={ambient_data['raw_obs'][8]:.2f} | "
            f"Chosen Act {ambient_data['chosen_action']}: Heater={ambient_data['action_meta']['heater_level']}, FiO2={ambient_data['action_meta']['fio2_command']}]"
        ),
    )

    # 4. Contrastive action preference plot (Real Desaturation Peak state: Scn 5, Step 90)
    # At desat peak: Chosen Action 9 (FiO2=0.60, max oxygen) vs Action 5 (FiO2=0.21, room air)
    o2_data = real_states["Oxygen Desaturation Peak (Scn 5, Step 90)"]
    act_winner = o2_data["chosen_action"]  # Action 9
    act_comp = 5                           # Action 5 (Heater 0.25, FiO2 0.21)

    contrast_png = os.path.join(output_dir, "shap_action_contrast.png")
    logger.info(f"Generating contrastive action plot (Act {act_winner} vs Act {act_comp}) -> {contrast_png}")
    contrast_res = explainer.explain_action_difference(o2_data["raw_obs"], act_winner, act_comp)
    explainer.plot_action_contrast(
        raw_obs=o2_data["raw_obs"],
        action_winner=act_winner,
        action_loser=act_comp,
        output_path=contrast_png,
        title=(
            f"Contrastive Attribution at Oxygen Desaturation Peak (Scn 5, Step 90)\n"
            f"Chosen Act {act_winner} (FiO2={ACTION_TABLE[act_winner]['fio2_command']:.2f}) vs comparator {act_comp} (FiO2=0.21, Room Air) | "
            f"Delta Q = {contrast_res['delta_q']:+.4f}"
        ),
    )

    # 5. Multi-scenario comparative attribution plot across 4 genuine simulation states
    scenarios_dict = {
        label: explainer.explain_state(data["raw_obs"], action_idx=data["chosen_action"])
        for label, data in real_states.items()
    }

    scenario_comp_png = os.path.join(output_dir, "shap_scenario_comparison.png")
    logger.info(f"Generating multi-scenario feature attribution profile -> {scenario_comp_png}")
    explainer.plot_scenario_feature_comparison(
        scenario_explanations=scenarios_dict,
        output_path=scenario_comp_png,
    )

    # Build comprehensive manifest storing all local explanations and numerical metadata
    manifest_explanations = {}
    for label, data in real_states.items():
        exp = scenarios_dict[label]
        manifest_explanations[label] = {
            "scenario": data["scenario"],
            "step": data["step"],
            "raw_obs": {OBS_NAMES[i]: float(data["raw_obs"][i]) for i in range(len(OBS_NAMES))},
            "norm_obs": [float(x) for x in data["norm_obs"]],
            "chosen_action": data["chosen_action"],
            "action_meta": data["action_meta"],
            "all_q_values": [float(q) for q in data["q_values"]],
            "expected_value": float(exp["base_value"]),
            "shap_values": {OBS_NAMES[i]: float(exp["shap_values"][i]) for i in range(len(OBS_NAMES))},
            "efficiency_error": float(exp["efficiency_error"]),
        }

    manifest = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
        "feature_names": list(OBS_NAMES),
        "fallback_reason": explainer.fallback_reason,
        "expected_values": explainer.expected_values.tolist(),
        "global_plot_data": {"array_file": "global_attributions.npz", "summary_action": int(chosen_act),
                             "shape": list(global_sv.shape), "axes": ["state", "feature", "action"]},
        "explainer_requested": method,
        "explainer_used": explainer.explainer_used,
        "device": device,
        "n_background_samples": len(raw_bg),
        "background_metadata": bg_meta,
        "generated_plots": [
            summary_png,
            feat_imp_png,
            policy_feat_imp_png,
            waterfall_png,
            contrast_png,
            scenario_comp_png,
        ],
        "global_feature_importance": {
            "policy_level": {
                OBS_NAMES[i]: float(policy_importance[i]) for i in range(len(OBS_NAMES))
            },
            "action_specific_action_22": {
                "action_idx": 22,
                "action_meta": ACTION_TABLE[22],
                "importance": {
                    OBS_NAMES[i]: float(action_22_importance[i]) for i in range(len(OBS_NAMES))
                },
            },
        },
        "contrastive_explanation": {
            "scenario": int(o2_data["scenario"]),
            "step": int(o2_data["step"]),
            "context": "Oxygen Desaturation Peak (Scn 5, Step 90)",
            "chosen_action": int(act_winner),
            "competing_action": int(act_comp),
            "chosen_action_meta": contrast_res["action_a_meta"],
            "competing_action_meta": contrast_res["action_b_meta"],
            "q_chosen": float(o2_data["q_values"][act_winner]),
            "q_competing": float(o2_data["q_values"][act_comp]),
            "delta_q": float(contrast_res["delta_q"]),
            "delta_base": float(contrast_res["delta_base"]),
            "delta_shap": {
                OBS_NAMES[i]: float(contrast_res["delta_shap"][i]) for i in range(len(OBS_NAMES))
            },
            "efficiency_error": float(contrast_res["efficiency_error"]),
        },
        "local_explanations": manifest_explanations,
    }

    np.savez_compressed(os.path.join(output_dir, "global_attributions.npz"), raw=raw_bg,
                        normalized=norm_bg, phi=global_sv, greedy_actions=greedy,
                        q=agent.q_values(norm_bg))
    manifest_path = os.path.join(output_dir, "shap_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    logger.info(f"SHAP analysis successfully completed. Manifest stored at '{manifest_path}'.")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SHAP Explainability Engine for Neonatal Incubator DDQN Agent"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/ddqn_tuned_seed0/best.pt",
        help="Path to trained DDQN checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/member03/shap",
        help="Output directory for SHAP visualizations and manifest",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["deep", "kernel"],
        default="deep",
        help="SHAP explainer engine (deep=DeepExplainer, kernel=KernelExplainer)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load model on ('cpu' or 'gpu')",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for background state collection",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_shap_analysis(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        method=args.method,
        device=args.device,
        seed=args.seed,
    )

