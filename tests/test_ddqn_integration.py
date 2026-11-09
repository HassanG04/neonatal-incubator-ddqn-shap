#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_ddqn_integration.py -- End-to-end integration tests with the actual
NeonatalIncubatorEnv across all 5 scenarios (Gate B & Stage 7).

Verifies:
  1. All 5 scenarios and all 25 actions interact cleanly with DDQNAgent and
     ObservationNormalizer.
  2. Complete train -> save -> reload -> evaluate cycle succeeds without API
     errors and produces episodes.csv, updates.csv, validation.csv,
     manifest.json, best.pt, and final.pt.
  3. Custom action-grid (3x2 = 6 actions) environment dynamically sets
     state_size=11 and action_size=6 in DDQNAgent without hard-coded sizes.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ddqn_agent import DDQNAgent  # noqa: E402
from environment import NeonatalIncubatorEnv  # noqa: E402
from q_network import ObservationNormalizer  # noqa: E402
from rl_utils import AgentPolicy, BaselinePolicy, run_episode, test_seed as make_test_seed  # noqa: E402
from train_ddqn import parse_args, train  # noqa: E402


def test_all_scenarios_and_actions_with_agent():
    """Step through all 5 scenarios and exercise every valid action 0..24."""
    probe = NeonatalIncubatorEnv(scenario=1, seed=42)
    norm = ObservationNormalizer.from_env(probe)
    agent = DDQNAgent(
        state_size=probe.observation_space.shape[0],
        action_size=probe.action_space.n,
        normalizer=norm,
        device="cpu",
        seed=42,
    )
    probe.close()

    for scn in (1, 2, 3, 4, 5):
        env = NeonatalIncubatorEnv(scenario=scn, seed=42 + scn)
        obs, _ = env.reset(seed=42 + scn)
        for a in range(env.action_space.n):
            s = agent.preprocess(obs)
            assert s.shape == (11,) and np.all(np.isfinite(s))
            obs, r, term, trunc, _ = env.step(a)
            assert np.isfinite(r)
            if term or trunc:
                obs, _ = env.reset(seed=100 + scn + a)
        env.close()


def test_train_save_reload_evaluate_cycle():
    """Gate B acceptance: complete train -> save -> reload -> evaluate cycle."""
    with tempfile.TemporaryDirectory() as tmp:
        models_dir = os.path.join(tmp, "models")
        results_dir = os.path.join(tmp, "results")
        args = parse_args([
            "--smoke",
            "--seed", "0",
            "--algo", "ddqn",
            "--device", "cpu",
            "--run-name", "integration_smoke",
            "--models-dir", models_dir,
            "--results-dir", results_dir,
            "--set", "total_steps=600", "warmup_steps=100", "eval_interval=300", "log_interval=50",
            "--quiet",
        ])
        res = train(args)
        assert res["env_steps"] == 600
        assert res["grad_steps"] > 0

        best_pt = os.path.join(models_dir, "integration_smoke", "best.pt")
        final_pt = os.path.join(models_dir, "integration_smoke", "final.pt")
        assert os.path.isfile(best_pt)
        assert os.path.isfile(final_pt)
        for fname in ("episodes.csv", "updates.csv", "validation.csv", "manifest.json"):
            assert os.path.isfile(os.path.join(results_dir, "integration_smoke", fname))

        ep_df = pd.read_csv(os.path.join(results_dir, "integration_smoke", "episodes.csv"))
        assert set(ep_df["scenario"].unique()).issubset({1, 2, 3, 4, 5})
        assert len(ep_df) >= 3

        # Reload checkpoint and evaluate greedily on all 5 scenarios alongside baseline
        reloaded = DDQNAgent.load(best_pt, device="cpu", seed=0)
        ddqn_policy = AgentPolicy(reloaded, name="ddqn")
        base_policy = BaselinePolicy(name="baseline")
        for scn in (1, 2, 3, 4, 5):
            env = NeonatalIncubatorEnv(scenario=scn)
            out_ddqn = run_episode(env, ddqn_policy, seed=make_test_seed(scn, 0), record_trajectory=True)
            out_base = run_episode(env, base_policy, seed=make_test_seed(scn, 0), record_trajectory=False)
            assert np.isfinite(out_ddqn["metrics"]["total_reward"])
            assert np.isfinite(out_base["metrics"]["total_reward"])
            assert len(out_ddqn["trajectory"]) == out_ddqn["metrics"]["episode_length"] + 1
            env.close()


def test_dynamic_custom_grid_agent():
    """Verify DDQNAgent dynamically infers action_size=6 on a 3x2 custom grid."""
    custom_cfg = {
        "environment": {
            "heater_levels": [0.0, 0.5, 1.0],
            "fio2_levels": [0.21, 0.50],
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
        yaml.dump(custom_cfg, fh)
        cfg_path = fh.name
    try:
        env = NeonatalIncubatorEnv(scenario=1, seed=0, config_path=cfg_path)
        norm = ObservationNormalizer.from_env(env)
        agent = DDQNAgent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            normalizer=norm,
            device="cpu",
            seed=0,
        )
        assert agent.state_size == 11
        assert agent.action_size == 6
        obs, _ = env.reset(seed=0)
        a = agent.greedy_action(agent.preprocess(obs))
        assert 0 <= a < 6
        env.close()
    finally:
        os.unlink(cfg_path)

