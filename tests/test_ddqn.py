#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_ddqn.py -- Unit tests for QNetwork, ObservationNormalizer, ReplayBuffer,
and DDQNAgent (Stage 7 requirements, TensorFlow / Keras backend).

Covers:
  1. Q-network output shape, dynamic state/action sizes, linear (unbounded)
     output layer, and valid actions in 0..action_size-1.
  2. ObservationNormalizer determinism, bounds, clipping, and state_dict
     round-trip serialization.
  3. ReplayBuffer capacity, circular overwrite, sample shapes/dtypes,
     separate terminated vs truncated storage, scenario counts, and small/
     empty buffer handling.
  4. Double DQN online-selection / target-evaluation formula vs standard DQN
     on hand-constructed Q-value matrices where argmax(online) != argmax(target).
  5. Bootstrap mask: terminated=True zeros the bootstrap (y = r), whereas
     truncated=True with terminated=False keeps the bootstrap.
  6. Target network synchronization (hard and soft), non-trainable target
     parameters, finite gradient clipping, optimizer updates, non-finite loss
     guard, checkpoint save/load round-tripping, and deterministic greedy
     inference.
"""
from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ddqn_agent import (  # noqa: E402
    AgentConfig,
    DDQNAgent,
    LinearEpsilonSchedule,
    double_dqn_targets,
    dqn_targets,
)
from environment import NeonatalIncubatorEnv  # noqa: E402
from q_network import ObservationNormalizer, QNetwork  # noqa: E402
from replay_buffer import ReplayBuffer  # noqa: E402


# ---------------------------------------------------------------------------
# 1. QNetwork & ObservationNormalizer
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("state_size,action_size", [(11, 25), (11, 6), (8, 12)])
def test_q_network_shapes_and_linear_output(state_size: int, action_size: int):
    net = QNetwork(state_size=state_size, action_size=action_size, hidden_sizes=(128, 128), seed=0)
    x = tf.random.normal((7, state_size), dtype=tf.float32)
    out = net(x)
    assert tuple(out.shape) == (7, action_size)
    assert out.dtype == tf.float32
    # Verify last layer is Dense with linear activation (not softmax/sigmoid)
    # and can produce arbitrary negative and positive real values (> 1 and < -1).
    last_layer = net.net.layers[-1]
    assert isinstance(last_layer, tf.keras.layers.Dense)
    assert last_layer.activation == tf.keras.activations.linear
    w_shape = last_layer.get_weights()[0].shape
    last_layer.set_weights([
        np.zeros(w_shape, dtype=np.float32),
        np.linspace(-50.0, 50.0, action_size, dtype=np.float32),
    ])
    y = net(tf.ones((1, state_size), dtype=tf.float32)).numpy()
    assert float(y.min()) == pytest.approx(-50.0, abs=1e-5)
    assert float(y.max()) == pytest.approx(50.0, abs=1e-5)


def test_q_network_invalid_dims():
    with pytest.raises(ValueError):
        QNetwork(0, 25)
    with pytest.raises(ValueError):
        QNetwork(11, 0)


def test_observation_normalizer_round_trip_and_bounds():
    env = NeonatalIncubatorEnv(scenario=1, seed=42)
    norm = ObservationNormalizer.from_env(env, clip_value=5.0)
    assert norm.state_size == 11
    obs, _ = env.reset(seed=42)
    z = norm(obs)
    assert z.shape == (11,)
    assert z.dtype == np.float32
    assert np.all(np.isfinite(z))
    assert np.all(np.abs(z) <= 5.0)

    # Extreme values are clipped to [-clip_value, +clip_value]
    extreme = np.full(11, 1e6, dtype=np.float32)
    z_ext = norm(extreme)
    np.testing.assert_allclose(z_ext, np.full(11, 5.0, dtype=np.float32))

    # Serialization round-trip
    sd = norm.state_dict()
    norm2 = ObservationNormalizer.from_state_dict(sd)
    np.testing.assert_allclose(norm(obs), norm2(obs), atol=1e-7)

    with pytest.raises(ValueError):
        ObservationNormalizer([0.0, 0.0], [1.0, 0.0])  # zero scale rejected
    env.close()


# ---------------------------------------------------------------------------
# 2. ReplayBuffer
# ---------------------------------------------------------------------------
def test_replay_buffer_empty_small_and_capacity_overwrite():
    buf = ReplayBuffer(capacity=5, state_size=11, seed=0)
    assert len(buf) == 0
    assert not buf.can_sample(1)
    with pytest.raises(ValueError):
        buf.sample(1)
    with pytest.raises(ValueError):
        buf.sample(0)

    # Add 3 transitions -> can sample <= 3, cannot sample 4
    for i in range(3):
        s = np.full(11, float(i), dtype=np.float32)
        buf.add(s, action=i, reward=-float(i), next_state=s + 1.0,
                terminated=(i == 1), truncated=(i == 2), scenario=(i % 5) + 1)
    assert len(buf) == 3
    assert buf.can_sample(3)
    assert not buf.can_sample(4)
    with pytest.raises(ValueError):
        buf.sample(4)

    # Add 4 more transitions (total 7 > capacity 5) -> overwrites oldest 2 (0 and 1)
    for i in range(3, 7):
        s = np.full(11, float(i), dtype=np.float32)
        buf.add(s, action=i, reward=-float(i), next_state=s + 1.0,
                terminated=False, truncated=(i == 6), scenario=5)
    assert len(buf) == 5
    stored_first_col = set(buf.states[:, 0].tolist())
    assert stored_first_col == {2.0, 3.0, 4.0, 5.0, 6.0}


def test_replay_buffer_sample_dtypes_and_terminated_vs_truncated():
    buf = ReplayBuffer(capacity=20, state_size=11, seed=123)
    for i in range(10):
        s = np.arange(11, dtype=np.float32) + i
        # Even transitions: terminated=False, truncated=(i==4)
        # Transition 7: terminated=True, truncated=False
        buf.add(
            state=s,
            action=i % 25,
            reward=-0.5 * i,
            next_state=s + 0.5,
            terminated=(i == 7),
            truncated=(i == 4),
            scenario=(i % 5) + 1,
        )
    batch = buf.sample(10)
    assert batch["states"].shape == (10, 11) and batch["states"].dtype == np.float32
    assert batch["next_states"].shape == (10, 11) and batch["next_states"].dtype == np.float32
    assert batch["actions"].shape == (10,) and batch["actions"].dtype == np.int64
    assert batch["rewards"].shape == (10,) and batch["rewards"].dtype == np.float32
    assert batch["terminated"].shape == (10,) and batch["terminated"].dtype == np.bool_
    assert batch["truncated"].shape == (10,) and batch["truncated"].dtype == np.bool_
    assert batch["scenarios"].shape == (10,) and batch["scenarios"].dtype == np.int16

    # Exactly 1 terminated and 1 truncated in the 10 stored transitions
    assert int(batch["terminated"].sum()) == 1
    assert int(batch["truncated"].sum()) == 1
    # And they are on different transitions!
    assert not np.any(batch["terminated"] & batch["truncated"])
    assert sum(buf.scenario_counts().values()) == 10


# ---------------------------------------------------------------------------
# 3. Double DQN vs Standard DQN Target Calculation & Bootstrap Masks
# ---------------------------------------------------------------------------
def test_double_dqn_differs_from_standard_dqn_on_artificial_q_values():
    """Construct artificial Q-matrices where online argmax != target argmax.

    State 0:
      next_q_online = [10.0,  2.0,  1.0]  -> online selects action 0
      next_q_target = [ 3.0, 20.0,  5.0]  -> target evaluates action 0 as 3.0
                                             (whereas standard DQN takes max = 20.0)
    State 1:
      next_q_online = [ 0.0,  4.0,  9.0]  -> online selects action 2
      next_q_target = [ 8.0,  7.0, -2.0]  -> target evaluates action 2 as -2.0
                                             (whereas standard DQN takes max = 8.0)
    """
    rewards = tf.constant([1.0, -1.0], dtype=tf.float32)
    next_q_online = tf.constant([[10.0, 2.0, 1.0], [0.0, 4.0, 9.0]], dtype=tf.float32)
    next_q_target = tf.constant([[3.0, 20.0, 5.0], [8.0, 7.0, -2.0]], dtype=tf.float32)
    terminated = tf.constant([0.0, 0.0], dtype=tf.float32)
    gamma = 0.9

    y_ddqn = double_dqn_targets(rewards, next_q_online, next_q_target, terminated, gamma).numpy()
    y_dqn = dqn_targets(rewards, next_q_target, terminated, gamma).numpy()

    expected_ddqn = np.array([1.0 + 0.9 * 3.0, -1.0 + 0.9 * (-2.0)], dtype=np.float32)
    expected_dqn = np.array([1.0 + 0.9 * 20.0, -1.0 + 0.9 * 8.0], dtype=np.float32)

    np.testing.assert_allclose(y_ddqn, expected_ddqn, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(y_dqn, expected_dqn, rtol=1e-6, atol=1e-6)
    assert not np.allclose(y_ddqn, y_dqn)


def test_bootstrap_mask_terminated_vs_truncated():
    """Verify:
      * Transition 0: terminated=True,  truncated=False -> NO bootstrap (y = r)
      * Transition 1: terminated=False, truncated=True  -> BOOTSTRAPS (y = r + gamma * Q_target(s', a*))
      * Transition 2: terminated=False, truncated=False -> BOOTSTRAPS
    """
    norm = ObservationNormalizer(center=np.zeros(4), scale=np.ones(4))
    cfg = AgentConfig(gamma=0.99, double=True)
    agent = DDQNAgent(state_size=4, action_size=3, normalizer=norm, config=cfg, device="cpu", seed=0)

    # Set deterministic weights so Q_target(s') != 0
    agent.online.set_weights([np.full_like(w, 0.2) for w in agent.online.get_weights()])
    agent.target.set_weights([np.full_like(w, 0.5) for w in agent.target.get_weights()])

    batch = {
        "states": np.ones((3, 4), dtype=np.float32),
        "actions": np.array([0, 1, 2], dtype=np.int64),
        "rewards": np.array([-10.0, -1.5, -0.5], dtype=np.float32),
        "next_states": np.ones((3, 4), dtype=np.float32),
        "terminated": np.array([True, False, False], dtype=np.bool_),
        "truncated": np.array([False, True, False], dtype=np.bool_),
    }
    t = agent._to_tensors(batch)
    targets = agent.compute_targets(t["rewards"], t["next_states"], t["terminated"]).numpy()

    # Transition 0 (terminated=True): target must equal reward exactly (-10.0)
    assert float(targets[0]) == pytest.approx(-10.0, abs=1e-6)

    # Transition 1 (truncated=True, terminated=False): must include non-zero bootstrap
    next_q_val = float(agent.target(t["next_states"][1:2]).numpy()[0, 0])
    assert abs(next_q_val) > 1.0
    expected_trunc_target = -1.5 + 0.99 * next_q_val
    assert float(targets[1]) == pytest.approx(expected_trunc_target, rel=1e-5)

    # Transition 2 (normal step): also bootstraps identically
    expected_norm_target = -0.5 + 0.99 * next_q_val
    assert float(targets[2]) == pytest.approx(expected_norm_target, rel=1e-5)


# ---------------------------------------------------------------------------
# 4. Agent Acting, Learning, Target Sync, and Checkpointing
# ---------------------------------------------------------------------------
def test_agent_action_selection_and_greedy_determinism():
    norm = ObservationNormalizer(center=np.zeros(11), scale=np.ones(11))
    agent = DDQNAgent(state_size=11, action_size=25, normalizer=norm, device="cpu", seed=42)
    s = np.random.default_rng(0).normal(size=11).astype(np.float32)

    # Greedy action is completely deterministic and equals argmax(q_values)
    q = agent.q_values(s)[0]
    assert q.shape == (25,)
    expected_best = int(np.argmax(q))
    for _ in range(20):
        assert agent.greedy_action(s) == expected_best
        assert agent.act(s, epsilon=0.0) == expected_best

    # Exploratory actions (epsilon=1.0) stay within 0..action_size-1
    actions = {agent.act(s, epsilon=1.0) for _ in range(200)}
    assert all(0 <= a < 25 for a in actions)
    assert len(actions) > 10  # covers many actions


def test_agent_learn_target_sync_and_checkpoint_round_trip():
    norm = ObservationNormalizer(center=np.zeros(11), scale=np.ones(11))
    cfg = AgentConfig(
        gamma=0.99,
        lr=1e-3,
        batch_size=16,
        hidden_sizes=(64, 64),
        target_update="hard",
        target_update_interval=4,
    )
    agent = DDQNAgent(state_size=11, action_size=25, normalizer=norm, config=cfg, device="cpu", seed=7)

    # Target network is non-trainable and initially equals online parameters
    assert not agent.target.trainable
    assert len(agent.target.trainable_variables) == 0
    for tp, op in zip(agent.target.get_weights(), agent.online.get_weights()):
        np.testing.assert_allclose(tp, op)

    buf = ReplayBuffer(capacity=100, state_size=11, seed=7)
    rng = np.random.default_rng(7)
    for i in range(40):
        s = rng.normal(size=11).astype(np.float32)
        s2 = rng.normal(size=11).astype(np.float32)
        buf.add(s, int(i % 25), float(-rng.uniform(0.1, 2.0)), s2,
                terminated=(i % 10 == 0), truncated=(i % 15 == 0), scenario=(i % 5) + 1)

    target_before = [w.copy() for w in agent.target.get_weights()]
    online_before = [w.copy() for w in agent.online.get_weights()]

    # Steps 1..3: online updates, target stays frozen
    for step in range(1, 4):
        diag = agent.learn(buf.sample(16))
        assert np.isfinite(diag["loss"])
        assert np.isfinite(diag["grad_norm"]) and diag["grad_norm"] > 0.0
        assert agent.grad_steps == step
        assert agent.target_syncs == 0

    for op_old, op_new in zip(online_before, agent.online.get_weights()):
        assert not np.allclose(op_old, op_new)
    for tp_old, tp_new in zip(target_before, agent.target.get_weights()):
        np.testing.assert_allclose(tp_old, tp_new)

    # Step 4 triggers hard target sync (interval=4)
    agent.learn(buf.sample(16))
    assert agent.grad_steps == 4
    assert agent.target_syncs == 1
    for tp, op in zip(agent.target.get_weights(), agent.online.get_weights()):
        np.testing.assert_allclose(tp, op)

    # Soft update check
    agent.config.target_update = "soft"
    agent.config.tau = 0.1
    agent.learn(buf.sample(16))
    assert agent.target_syncs == 2

    # Non-finite loss check fails fast with FloatingPointError
    bad_batch = buf.sample(16)
    bad_batch["rewards"] = np.full(16, np.nan, dtype=np.float32)
    with pytest.raises(FloatingPointError):
        agent.learn(bad_batch)

    # Checkpoint save/load round-trip
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fh:
        ckpt_path = fh.name
    try:
        agent.save(ckpt_path, extra={"tag": "unit_test"})
        loaded = DDQNAgent.load(ckpt_path, device="cpu", seed=7)
        assert loaded.state_size == 11
        assert loaded.action_size == 25
        assert loaded.grad_steps == agent.grad_steps
        assert loaded.loaded_extra["tag"] == "unit_test"
        test_obs = rng.normal(size=11).astype(np.float32)
        np.testing.assert_allclose(
            agent.q_values(agent.preprocess(test_obs)),
            loaded.q_values(loaded.preprocess(test_obs)),
            atol=1e-6,
        )
        assert agent.greedy_action(agent.preprocess(test_obs)) == loaded.greedy_action(
            loaded.preprocess(test_obs)
        )
    finally:
        os.unlink(ckpt_path)


def test_linear_epsilon_schedule():
    sched = LinearEpsilonSchedule(start=1.0, end=0.05, decay_steps=100, warmup_steps=20)
    assert sched(0) == pytest.approx(1.0)
    assert sched(20) == pytest.approx(1.0)
    assert sched(70) == pytest.approx(0.525)
    assert sched(120) == pytest.approx(0.05)
    assert sched(500) == pytest.approx(0.05)
