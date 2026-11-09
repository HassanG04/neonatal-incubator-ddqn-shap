#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ddqn.py -- Config-driven Double DQN (or standard DQN) training.

One policy is trained across ALL five scenarios with balanced scenario
sampling (each block of 5 episodes visits every scenario once).  Training,
validation and test seeds are disjoint (see ``rl_utils``); the checkpoint
``best.pt`` is selected on VALIDATION episodes only.

Outputs (``<run>`` = ``--run-name``, default ``{algo}_seed{seed}``)::

    models/<run>/best.pt            best validation checkpoint (+ normaliser)
    models/<run>/final.pt           last checkpoint (resumable: + optimiser)
    results/ddqn/<run>/episodes.csv     one row per training episode
    results/ddqn/<run>/updates.csv      loss / Q / TD-error / grad-norm log
    results/ddqn/<run>/validation.csv   periodic greedy validation
    results/ddqn/<run>/manifest.json    versions, device, seeds, config, results

Usage::

    # seconds-long smoke run (checks every shape / code path)
    python src/train_ddqn.py --smoke
    # full run, 3 seeds, DDQN and the DQN comparison
    python src/train_ddqn.py --seed 0
    python src/train_ddqn.py --seed 0 --algo dqn

>>> SIMULATION-ONLY ACADEMIC PROJECT -- NOT FOR CLINICAL USE. <<<
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from collections import Counter, deque
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import tensorflow as tf  # noqa: E402

from baseline import RuleBasedBaselineController  # noqa: E402
from ddqn_agent import AgentConfig, DDQNAgent, LinearEpsilonSchedule, select_device  # noqa: E402
from environment import NeonatalIncubatorEnv, SCENARIO_NAMES  # noqa: E402
from q_network import ObservationNormalizer  # noqa: E402
from replay_buffer import ReplayBuffer  # noqa: E402
from rl_utils import (PROJECT_ROOT, AgentPolicy, ScenarioSampler, build_manifest,  # noqa: E402
                      run_episode, set_global_seeds, termination_cause,
                      train_seed, validation_seed, write_json)

# Defaults used when config.yaml has no / partial ``training`` section.
DEFAULT_TRAINING: Dict[str, Any] = {
    "gamma": 0.99,
    "lr": 5.0e-4,
    "batch_size": 128,
    "hidden_sizes": [128, 128],
    "buffer_size": 200_000,
    "warmup_steps": 5_000,
    "total_steps": 300_000,
    "train_freq": 1,
    "gradient_steps": 1,
    "target_update": "hard",
    "target_update_interval": 1_000,
    "tau": 0.005,
    "grad_clip_norm": 10.0,
    "huber_delta": 1.0,
    "reward_scale": 0.1,
    "eps_start": 1.0,
    "eps_end": 0.02,
    "eps_decay_steps": 150_000,
    "baseline_guided_prob": 0.0,
    "scenario_sampling": "balanced",
    "scenarios": [1, 2, 3, 4, 5],
    "eval_interval": 10_000,
    "val_episodes_per_scenario": 3,
    "log_interval": 500,
    "seeds": [0, 1, 2],
    "device": "auto",
}

SMOKE_OVERRIDES: Dict[str, Any] = {
    "total_steps": 1_500, "warmup_steps": 300, "eps_decay_steps": 1_000,
    "eval_interval": 750, "val_episodes_per_scenario": 1, "log_interval": 100,
    "target_update_interval": 200, "buffer_size": 5_000,
}


def load_training_config(config_path: Optional[str]) -> Dict[str, Any]:
    cfg = dict(DEFAULT_TRAINING)
    path = config_path or os.path.join(PROJECT_ROOT, "config.yaml")
    if os.path.isfile(path):
        import yaml
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        for k, v in (data.get("training") or {}).items():
            if k in cfg and v is not None:
                cfg[k] = v
    return cfg


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--seed", type=int, default=0, help="training run seed")
    p.add_argument("--algo", choices=["ddqn", "dqn"], default="ddqn")
    p.add_argument("--config", type=str, default=None, help="config.yaml path")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--device", type=str, default=None, help="auto | cpu | cuda")
    p.add_argument("--smoke", action="store_true",
                   help="tiny run to verify shapes/code paths (seconds)")
    p.add_argument("--models-dir", type=str, default=os.path.join(PROJECT_ROOT, "models"))
    p.add_argument("--results-dir", type=str,
                   default=os.path.join(PROJECT_ROOT, "results", "ddqn"))
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override training keys, e.g. --set lr=1e-4 gamma=0.98")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def _coerce(value: str, like: Any) -> Any:
    import yaml
    v = yaml.safe_load(value)
    if isinstance(like, float) and isinstance(v, int):
        v = float(v)
    return v


def validate(agent: DDQNAgent, envs: Dict[int, NeonatalIncubatorEnv],
             episodes_per_scenario: int) -> Dict[str, Any]:
    """Greedy evaluation on the VALIDATION seed set (never the test set)."""
    policy = AgentPolicy(agent, name="val")
    per_scn: Dict[int, List[Dict[str, float]]] = {}
    for scn, env in envs.items():
        per_scn[scn] = [run_episode(env, policy, validation_seed(scn, i),
                                    record_trajectory=False)["metrics"]
                        for i in range(episodes_per_scenario)]
    allm = [m for ms in per_scn.values() for m in ms]
    out = {
        "val_return_mean": float(np.mean([m["total_reward"] for m in allm])),
        "val_return_std": float(np.std([m["total_reward"] for m in allm])),
        "val_survival": float(np.mean([m["survived"] for m in allm])),
        "val_len_mean": float(np.mean([m["episode_length"] for m in allm])),
    }
    for scn, ms in per_scn.items():
        out[f"val_return_s{scn}"] = float(np.mean([m["total_reward"] for m in ms]))
        out[f"val_survival_s{scn}"] = float(np.mean([m["survived"] for m in ms]))
    return out


def train(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_training_config(args.config)
    if args.smoke:
        cfg.update(SMOKE_OVERRIDES)
    for kv in args.set:
        k, _, v = kv.partition("=")
        if k not in cfg:
            raise SystemExit(f"unknown training key {k!r}")
        cfg[k] = _coerce(v, cfg[k])
    if args.total_steps is not None:
        cfg["total_steps"] = int(args.total_steps)
    if args.device is not None:
        cfg["device"] = args.device
    cfg["algo"] = args.algo

    run_name = args.run_name or (f"{args.algo}_seed{args.seed}"
                                 + ("_smoke" if args.smoke else ""))
    model_dir = os.path.join(args.models_dir, run_name)
    res_dir = os.path.join(args.results_dir, run_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    set_global_seeds(args.seed)
    device = select_device(cfg["device"])
    log = (lambda *a, **k: None) if args.quiet else print

    # ---- Environments: one instance per scenario (training) + validation.
    scenarios = [int(s) for s in cfg["scenarios"]]
    train_envs = {s: NeonatalIncubatorEnv(scenario=s, config_path=args.config)
                  for s in scenarios}
    val_envs = {s: NeonatalIncubatorEnv(scenario=s, config_path=args.config)
                for s in scenarios}
    probe = train_envs[scenarios[0]]
    state_size = int(probe.observation_space.shape[0])   # 11 by default
    action_size = int(probe.action_space.n)              # 25 by default

    normalizer = ObservationNormalizer.from_env(probe)
    agent_cfg = AgentConfig(
        gamma=float(cfg["gamma"]), lr=float(cfg["lr"]),
        batch_size=int(cfg["batch_size"]),
        hidden_sizes=tuple(int(h) for h in cfg["hidden_sizes"]),
        grad_clip_norm=float(cfg["grad_clip_norm"]),
        target_update=str(cfg["target_update"]),
        target_update_interval=int(cfg["target_update_interval"]),
        tau=float(cfg["tau"]), huber_delta=float(cfg["huber_delta"]),
        double=(args.algo == "ddqn"),
    )
    agent = DDQNAgent(state_size, action_size, normalizer, agent_cfg,
                      device=device, seed=args.seed)
    buffer = ReplayBuffer(int(cfg["buffer_size"]), state_size, seed=args.seed + 17)
    sampler = ScenarioSampler(scenarios, seed=args.seed + 31,
                              mode=str(cfg["scenario_sampling"]))
    eps_sched = LinearEpsilonSchedule(float(cfg["eps_start"]), float(cfg["eps_end"]),
                                      int(cfg["eps_decay_steps"]),
                                      warmup_steps=int(cfg["warmup_steps"]))
    reward_scale = float(cfg["reward_scale"])
    guided_p = float(cfg["baseline_guided_prob"])

    log("=" * 78)
    log(f"[train] run={run_name} algo={args.algo} seed={args.seed} device={device}")
    log(f"[train] state_size={state_size} action_size={action_size} "
        f"scenarios={scenarios} total_steps={cfg['total_steps']}")
    log("=" * 78)

    episodes_log: List[Dict[str, Any]] = []
    updates_log: List[Dict[str, Any]] = []
    val_log: List[Dict[str, Any]] = []
    action_counts: Counter = Counter()
    upd_window: List[Dict[str, float]] = []
    recent = deque(maxlen=50)
    best_val = -np.inf
    best_step = -1

    total_steps = int(cfg["total_steps"])
    warmup = int(cfg["warmup_steps"])
    batch_size = int(cfg["batch_size"])
    env_step = 0
    ep_idx = 0
    t0 = time.time()

    while env_step < total_steps:
        scn = sampler.next()
        env = train_envs[scn]
        seed = train_seed(args.seed, ep_idx)
        obs, info = env.reset(seed=seed)
        guide = RuleBasedBaselineController.from_env(env) if guided_p > 0 else None
        s = agent.preprocess(obs)
        ep_ret = 0.0
        ep_len = 0
        n_guided = 0
        n_random = 0
        n_greedy = 0
        last_action_source = "none"
        terminated = truncated = False
        while not (terminated or truncated):
            eps = eps_sched(env_step)
            guide_action = guide.select_action(obs) if guide is not None else None
            if agent.rng.random() < eps:
                if guide is not None and agent.rng.random() < guided_p:
                    action = int(guide_action)
                    n_guided += 1
                    last_action_source = "guided"
                else:
                    action = int(agent.rng.integers(action_size))
                    n_random += 1
                    last_action_source = "random"
            else:
                action = agent.greedy_action(s)
                n_greedy += 1
                last_action_source = "greedy"
            obs2, reward, terminated, truncated, info = env.step(action)
            s2 = agent.preprocess(obs2)
            # Real final observation is stored (no auto-reset); truncated
            # transitions keep their bootstrap, terminated ones do not.
            buffer.add(s, action, reward * reward_scale, s2, terminated, truncated, scn)
            action_counts[action] += 1
            s, obs = s2, obs2
            ep_ret += float(reward)
            ep_len += 1
            env_step += 1

            if (env_step >= warmup and buffer.can_sample(batch_size)
                    and env_step % int(cfg["train_freq"]) == 0):
                for _ in range(int(cfg["gradient_steps"])):
                    upd_window.append(agent.learn(buffer.sample(batch_size)))
                if len(upd_window) >= int(cfg["log_interval"]):
                    row = {k: float(np.mean([u[k] for u in upd_window]))
                           for k in upd_window[0]}
                    row["td_abs_max"] = float(np.max([u["td_abs_max"] for u in upd_window]))
                    row.update({"env_step": env_step, "grad_steps": agent.grad_steps,
                                "epsilon": eps,
                                "steps_per_sec": env_step / max(1e-9, time.time() - t0)})
                    updates_log.append(row)
                    upd_window = []

            if env_step % int(cfg["eval_interval"]) == 0 and env_step >= warmup:
                v = validate(agent, val_envs, int(cfg["val_episodes_per_scenario"]))
                v.update({"env_step": env_step, "episode": ep_idx, "epsilon": eps,
                          "wall_s": time.time() - t0})
                val_log.append(v)
                if v["val_return_mean"] > best_val:
                    best_val = v["val_return_mean"]
                    best_step = env_step
                    agent.save(os.path.join(model_dir, "best.pt"),
                               extra={"env_step": env_step, "algo": args.algo,
                                      "seed": args.seed, "training_config": cfg,
                                      "validation": v},
                               include_optimizer=False)
                log(f"[val ] step={env_step:>7d} eps={eps:.3f} "
                    f"ret={v['val_return_mean']:9.2f} surv={v['val_survival']:.2f} "
                    f"len={v['val_len_mean']:6.1f} best={best_val:9.2f}@{best_step}")
            if env_step >= total_steps:
                break

        cause = termination_cause(info, terminated, truncated)
        episodes_log.append({
            "episode": ep_idx, "env_step": env_step, "scenario": scn, "seed": seed,
            "return": ep_ret, "length": ep_len, "terminated": bool(terminated),
            "truncated": bool(truncated), "cause": cause,
            "epsilon": eps_sched(env_step), "guided_actions": n_guided,
            "random_actions": n_random, "greedy_actions": n_greedy,
            "last_action_source": last_action_source,
        })
        recent.append((ep_ret, ep_len, terminated))
        ep_idx += 1
        if not args.quiet and ep_idx % 200 == 0:
            r = np.array(recent, dtype=float)
            log(f"[ep  ] ep={ep_idx:>6d} step={env_step:>7d} eps={eps_sched(env_step):.3f} "
                f"ret50={r[:, 0].mean():9.2f} len50={r[:, 1].mean():6.1f} "
                f"term50={r[:, 2].mean():.2f} sps={env_step / (time.time() - t0):.0f}")

    wall = time.time() - t0
    agent.save(os.path.join(model_dir, "final.pt"),
               extra={"env_step": env_step, "algo": args.algo, "seed": args.seed,
                      "training_config": cfg})
    if best_step < 0:  # e.g. run shorter than eval_interval
        agent.save(os.path.join(model_dir, "best.pt"),
                   extra={"env_step": env_step, "algo": args.algo, "seed": args.seed,
                          "training_config": cfg, "note": "no validation performed"},
                   include_optimizer=False)

    ep_df = pd.DataFrame(episodes_log)
    ep_df.to_csv(os.path.join(res_dir, "episodes.csv"), index=False)
    pd.DataFrame(updates_log).to_csv(os.path.join(res_dir, "updates.csv"), index=False)
    pd.DataFrame(val_log).to_csv(os.path.join(res_dir, "validation.csv"), index=False)

    # Exclude unfinished final episode ("in_progress") from completed-episode safety statistics
    done_df = ep_df[ep_df["cause"] != "in_progress"] if len(ep_df) else ep_df
    late_steps_df = done_df[done_df["env_step"] >= int(0.8 * total_steps)] if len(done_df) else done_df
    term_df = done_df[done_df["terminated"]] if len(done_df) else done_df
    late_term_df = late_steps_df[late_steps_df["terminated"]] if len(late_steps_df) else late_steps_df

    results = {
        "env_steps": env_step, "episodes": ep_idx,
        "completed_episodes": int(len(done_df)),
        "wall_seconds": wall,
        "steps_per_second": env_step / max(wall, 1e-9),
        "grad_steps": agent.grad_steps, "target_syncs": agent.target_syncs,
        "best_val_return": None if best_step < 0 else float(best_val),
        "best_val_step": best_step,
        "train_terminated_fraction": float(done_df["terminated"].mean()) if len(done_df) else None,
        "train_terminated_fraction_last20pct": (
            float(done_df["terminated"].iloc[int(0.8 * len(done_df)):].mean()) if len(done_df) else None),
        "train_terminated_fraction_last20pct_env_steps": (
            float(late_steps_df["terminated"].mean()) if len(late_steps_df) else None),
        "train_terminated_per_env_step_last20pct": (
            float(late_steps_df["terminated"].sum() / max(1, late_steps_df["length"].sum()))
            if len(late_steps_df) else None),
        "train_causes": done_df["cause"].value_counts().to_dict() if len(done_df) else {},
        "train_causes_last20pct_env_steps": (
            late_steps_df["cause"].value_counts().to_dict() if len(late_steps_df) else {}),
        "terminated_last_action_source": (
            term_df["last_action_source"].value_counts().to_dict() if len(term_df) else {}),
        "terminated_last_action_source_last20pct_env_steps": (
            late_term_df["last_action_source"].value_counts().to_dict() if len(late_term_df) else {}),
        "buffer_scenario_counts": buffer.scenario_counts(),
        "action_counts": {int(k): int(v) for k, v in sorted(action_counts.items())},
        "state_size": state_size, "action_size": action_size,
    }
    manifest = build_manifest(
        config={"training": cfg, "agent": agent_cfg.to_dict(),
                "normalizer": normalizer.state_dict(), "run_name": run_name},
        seeds={"run_seed": args.seed,
               "train_seed_rule": "1_000_000*(run_seed+1)+episode_index",
               "validation_seed_rule": "500_000+scenario*1000+i",
               "buffer_seed": args.seed + 17, "sampler_seed": args.seed + 31},
        device=str(device), results=results, config_path=args.config)
    write_json(manifest, os.path.join(res_dir, "manifest.json"))
    log(f"[done] {env_step} steps, {ep_idx} episodes in {wall:.1f}s "
        f"({results['steps_per_second']:.0f} steps/s); best val={results['best_val_return']} "
        f"@ {best_step}. Outputs: {model_dir} , {res_dir}")
    return results


def main(argv=None) -> int:
    train(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
