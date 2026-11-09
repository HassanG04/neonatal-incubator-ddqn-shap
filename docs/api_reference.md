# API Reference — Neonatal Incubator Environment (v3.1)

> This document is the **programmer's reference** for the
> `NeonatalIncubatorEnv` class in `src/environment.py`. The environment is
> Gymnasium-compatible (5-tuple `step`, 2-tuple `reset`) and is the contract
> between Member 1 (environment + baseline), Member 2 (DDQN trainer), and
> Member 3 (SHAP / XAI).
> v3.1 changes (V3-01): `load_config()` returns a resolved dict (no
> module-global mutation), `__init__` stores ALL params on the instance,
> and `action_table` is rebuilt per-instance from `self.q_heater_max`. A
> custom `config_path` argument can override ALL parameters.
> All numbers are simulation parameters, not clinical guidance.

---

## 1. Quick Start

```python
import sys; sys.path.insert(0, "src")
from environment import NeonatalIncubatorEnv, OBS_NAMES

env = NeonatalIncubatorEnv(scenario=1, seed=42)          # loads ../config.yaml (V3-01)
obs, info = env.reset(seed=42)
print("obs names :", OBS_NAMES)                            # 11 features
print("obs shape :", obs.shape, obs.dtype)                 # (11,) float32
print("action    :", env.action_space)                    # Discrete(25)

action = 12                                                # heater_idx=2, fio2=0.40
obs, reward, terminated, truncated, info = env.step(action)
print(f"step={info['step']}  Tc={info['Tc']:.2f}  SpO2={info['SpO2']:.1f}  "
      f"r={reward:+.3f}  terminated={terminated}  truncated={truncated}")
```

For ablation studies (V3-01), pass a custom `config_path`:

```python
env_ablation = NeonatalIncubatorEnv(
    scenario=1, seed=42,
    config_path="ablation.yaml",   # overrides any subset of parameters
)
print(env_ablation.desat_offset)   # 5.0 if ablation.yaml sets it
```

---

## 2. Class Signature (V3-01)

```python
class NeonatalIncubatorEnv(gym.Env):

    def __init__(
        self,
        scenario: int = 1,              # 1..5 (see SCENARIO_NAMES)
        max_steps: int | None = None,  # default config.yaml -> environment.max_steps = 200
        dt: float | None = None,        # default config.yaml -> environment.dt = 45.0 s
        seed: int | None = None,
        config_path: str | None = None, # V3-01: path to YAML config (default ../config.yaml)
    ) -> None
```

The `config_path` parameter is the V3-01 hook that makes `config.yaml` the
single source of truth. When `config_path is None`, the constructor looks
for `config.yaml` next to the `src/` directory; if found, the file is
parsed by `load_config()` into a **resolved dict** and used to populate
all instance attributes. If the file is missing or `pyyaml` is
unavailable, the module-level defaults are used silently (the simulator
stays self-contained).

Crucially, V3-01 changed `load_config()` from a side-effectful mutator of
module globals into a **pure function** that returns a `dict`. The
`__init__` method then reads from that dict and stores every parameter
on the instance (no module globals are touched). This means:

* two `NeonatalIncubatorEnv` instances built with different `config_path`s
  in the same process do not interfere with each other;
* a downstream consumer (Member 2 trainer, Member 3 SHAP explainer) can
  read the parameters from `env.k_loss`, `env.w_thermal`,
  `env.r_safety_penalty`, etc., without inspecting the YAML;
* the resolved config dict is also exposed as `env._config` for full
  inspection (useful when running ablation studies).

---

## 3. Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `reset` | `reset(*, seed=None, options=None) -> (obs, info)` | Reset to the scenario's initial state; returns an 11-dim `obs` and an info dict. |
| `step` | `step(action: int) -> (obs, r, terminated, truncated, info)` | Advance the simulation by one control step (45 s simulated time). |
| `render` | `render(mode="human")` | Print a one-line state summary. |
| `close` | `close()` | No-op (no external resources). |
| `get_state_dict` | `get_state_dict() -> dict` | Snapshot of the full state (for logging / replay). |

The constructor also exposes the frozen contract attributes
`obs_names`, `heater_levels`, `fio2_levels`, `action_table`,
`observation_space`, `action_space`, plus — as a V3-01 change — the full
set of physics, reward, and safety parameters listed in §4 below.

---

## 4. Instance-Level Parameters (V3-01)

Every parameter that influences the dynamics is stored on the instance.
The table below lists the attribute name, type, default value, and
provenance category. Member 2 may read any of these directly from the
env object (e.g. `env.k_loss`, `env.w_thermal`).

| Attribute | Type | Default | Category |
|-----------|------|--------:|----------|
| `k_loss` | float | 8.0 | physics (literature-derived) |
| `k_baby` | float | 4.0 | physics (assumed) |
| `c_air` | float | 1500.0 | physics (calibrated educational parameter, V3-02) |
| `c_core` | float | 2000.0 | physics (calibrated educational parameter, V3-02) |
| `porthole_h_c_factor` | float | 2.0 | physics (calibrated) |
| `k_blend` | float | 0.05 | physics (assumed) |
| `k_leak` | float | 0.005 | physics (assumed) |
| `desat_offset` | float | 5.0 | physics (calibrated educational parameter, V3-03) |
| `k_spo2` | float | 0.02 | physics (literature-derived) |
| `spo2_base` | float | 92.0 | physics (literature-derived) |
| `spo2_gain` | float | 0.15 | physics (assumed) |
| `spo2_min` | float | 70.0 | physics (assumed) |
| `spo2_max` | float | 100.0 | physics (literature-derived) |
| `tc_safe_low` | float | 35.5 | safety (literature-derived) |
| `tc_safe_high` | float | 38.0 | safety (literature-derived) |
| `spo2_safe_low` | float | 85.0 | safety (literature-derived) |
| `spo2_safe_high` | float | 96.0 | safety (literature-derived) |
| `w_thermal` | float | 1.0 | reward (calibrated) |
| `w_oxygen` | float | 0.5 | reward (calibrated) |
| `w_energy` | float | 0.05 | reward (calibrated) |
| `w_safety` | float | 1.0 | reward (calibrated) |
| `r_safety_penalty` | float | 100.0 | reward (calibrated) |
| `tc_setpoint` | float | 37.0 | reward (literature-derived) |
| `spo2_target_low` | float | 91.0 | reward (literature-derived, NICE NG124) |
| `spo2_target_high` | float | 95.0 | reward (literature-derived, NICE NG124) |
| `q_heater_max` | float | 360.0 | environment (literature-derived) |
| `max_steps` | int | 200 | environment (assumed) |
| `dt` | float | 45.0 | environment (assumed) |
| `n_substeps` | int | 10 | environment (calibrated) |
| `_config` | dict | (resolved) | the full config dict (V3-01) |

---

## 5. Configuration (V3-01)

The `config_path` parameter controls where the constructor looks for the
YAML file. The resolved config dict is stored on `env._config`. The
following snippets illustrate the three typical use cases:

### 5.1 Default config

```python
env = NeonatalIncubatorEnv(scenario=1, seed=42)
# env._config == resolved dict from ../config.yaml
# env.k_loss == 8.0, env.desat_offset == 5.0, env.q_heater_max == 360.0
```

### 5.2 Custom config (ablation)

```python
# ablation.yaml overrides only the parameters you want to change
env = NeonatalIncubatorEnv(scenario=1, seed=42,
                           config_path="ablation.yaml")
# env.k_loss now reflects the override; env.action_table is rebuilt
# from the new env.q_heater_max (V3-01)
```

### 5.3 Comparing two configs side-by-side

Because V3-01 makes `load_config()` a pure function, two environments
with different `config_path`s can coexist in the same Python process:

```python
env_default = NeonatalIncubatorEnv(scenario=5, seed=42)
env_high_desat = NeonatalIncubatorEnv(
    scenario=5, seed=42,
    config_path="config_desat_offset_8.yaml",  # V3-03 ablation
)
assert env_default.desat_offset == 5.0
assert env_high_desat.desat_offset == 8.0
# Each env has its OWN action_table built from its own q_heater_max.
```

### 5.4 `action_table` is per-instance (V3-01)

```python
env = NeonatalIncubatorEnv(scenario=1, seed=42)
# env.action_table is a list[dict] of 25 entries, each with keys:
#   action, heater_idx, o2_idx, heater_level, fio2_command, Q_heater_W
# The Q_heater_W column is computed as heater_level * env.q_heater_max.
# If q_heater_max is overridden via config_path, the action_table is
# rebuilt from the new value at construction time.
for row in env.action_table[:5]:
    print(row)
```

---

## 6. Observation Space

`spaces.Box(low, high, shape=(11,), dtype=float32)`. Feature order is
frozen by `OBS_NAMES` and mirrored in `config.yaml → environment.obs_names`.

| # | Feature | Unit | Low | High | Description |
|---|---------|------|----:|-----:|-------------|
| 0 | `Tc` | °C | 30 | 42 | Core body temperature |
| 1 | `Tsk` | °C | 30 | 42 | Skin temperature (`= Tc − 0.5`) |
| 2 | `Ta` | °C | 20 | 40 | Incubator air temperature |
| 3 | `C_O2` | % | 21 | 60 | Inspired O2 concentration |
| 4 | `SpO2` | % | 70 | 100 | Arterial O2 saturation (separate lagged state) |
| 5 | `Tamb` | °C | 15 | 35 | Ambient room temperature |
| 6 | `dTa_dt` | °C/s | −2 | +2 | Realized rate of `Ta` |
| 7 | `dC_O2_dt` | %/s | −3 | +3 | Realized rate of `C_O2` |
| 8 | `phase` | − | 0 | 1 | Normalised episode time |
| 9 | `disturbance` | − | 0 | 2 | 0 idle, 1 porthole, 2 O2 desaturation |
| 10 | `scenario_id` | − | 0 | 1 | `(scenario − 1)/4` (restores Markov) |

**Note**: the observation is **11-dimensional, not 10**. The
`scenario_id` feature (index 10) exposes the hidden scenario context.
Downstream consumers must not assume a 10-dim observation.

---

## 7. Action Space

`spaces.Discrete(25)` = 5 heater levels × 5 FiO2 commands.
`action = heater_idx × 5 + o2_idx`.

| action | heater_idx | o2_idx | heater_level | fio2_command | Q_heater (W) |
|--------|------------|--------|--------------|--------------|-------------|
| 0  | 0 | 0 | 0.00 | 0.21 |   0 |
| 1  | 0 | 1 | 0.00 | 0.30 |   0 |
| 2  | 0 | 2 | 0.00 | 0.40 |   0 |
| 3  | 0 | 3 | 0.00 | 0.50 |   0 |
| 4  | 0 | 4 | 0.00 | 0.60 |   0 |
| 5  | 1 | 0 | 0.25 | 0.21 |  90 |
| 6  | 1 | 1 | 0.25 | 0.30 |  90 |
| 7  | 1 | 2 | 0.25 | 0.40 |  90 |
| 8  | 1 | 3 | 0.25 | 0.50 |  90 |
| 9  | 1 | 4 | 0.25 | 0.60 |  90 |
| 10 | 2 | 0 | 0.50 | 0.21 | 180 |
| 11 | 2 | 1 | 0.50 | 0.30 | 180 |
| 12 | 2 | 2 | 0.50 | 0.40 | 180 |
| 13 | 2 | 3 | 0.50 | 0.50 | 180 |
| 14 | 2 | 4 | 0.50 | 0.60 | 180 |
| 15 | 3 | 0 | 0.75 | 0.21 | 270 |
| 16 | 3 | 1 | 0.75 | 0.30 | 270 |
| 17 | 3 | 2 | 0.75 | 0.40 | 270 |
| 18 | 3 | 3 | 0.75 | 0.50 | 270 |
| 19 | 3 | 4 | 0.75 | 0.60 | 270 |
| 20 | 4 | 0 | 1.00 | 0.21 | 360 |
| 21 | 4 | 1 | 1.00 | 0.30 | 360 |
| 22 | 4 | 2 | 1.00 | 0.40 | 360 |
| 23 | 4 | 3 | 1.00 | 0.50 | 360 |
| 24 | 4 | 4 | 1.00 | 0.60 | 360 |

`fio2_command` is the **absolute** volume fraction (0.21–0.60), not a
normalised `[0, 1]` fraction. The `Q_heater (W)` column is computed from
`self.q_heater_max`; if you override `q_heater_max` via `config_path`
(V3-01), the per-instance `action_table` is rebuilt automatically — the
`heater_level` column stays the same but the `Q_heater_W` column scales
with the new maximum.

---

## 8. Reward (exact weighted formula)

```python
r_thermal = -(Tc - 37.0)**2

if SpO2 < 91:    spo2_err = (91 - SpO2)**2
elif SpO2 > 95:   spo2_err = (SpO2 - 95)**2
else:             spo2_err = 0.0
r_oxygen  = -spo2_err

u_fio2_norm = (fio2_cmd - 0.21) / (0.60 - 0.21)
r_energy  = -(u_heater**2 + u_fio2_norm**2)

unsafe    = not (35.5 <= Tc <= 38.0 and 85 <= SpO2 <= 96)
r_safety  = -100.0 if unsafe else 0.0

reward = w_thermal*r_thermal + w_oxygen*r_oxygen \
       + w_energy*r_energy   + w_safety*r_safety
```

Default weights (read from `env.w_thermal`, `env.w_oxygen`,
`env.w_energy`, `env.w_safety` in V3-01):
`w_thermal = 1.0`, `w_oxygen = 0.5`, `w_energy = 0.05`, `w_safety = 1.0`.

Worked example (`Tc=37.0`, `SpO2=93.0`, `u_heater=0.5`, `fio2=0.30`, safe):
`r_thermal = 0`, `r_oxygen = 0`, `r_energy = -0.303`, `r_safety = 0`,
`R = -0.0152`.

---

## 9. Termination vs Truncation

| Flag | Trigger | Effect on reward | Bootstrap target |
|------|---------|------------------|------------------|
| `terminated` | `Tc ∉ [35.5, 38.0]` OR `SpO2 ∉ [85, 96]` | `r_safety = -100` is added to the step reward | `target = r` (terminal state, do **not** bootstrap `V(s')`) |
| `truncated` | `current_step >= max_steps` (200) | No penalty; episode simply ends | `target = r + γ · Q(s', argmax_a Q(s', a; θ))` (bootstrap normally — truncation is *not* terminal) |

The standard Gymnasium convention applies: on `terminated=True` the
bootstrap target uses reward only; on `truncated=True` the bootstrap uses
the normal TD target (the episode ended for budget reasons, not because
the agent failed).

In the v3.1 baseline, every one of the 25 evaluation episodes (5 scenarios
× 5 seeds) ran to `step = 200` with **0** safety violations.

---

## 10. Episode Lifecycle

```
env = NeonatalIncubatorEnv(scenario=3, seed=42)
obs, info = env.reset(seed=42)        # obs.shape = (11,) float32
                                        # info carries the full step-0 snapshot

done = False
while not done:
    action = policy(obs)               # int in [0, 25)
    obs, r, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

* Episode length: `200` control steps × `dt = 45 s` ⇒ `2.5 h` of simulated time.
* Sub-steps: `N_SUBSTEPS = 10` internal Euler steps per control step
  (`dt_sub = 4.5 s`) — keeps every rate × `dt_sub < 0.5` for stability.
* Per-step info dict fields: `step`, `scenario`, `scenario_name`,
  `Tc`, `Tsk`, `Ta`, `C_O2`, `SpO2`, `Tamb`, `dTa_dt`, `dC_O2_dt`, `phase`,
  `disturbance`, `scenario_id`, `heater_idx`, `o2_idx`, `u_heater`,
  `fio2_cmd`, `Q_heater_W`, `r_thermal`, `r_oxygen`, `r_energy`, `r_safety`,
  `reward`, `w_thermal`, `w_oxygen`, `w_energy`, `w_safety`,
  `hypothermia`, `hyperthermia`, `hypoxia`, `hyperoxia`, `safe`,
  `terminated`, `truncated`, `desat_offset`.

The info dict exposes the same per-instance parameters as the env object
itself, so Member 2 can log the actual reward weights used in each step
even when running ablations with non-default configs.

---

## 11. Scenarios

| ID | Name | Initial state | Disturbance |
|----|------|---------------|-------------|
| 1 | `normal_standard` | `Tc=37, Ta=34, SpO2=92, Tamb=25` | none |
| 2 | `ambient_temperature_drop` | `Tc=37, Ta=34, SpO2=92, Tamb=25` | `Tamb(t) = 25 - 7·(step/max_steps)` → 18 °C |
| 3 | `porthole_opening_event` | `Tc=37, Ta=36 (pre-warmed), SpO2=92, Tamb=25` | `h_c = 2.0` on steps [50, 80]; `disturbance=1` |
| 4 | `initial_hypothermia` | `Tc=35.8, Ta=36, SpO2=88, Tamb=25` | none (initial-condition stress) |
| 5 | `oxygen_disturbance_desaturation` | `Tc=37, Ta=34, SpO2=92, Tamb=25` | `desat_offset(t) = 5·sin(π·(step-60)/60)` on [60, 120]; `disturbance=2` (V3-03) |

See `docs/scenarios.md` for full clinical context, difficulty and
evaluation protocol. Note that the Scenario 5 disturbance amplitude is
`5.0` in v3.1 (down from `8.0` in v3.0), reflecting the V3-03 retune
that makes the desaturation event a recoverable challenge rather than a
stress test.

---

## 12. Reproducibility, Known Quirks & DDQN Integration

### 12.1 Reproducibility

* `env.reset(seed=42)` seeds `numpy.random.default_rng(seed)`. All sensor
  noise (Gaussian, `σ_Tc = 0.01 °C`, `σ_SpO2 = 0.05 %`) is drawn from this
  generator, so identical seeds produce byte-identical trajectories.
* The ODE integration is deterministic (explicit Euler, no adaptive steps).
* `config.yaml` is the single source of truth — Members 2 and 3 must use
  the same YAML (or supply the same `config_path`) to reproduce the
  baseline numbers below.
* Held-out RL evaluation protocol (`src/evaluate_controllers.py`):
  defaults to **10 held-out test episodes per scenario** (`50` episodes per
  controller/seed on disjoint `test_seed(scenario, i) = 900_000 + scenario*1000 + i`),
  stored in `results/ddqn/`. An expanded **30 episodes per scenario**
  (`150` episodes per controller/seed) evaluation is also provided in
  `results/ddqn_30ep/`.

### 12.2 Known quirks

| # | Quirk | Implication |
|---|-------|-------------|
| 1 | **11-dim, not 10.** The observation has `scenario_id` at index 10. | Downstream code must not hard-code `obs.shape = (10,)`. The Q-network input layer and SHAP explainer must use `len(env.obs_names) = 11`. |
| 2 | **FiO2 is absolute.** `fio2_command ∈ {0.21, 0.30, 0.40, 0.50, 0.60}` are *volume fractions*, not normalised `[0, 1]`. | The oxygen mass balance uses `FiO2·100` to convert to percent. Reward energy term normalises internally via `(fio2 − 0.21)/0.39`. |
| 3 | **`scenario_id` restores Markov.** Two states with identical `Tc, Ta, SpO2` from different scenarios previously aliased. | Member 2's Q-network must include `scenario_id` as an input feature, or the agent will under-perform on scenarios 3 and 5. |
| 4 | **Realized rates.** `dTa_dt` and `dC_O2_dt` are *post-clipping* per-second differences between consecutive states. | They are zero at reset (no prior state to difference) and always within `observation_space` bounds. |
| 5 | **`desat_offset` depresses `SpO2_eq` only.** Scenario 5's disturbance does **not** lower `C_O2` (which is floored at 21 %). | Member 2 cannot "fix" scenario 5 by raising heater power; the only effective lever is `FiO2`. |
| 6 | **Porthole feasibility.** `K_LOSS = 8 W/°C`, `C_AIR = 1500 J/°C` (effective mass, V3-02), `PORTHOLE_H_C_FACTOR = 2.0` (down from 3.5) are jointly tuned so full heating can hold `Tc ≥ 35.5` during the porthole window. | If any of these is perturbed, re-run `tests/test_environment.py::test_porthole_feasibility` before training. |
| 7 | **`config_path` is authoritative (V3-01).** The resolved dict from `load_config(config_path)` overrides module defaults at construction. | Edit YAML (or pass a custom `config_path`), not Python, to retune the simulator. The resolved dict is exposed as `env._config`. |
| 8 | **`action_table` is per-instance (V3-01).** The table is rebuilt from `env.q_heater_max` at construction time. | If you override `q_heater_max` via `config_path`, re-read `env.action_table` after construction; do not cache a reference to a module-level `ACTION_TABLE`. |
| 9 | **Checkpoint format (`neonatal-ddqn-tf-v1`).** Checkpoints in `models/*/*.pt` are **TensorFlow/Keras** dictionaries serialized via Python `pickle`. | Load trusted checkpoints via `DDQNAgent.load(path)` from `src/ddqn_agent.py`. Do **not** use `torch.load()`. |

### 12.3 DDQN integration snippet (TensorFlow / Keras)

```python
import sys
sys.path.insert(0, "src")

import numpy as np
import tensorflow as tf
from environment import NeonatalIncubatorEnv, OBS_NAMES
from q_network import ObservationNormalizer
from ddqn_agent import AgentConfig, DDQNAgent, double_dqn_targets

env = NeonatalIncubatorEnv(scenario=1, seed=0, config_path="config.yaml")
obs_dim = len(OBS_NAMES)              # 11
n_actions = env.action_space.n        # 25

normalizer = ObservationNormalizer.from_env(env)
cfg = AgentConfig(
    gamma=0.99,
    lr=3.0e-4,
    batch_size=128,
    hidden_sizes=(128, 128),
    target_update="hard",
    target_update_interval=500,
    double=True,
)
agent = DDQNAgent(obs_dim, n_actions, normalizer, cfg, device="cpu", seed=0)

obs, _ = env.reset(seed=0)
norm_state = agent.preprocess(obs)

# Greedy action or epsilon-greedy action
action = agent.act(norm_state, epsilon=0.1)
next_obs, reward, terminated, truncated, info = env.step(action)

# Safe checkpoint save & reload (pickle-serialized TensorFlow/Keras dict; load only trusted files)
agent.save("models/example_ddqn.pt", extra={"step": 1})
reloaded = DDQNAgent.load("models/example_ddqn.pt", device="cpu", seed=0)
q_vec = reloaded.q_values(reloaded.preprocess(next_obs))[0]  # shape (25,)
```

The full reference setup is documented in
`docs/handoff_to_member2.md §7 (Recommended DDQN Setup)` and the
hyperparameters live in `config.yaml → training`.

**Simulation parameters, not clinical guidance.**
