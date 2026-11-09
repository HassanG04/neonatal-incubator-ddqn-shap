"""Replay the submitted experiment independently of the Member 03 exporter.

Run from the project root. Only the known, hash-checked pickle is deserialized.
For Deep SHAP, independently propagate DeepLIFT rescale multipliers through the
two dense ReLU layers. This checks attribution values, not just efficiency.
"""
import argparse
import hashlib
import json
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from baseline import RuleBasedBaselineController
from environment import NeonatalIncubatorEnv, OBS_NAMES

HASH = "823a2ca6988adbf7ddb4a303da82c1134bcff0d18ef8c59ac21f52cc375411ce"
NAMES = list(OBS_NAMES)
ATOL = 1e-4


def close(actual, expected, tol=1e-8):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tol, equal_nan=True)


class ReferenceNetwork:
    def __init__(self, path):
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == HASH
        ck = pickle.loads(data)
        assert ck["normalizer"]["names"] == NAMES
        assert ck["agent_config"]["double"]
        self.w3, self.b3, self.w1, self.b1, self.w2, self.b2 = ck["online"]
        assert [w.shape for w in (self.w1, self.w2, self.w3)] == [(11, 128), (128, 128), (128, 25)]
        self.norm = ck["normalizer"]

    def normalize(self, x):
        center, scale = [np.array(self.norm[k], dtype=np.float32) for k in ("center", "scale")]
        return np.clip((np.asarray(x, dtype=np.float32) - center) / scale,
                       -self.norm["clip_value"], self.norm["clip_value"])

    def layers(self, x):
        h1 = x @ self.w1 + self.b1
        h2 = np.maximum(h1, 0) @ self.w2 + self.b2
        return h1, h2, np.maximum(h2, 0) @ self.w3 + self.b3

    def q(self, raw):
        return self.layers(self.normalize(raw))[-1]

    def deep_values(self, x, bg, ref_layers, action):
        """Mean of reference-to-input contributions for one scalar Q head."""
        levels = self.layers(x)
        slopes = []
        for value, ref in zip(levels[:2], ref_layers[:2]):
            delta = value - ref
            slope = (ref > 0).astype(np.float32)
            np.divide(np.maximum(value, 0) - np.maximum(ref, 0), delta, out=slope,
                      where=np.abs(delta) >= 1e-6)
            slopes.append(slope)
        mult2 = self.w3[:, action] * slopes[1]
        mult1 = (mult2 @ self.w2.T) * slopes[0]
        return ((x - bg) * (mult1 @ self.w1.T)).mean(axis=0)


def kernel_reference(net, bg):
    """Use the third-party estimator directly on the independent NumPy network."""
    import shap
    models = {}
    def explain(x, action=None):
        if action not in models:
            def predict(batch):
                q = net.layers(np.asarray(batch, dtype=np.float32))[-1]
                return q if action is None else q[:, action]
            models[action] = shap.KernelExplainer(predict, bg)
        np.random.seed(zlib.crc32(x.tobytes()))
        return models[action].shap_values(x[None], nsamples=100, l1_reg=0, silent=True)[0]
    return explain


def first_hold(mask, start):
    for i in range(start, len(mask) - 4):
        if mask[i:i+5].all():
            return i - start
    return np.nan


def recalculate(t, scenario):
    tc, o2, heat, fio2, action = [t[k].to_numpy() for k in ("Tc", "SpO2", "u_heater", "fio2_command", "action")]
    oxygen = (fio2 - .21) / .39
    temp_error = abs(tc - 37)
    temp_band = (tc >= 36.5) & (tc <= 37.5)
    oxygen_error = np.maximum.reduce([91 - o2, o2 - 95, np.zeros(len(o2))])
    oxygen_band = oxygen_error == 0
    breach = (tc < 35.5) | (tc > 38) | (o2 < 85) | (o2 > 96)
    close(breach, t.terminated)
    close(t.reward, t.r_thermal + .5*t.r_oxygen + .05*t.r_energy + t.r_safety)
    close(t.r_energy, -(heat**2 + oxygen**2))
    close(t.r_thermal, -temp_error**2, 1e-5)
    close(t.r_oxygen, -oxygen_error**2, 3e-5)
    recovery = overshoot = np.nan
    if scenario == 2:
        overshoot = temp_error[len(tc)//2:].max()
    elif scenario in (3, 4):
        start = 80 if scenario == 3 else 0
        recovery = first_hold(temp_band, start)
        overshoot = max(0, tc[start:].max() - 37.5)
    elif scenario == 5:
        recovery = first_hold(oxygen_band, 60)
        overshoot = max(0, o2[60:].max() - 95)
    switched = np.diff(action) != 0
    return dict(total_reward=t.reward.sum(), episode_length=len(t), survived=float(not breach.any()),
                safety_violation=float(breach.any()), tc_mean_abs_err=temp_error.mean(), tc_max_abs_err=temp_error.max(),
                tc_time_in_band=temp_band.mean(), spo2_mean_band_err=oxygen_error.mean(), spo2_max_band_err=oxygen_error.max(),
                spo2_time_in_band=oxygen_band.mean(), mean_heater_frac=heat.mean(), mean_fio2_norm=oxygen.mean(),
                control_effort=(heat+oxygen).mean(), heater_energy_kJ=(360*45/1000)*heat.sum(),
                action_switches=switched.sum(), recovery_steps=recovery, overshoot=overshoot,
                action_switch_ratio=switched.mean(), spo2_min=o2.min(), spo2_max=o2.max())


def check_summary(frame, data):
    sources = dict(mean_total_reward="total_reward", modeled_termination_avoidance_rate="survived",
                   tc_mean_abs_err="tc_mean_abs_err", tc_time_in_comfort_band="tc_time_in_band",
                   spo2_mean_band_err="spo2_mean_band_err", spo2_time_in_comfort_band="spo2_time_in_band",
                   control_effort="control_effort", heater_energy_kJ="heater_energy_kJ", mean_action_switches="action_switches")
    values = {k: frame[v].mean() for k, v in sources.items()}
    values.update(std_total_reward=frame.total_reward.std(ddof=1), safety_threshold_violations=frame.safety_violation.sum(),
                  spo2_min_observed=frame.spo2_min.min(), episodes_with_observed_recovery=frame.recovery_steps.count(),
                  mean_recovery_steps_observed=frame.recovery_steps.mean())
    for key, actual in data.items():
        close(np.nan if actual is None else actual, values[key])


def background_replay(net, records):
    raw = []
    for (scenario, policy), group in pd.DataFrame(records).groupby(["scenario", "policy"], sort=False):
        env = NeonatalIncubatorEnv(scenario=int(scenario))
        obs, _ = env.reset(seed=int(group.seed.iloc[0]))
        ctrl = RuleBasedBaselineController.from_env(env)
        targets = set(group.step)
        for tick in range(201):
            if tick in targets:
                raw.append(obs.copy())
            if tick == max(targets):
                break
            act = ctrl.select_action(obs) if policy == "baseline" else np.argmax(net.q(obs))
            obs, _, term, trunc, _ = env.step(int(act))
            assert not (term or trunc)
    return np.stack(raw)


def verify(base, incoming=False):
    net = ReferenceNetwork(ROOT / "models/ddqn_tuned_seed0/best.pt")
    sdir, vdir = base / "shap", base / "validation"
    summary = json.loads((vdir / "validation_summary.json").read_text())
    episodes = pd.read_csv(vdir / "validation_episodes.csv")
    manifest = json.loads((sdir / "shap_manifest.json").read_text())
    metadata = json.loads((sdir / "background_metadata.json").read_text())
    assert metadata == manifest["background_metadata"]
    assert manifest["explainer_used"] == summary["shap_configuration"]["shap_method_used"]
    if incoming:
        # Original metadata has no individual rows. Independently reconstruct the scheduled samples.
        steps = [[0,25,50,75,100,125,150,195],[10,30,50,75,100,125,150,170,185,198],
                 [15,35,48,55,65,75,79,85,100,130,180],[0,5,12,25,40,60,85,120,160,195],
                 [20,40,55,65,75,90,105,118,125,150,190]]
        records = [dict(scenario=s, policy=p, step=t) for s, times in enumerate(steps,1)
                   for p in ("baseline","ddqn") for t in times]
    else:
        records = metadata["records"]
        assert manifest["checkpoint_sha256"] == HASH and manifest["feature_names"] == NAMES
    for rec in records:
        rec["seed"] = metadata["random_seed"] + rec["scenario"]*100 + int(rec["policy"] == "ddqn")
    raw = background_replay(net, records)
    bg = net.normalize(raw)
    layers = net.layers(bg)
    base_q = layers[-1].mean(axis=0)
    counts = np.bincount(raw[:,9].astype(int), minlength=3).tolist()
    assert counts == [metadata["disturbance_counts"][k] for k in ("idle_0","porthole_1","oxygen_desat_2")] == [82,8,10]
    assert len(raw) == metadata["total_samples"] == 100
    for j, name in enumerate(NAMES):
        f = metadata["feature_ranges"][name]
        close([raw[:,j].min(),raw[:,j].max(),raw[:,j].mean()], [f["min"],f["max"],f["mean"]], 2e-5)
    deep = manifest["explainer_used"] == "deep"
    kernel = None if deep else kernel_reference(net, bg)
    if not incoming:
        ar = np.load(sdir / manifest["global_plot_data"]["array_file"])
        close(ar["raw"],raw); close(ar["normalized"],bg); close(ar["q"],layers[-1],1e-5)
        close(ar["greedy_actions"],layers[-1].argmax(axis=1))
        close(manifest["expected_values"],base_q,1e-5)
        full = ar["phi"]
        assert full.shape == (100,11,25)
        close(full.sum(axis=1)+base_q,layers[-1],ATOL)
        if deep:
            for i,x in enumerate(bg):
                for a in range(25):
                    close(full[i,:,a], net.deep_values(x,bg,layers,a),ATOL)
        else:
            for i, x in enumerate(bg):
                close(full[i], kernel(x), ATOL)
        glob = manifest["global_feature_importance"]
        close(abs(full[:,:,22]).mean(0),list(glob["action_specific_action_22"]["importance"].values()),1e-6)
        selected = full[np.arange(len(bg)),:,ar["greedy_actions"]]
        close(abs(selected).mean(0),list(glob["policy_level"].values()),1e-6)
    parts = [pd.read_csv(f) for f in sorted(vdir.glob("*_results.csv"))]
    traj = pd.concat(parts,ignore_index=True).copy()
    if incoming:
        group = traj.groupby(["scenario","policy"])
        episode_index = group.step.transform(lambda x:x.eq(0).cumsum()-1)
        traj["seed"] = 900000+1000*traj.scenario+episode_index
    decision_cols = ["decision_obs_"+f for f in NAMES]
    next_cols = ["next_obs_"+f for f in NAMES]
    phi_cols = ["shap_"+f for f in NAMES]
    q_cols = ["q_val_"+str(a) for a in range(25)]
    recalculated = []
    worst_phi = worst_recon = 0.
    n_explained = heat_changes = oxygen_changes = both_changes = 0
    case_rows = []
    for (s,policy,seed), part in traj.groupby(["scenario","policy","seed"]):
        part = part.sort_values("step")
        assert part.step.tolist() == list(range(201)) and part.action.iloc[0] == -1
        t=part.iloc[1:]
        decision=t[decision_cols].to_numpy(dtype=np.float32)
        after=t[next_cols].to_numpy(dtype=np.float32)
        close(decision,part[next_cols].iloc[:-1].to_numpy(),1e-6)
        env=NeonatalIncubatorEnv(scenario=int(s))
        obs,_=env.reset(seed=int(seed))
        ctrl=RuleBasedBaselineController.from_env(env)
        actions=t.action.to_numpy(int)
        q=net.q(decision)
        if policy=="ddqn":
            close(t[q_cols],q,1e-5)
            close(t.decision_q_chosen,q[np.arange(len(t)),actions],1e-5)
            close(actions,q.argmax(axis=1))
            dh,do=np.diff(actions//5)!=0,np.diff(actions%5)!=0
            heat_changes+=int(sum(dh));oxygen_changes+=int(sum(do));both_changes+=int(sum(dh & do))
            mask=t.shap_base_val.notna().to_numpy()
            if not incoming: assert mask.all()
            n_explained+=int(mask.sum())
            for i in np.flatnonzero(mask):
                a=actions[i]
                row=t.iloc[i]
                phi=row[phi_cols].to_numpy(dtype=float)
                close(row.shap_base_val,base_q[a],1e-5)
                recon=abs(phi.sum()+row.shap_base_val-row.decision_q_chosen)
                assert recon<ATOL
                worst_recon=max(worst_recon,recon)
                norm = net.normalize(decision[i])
                ref = net.deep_values(norm,bg,layers,a) if deep else kernel(norm,int(a))
                err = float(max(abs(phi-ref)))
                worst_phi = max(worst_phi, err)
                close(phi, ref, ATOL)
                order=np.argsort(abs(phi))[::-1]
                for rank,idx in enumerate(order[:3],1):
                    assert row[f"shap_top{rank}_feature"]==NAMES[idx]
                    close(row[f"shap_top{rank}_val"],phi[idx])
        else:
            assert t[q_cols+phi_cols].isna().all().all()
        for i,act in enumerate(actions):
            close(obs,decision[i],1e-6)
            if policy=="baseline": assert int(ctrl.select_action(obs))==act
            obs,r,done,truncated,_=env.step(int(act))
            close(obs,after[i],1e-6)
            close(r,t.reward.iloc[i])
            assert done==t.terminated.iloc[i] and truncated==t.truncated.iloc[i]
        close(actions//5,t.heater_idx);close(actions%5,t.o2_idx)
        close(np.take([0,.25,.5,.75,1],actions//5),t.u_heater)
        close(np.take([.21,.30,.40,.50,.60],actions%5),t.fio2_command)
        stats=recalculate(t,s)
        recorded=episodes[(episodes.scenario==s)&(episodes.policy==policy)&(episodes.seed==seed)]
        assert len(recorded)==1
        for k,v in stats.items(): close(v,recorded.iloc[0][k])
        recalculated.append(dict(scenario=s,policy=policy,**stats))
        # First five consecutive in-band post-states AFTER event window, plus actual excursion depth.
        band=t.spo2_in_comfort if s==5 else t.tc_in_comfort
        event=(t.next_obs_disturbance.ne(0) if s in (3,5) else t.step.le(20) if s==4 else t.step.gt(100) if s==2 else t.step.gt(0))
        origin={3:81,4:0,5:121}.get(s)
        hold=first_hold(band.to_numpy(),origin) if origin is not None else np.nan
        case_rows.append(dict(scenario=s,policy=policy,seed=seed,tc_min=t.Tc.min(),tc_max=t.Tc.max(),
                         oxygen_outside_steps=int((~t.spo2_in_comfort).sum()),temperature_outside_steps=int((~t.tc_in_comfort).sum()),
                         window_mean_heater=t.loc[event,"u_heater"].mean(),window_mean_fio2=t.loc[event,"fio2_command"].mean(),
                         after_window_hold_offset=hold,oxygen_min=t.SpO2.min(),ambient_final=t.Tamb.iloc[-1]))
    stats=pd.DataFrame(recalculated)
    assert len(stats)==len(episodes)==summary["evaluation_design"]["total_evaluation_episodes"]
    for name,item in summary["by_scenario"].items():
        g=stats[stats.scenario==item["scenario_id"]]
        for policy in ("ddqn","baseline"): check_summary(g[g.policy==policy],item[policy])
        close(item["ddqn_advantage_return"],g[g.policy=="ddqn"].total_reward.mean()-g[g.policy=="baseline"].total_reward.mean())
    for policy in ("ddqn","baseline"): check_summary(stats[stats.policy==policy],summary["overall"][policy])
    close(summary["overall"]["net_ddqn_improvement"],stats[stats.policy=="ddqn"].total_reward.mean()-stats[stats.policy=="baseline"].total_reward.mean())
    for loc in manifest["local_explanations"].values():
        s,step,a=loc["scenario"],loc["step"],loc["chosen_action"]
        row=traj[(traj.scenario==s)&(traj.policy=="ddqn")&(traj.seed==900000+s*1000)&(traj.step==step+1)].iloc[0]
        x=row[decision_cols].to_numpy(dtype=np.float32)
        close(x,list(loc["raw_obs"].values()),1e-6)
        close(net.normalize(x),loc["norm_obs"]);close(net.q(x),loc["all_q_values"],1e-5)
        assert a==row.action
        close(loc["expected_value"],base_q[a],1e-5)
        phi=list(loc["shap_values"].values())
        close(sum(phi)+loc["expected_value"],row.decision_q_chosen,ATOL)
        if deep: close(phi,net.deep_values(net.normalize(x),bg,layers,a),ATOL)
        if not incoming: close(phi,row[phi_cols].to_numpy(float),ATOL)
    con=manifest["contrastive_explanation"]
    a,b=con["chosen_action"],con["competing_action"]
    assert (a,b)==(9,5)
    local=next(v for v in manifest["local_explanations"].values() if v["scenario"]==5)
    x=np.asarray(local["norm_obs"],dtype=np.float32)
    close(con["delta_q"],local["all_q_values"][a]-local["all_q_values"][b],1e-6)
    close(con["delta_base"],base_q[a]-base_q[b],1e-5)
    close(con["delta_q"],con["delta_base"]+sum(con["delta_shap"].values()),ATOL)
    if deep: close(list(con["delta_shap"].values()),net.deep_values(x,bg,layers,a)-net.deep_values(x,bg,layers,b),ATOL)
    if not deep:
        all_heads = kernel(x)
        close(list(con["delta_shap"].values()), all_heads[:,a]-all_heads[:,b], ATOL)
    print(json.dumps(dict(method=manifest["explainer_used"],episodes=len(stats),transitions=len(traj)-len(stats),
                          ddqn_decisions_explained=n_explained,background_counts=counts,
                          max_csv_reconstruction_error=worst_recon,max_independent_feature_difference=worst_phi,
                          heater_switches=heat_changes,oxygen_switches=oxygen_changes,simultaneous_switches=both_changes,
                          overall=summary["overall"]),indent=2))
    return pd.DataFrame(case_rows)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"outputs/member03")
    parser.add_argument("--incoming",action="store_true",help="Read the legacy sparse validation format")
    parser.add_argument("--case-csv",type=Path)
    args=parser.parse_args()
    detail=verify(args.output_dir,args.incoming)
    if args.case_csv: detail.to_csv(args.case_csv,index=False)



