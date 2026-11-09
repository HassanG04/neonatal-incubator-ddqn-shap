"""Derive defense figures and slide data from the frozen validation CSVs."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'presentation'
OUT.mkdir(exist_ok=True)
v=ROOT/'outputs/member03/validation'
df=pd.read_csv(v/'normal_scenario_results.csv')
selected=df[(df.seed==901000)&(df.step>0)]
fig,axes=plt.subplots(2,1,figsize=(12,5.2),sharex=True)
for policy,color in [('ddqn','#087e8b'),('baseline','#cd692d')]:
    d=selected[selected.policy==policy].sort_values('step')
    axes[0].step(d.step-1,d.u_heater,where='post',label=policy.upper(),color=color,alpha=.85,lw=1.2)
    axes[1].step(d.step-1,d.fio2_command,where='post',color=color,alpha=.85,lw=1.2)
axes[0].set_ylabel('Heater fraction')
axes[1].set_ylabel('Commanded FiO2')
axes[1].set_xlabel('Decision step (45 simulated seconds per step)')
axes[0].legend(ncol=2,loc='upper right')
for ax in axes: ax.grid(alpha=.2)
fig.suptitle('Normal scenario, paired seed 901000: actuator commands')
fig.tight_layout()
fig.savefig(OUT/'actuator_switching.png',dpi=180)
plt.close(fig)
selected[['policy','seed','step','u_heater','fio2_command','action']].to_csv(OUT/'actuator_switching_source.csv',index=False)
summary=json.loads((v/'validation_summary.json').read_text())
manifest=json.loads((ROOT/'outputs/member03/shap/shap_manifest.json').read_text())
(OUT/'evidence.json').write_text(json.dumps({'summary':summary,'shap':manifest},indent=2),encoding='utf8')
print('Created actuator plot and source CSV from 400 recorded decisions.')
