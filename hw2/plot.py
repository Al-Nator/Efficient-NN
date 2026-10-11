import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


folder = Path('results/short')
rows = list(csv.DictReader((folder / 'runs.csv').open()))
precision = {r['precision'] for r in rows}
if len(precision) != 1:
    raise ValueError('cannot combine runs with different precision')
modes = ['random', 'target', 'proxy']
summary = {'precision': precision.pop()}
for mode in modes:
    selected = sorted([r for r in rows if r['mode'] == mode], key=lambda r: int(r['seed']))
    if [int(r['seed']) for r in selected] != [0, 1, 2]:
        raise ValueError('expected three seeds per mode')
    summary[mode] = {}
    for key in ['error', 'selection_seconds', 'train_seconds']:
        values = np.array([float(r[key]) for r in selected])
        summary[mode][key] = {'mean': values.mean(), 'std': values.std(ddof=1)}

speed = []
for seed in range(3):
    times = {r['mode']: float(r['selection_seconds']) for r in rows if int(r['seed']) == seed}
    speed.append(times['target'] / times['proxy'])
summary['speedup'] = {'mean': float(np.mean(speed)), 'std': float(np.std(speed, ddof=1))}
summary['std_ddof'] = 1
(folder / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 18})
fig, ax = plt.subplots(figsize=(6, 4.2))
means = [summary[m]['error']['mean'] for m in modes]
std = [summary[m]['error']['std'] for m in modes]
ax.bar(modes, means, yerr=std, capsize=6, color=['#999999', '#555555', '#bbbbbb'])
ax.set_ylabel('Test error, %')
ax.set_ylim(0, 60)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(folder / 'error.png', dpi=200)
fig.savefig(folder / 'error.pdf')
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4.2))
x = np.arange(3)
select = np.array([summary[m]['selection_seconds']['mean'] / 60 for m in modes])
train = np.array([summary[m]['train_seconds']['mean'] / 60 for m in modes])
ax.bar(x, select, color='#555555', label='Отбор')
ax.bar(x, train, bottom=select, color='#bbbbbb', label='Обучение target')
ax.set_xticks(x, modes)
ax.set_ylabel('Время, мин')
ax.set_ylim(0, 8)
ax.legend(frameon=False, fontsize=15)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(folder / 'time.png', dpi=200)
fig.savefig(folder / 'time.pdf')
plt.close(fig)
print(json.dumps(summary, indent=2))
