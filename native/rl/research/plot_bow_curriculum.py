"""Plot the completed Bow curriculum from saved outcomes, without rerunning fits."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    read = lambda path: json.loads((args.artifacts / path).read_text())
    names = ['raw-s73', 'aim-s73', 'joint-s73', 'seeded-s73', 'seeded-s173', 'small-step-s73', 'small-step-s173']
    fits = {name: read(name + '/result.json') for name in names}
    evaluation = {name: read(name + '/final-evaluation.json') for name in names}
    transfer = {name: read(name + '/result.json') for name in
                ['final-plus16', 'final-minus16', 'final-transfer', 'final-moving',
                 'small-step-plus16', 'small-step-minus16', 'small-step-transfer', 'small-step-moving']}
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout='constrained')
    colors = ['#9a9a9a', '#36788e', '#cd772e', '#9f6979', '#bd939e', '#28724f', '#699837']
    labels = ['Raw PPO', 'Assisted\naiming', 'Aim seed\n+ hold bias', 'Teacher seed\n3e-4\nseed 73', 'Teacher seed\n3e-4\nseed 173', 'Teacher seed\n3e-5\nseed 73', 'Teacher seed\n3e-5\nseed 173']
    ax = axes[0, 0]
    values = [fits[n]['final_successes'] / fits[n]['final_trials'] for n in names]
    bars = ax.bar(range(len(names)), np.array(values) * 100, color=colors)
    bars[1].set_hatch('//')
    for i, name in enumerate(names):
        r = fits[name]
        ax.text(i, values[i]*100+1, f"{r['final_successes']}/{r['final_trials']}", ha='center')
    ax.axhline(90, ls='--', color='#777', lw=1)
    ax.tick_params(axis='x', labelsize=8)
    ax.set(xticks=range(len(names)), xticklabels=labels, ylim=(0, 112),
           ylabel='Successful hits (%)', title='A. Final fixed-layout trials; hatching means timing assistance')
    ax = axes[0, 1]
    for name, label, color in zip(names, labels, colors):
        rows = [json.loads(x) for x in (args.artifacts/name/'metrics.jsonl').read_text().splitlines()]
        ax.plot([x['step']/1000 for x in rows], [x['success_rate']*100 for x in rows], color=color, label=label.replace('\n', ' '))
    ax.set(xlabel='Learner decisions in each stage (thousands)', ylabel='Last 256 training trials: hits (%)',
           ylim=(0, 105), title='B. Learning dynamics; starting weights differ between stages')
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax = axes[1, 0]
    rows = [evaluation['small-step-s73'], transfer['small-step-minus16']['games'],
            transfer['small-step-plus16']['games'], transfer['small-step-moving']['games'],
            [x for x in transfer['small-step-transfer']['games'] if x['task']['map_id']==0 and x['task']['side']==1],
            [x for x in transfer['small-step-transfer']['games'] if x['task']['map_id']==3 and x['task']['side']==0]]
    for i, games in enumerate(rows):
        selected = [x for x in games if x['mode']=='sample']
        hits = sum(x['success'] for x in selected)
        rate = hits/len(selected)*100
        ax.bar(i, rate, color='#28724f' if i<3 else '#cd772e')
        ax.text(i, rate+1, f'{hits}/{len(selected)}', ha='center')
    ax.set(xticks=range(len(rows)), xticklabels=['Original', 'Farther\nstart', 'Closer\nstart', 'Moving\ntarget', 'Other\nside', 'Other\nmap'],
           ylim=(0, 112), ylabel='Successful hits (%)', title='C. Seed 73 at 3e-5: nearby transfer works, wider transfer fails')
    ax = axes[1, 1]
    initial = read('seeded-s73/initial-evaluation.json')
    for data, label, color in [(initial, 'Learned teacher initialization', '#9a9a9a'),
                              (evaluation['seeded-s73'], 'PPO 3e-4, seed 73', '#9f6979'),
                              (evaluation['small-step-s73'], 'PPO 3e-5, seed 73', '#28724f'),
                              (evaluation['small-step-s173'], 'PPO 3e-5, seed 173', '#699837')]:
        games = [x for x in data if x['mode']=='sample']
        ticks = np.arange(0, 241, 4)
        rate = [sum(x['success'] and x['ticks']<=t for x in games)/len(games)*100 for t in ticks]
        ax.step(ticks/60, rate, where='post', label=label, color=color)
    ax.set(xlabel='Seconds after control starts', ylabel='Attempts with a hit by this time (%)',
           ylim=(0, 105), title='D. Time to hit includes failed attempts in the denominator')
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Bow curriculum: a learned shot, with clear limits', fontsize=18, fontweight='bold')
    fig.supxlabel('The teacher is absent during seeded PPO and evaluation. These are short two-player drills, not full-game strength tests.', fontsize=10)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(args.output/f'overview.{extension}', dpi=170)
    summary = dict(plan=read('plan.json'), fits=fits, evaluations=evaluation, transfer=transfer,
                   contracts=read('contracts/result.json'), teacher_alignment=read('teacher-alignment/result.json'),
                   baseline_parity=read('baseline-parity/result.json'), learned_parity=read('learned-parity/result.json'),
                   shot_timing=read('shot-timing/result.json'), stable_parity=read('stable-parity/result.json'),
                   lr_contract=read('lr-contract/result.json'), checkpoints=read('checkpoints.json'))
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
