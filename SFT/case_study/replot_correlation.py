"""Re-render the case-study correlation figures from the archived vector PDFs.

This is useful when the per-step ``selection_records.json`` files behind
``result.ipynb`` (Llama-3.2-1B runs) are not at hand.  The correlation panels
of the paper figure are plain matplotlib vector PDFs, so the plotted per-block
statistics (mean, 25th and 75th percentile of the Spearman correlation for each
layer type) can be read back exactly from their content streams.  This script does
that and re-draws the panels with the same style as ``result.ipynb`` (cell 7), so
that cosmetic changes (here: the label ``rho`` -> ``rho_S``) can be applied without
the raw records.  Magnitude panels and the legend are untouched.

Usage:
    python SFT/case_study/replot_correlation.py \
        --src Paper/ICLR/Figures/SFT/case_study --out /path/to/outdir [--png]
"""
import argparse
import glob
import json
import os
import re
import zlib

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- must match result.ipynb ---------------------------------------------------
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12
plt.style.use('seaborn-v0_8-darkgrid')

LAYER_TYPES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
type_colors = {
    'q_proj': '#E41A1C', 'k_proj': '#FF7F00', 'v_proj': '#FFD700',
    'o_proj': '#984EA3', 'gate_proj': '#377EB8', 'up_proj': '#4DAF4A',
    'down_proj': '#A65628',
}
attn_markers = {'q_proj': 'o', 'k_proj': 's', 'v_proj': '^', 'o_proj': 'v'}
mlp_markers = {'gate_proj': 'D', 'up_proj': 'P', 'down_proj': '*'}
all_markers = {**attn_markers, **mlp_markers}

XLIM = (-0.75, 15.75)   # 16 blocks with the default 5 % axes margin
YLIM = (-0.15, 1.05)    # ax.set_ylim in result.ipynb


def style_ax(ax):
    ax.grid(True, alpha=0.3, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color('black')
    ax.tick_params(axis='both', which='major', labelsize=20, direction='out', length=5)


# ---- PDF recovery ----------------------------------------------------------------
def _hex2rgb(h):
    return tuple(round(int(h[i:i + 2], 16) / 255, 4) for i in (1, 3, 5))


COLOR2TYPE = {_hex2rgb(v): k for k, v in type_colors.items()}


def _content_stream(pdf):
    d = open(pdf, 'rb').read()
    cid = re.search(rb'/Contents (\d+) 0 R', d).group(1)
    m = re.search(cid + rb' 0 obj\s*<<.*?>>\s*stream\r?\n(.*?)\r?\nendstream', d, re.S)
    return zlib.decompress(m.group(1)).decode('latin1')


def _parse_paths(content):
    """Minimal PDF content-stream walker: returns filled/stroked paths with state."""
    st = {'rg': None, 'RG': None, 'w': 1.0, 'gs': None}
    stack, nums, path, paths, rects = [], [], [], [], []
    last_name = None
    for t in content.split():
        try:
            nums.append(float(t))
            continue
        except ValueError:
            pass
        if t == 'q':
            stack.append(dict(st))
        elif t == 'Q':
            st = stack.pop()
        elif t == 'm':
            path = [(nums[-2], nums[-1])]
        elif t in ('l', 'c'):
            path.append((nums[-2], nums[-1]))
        elif t == 're':
            rects.append(tuple(nums[-4:]))
        elif t == 'rg':
            st['rg'] = tuple(round(x, 4) for x in nums[-3:])
        elif t == 'RG':
            st['RG'] = tuple(round(x, 4) for x in nums[-3:])
        elif t == 'w':
            st['w'] = nums[-1]
        elif t == 'gs':
            st['gs'] = last_name
        elif t in ('S', 'B', 'f', 'f*', 'b', 'n') and path:
            paths.append({'op': t, 'pts': list(path), **st})
            path = []
        if t.startswith('/'):
            last_name = t
        nums = []
    return paths, rects


def recover_stats(pdf):
    """Return {layer_type: {'mean': [16], 'p25': [16], 'p75': [16]}} from a
    correlation_*.pdf produced by result.ipynb."""
    paths, rects = _parse_paths(_content_stream(pdf))
    x0, y0, w, h = [r for r in rects if r[2] > 300 and r[3] > 200][0]   # axes clip box

    def to_x(X):
        return XLIM[0] + (X - x0) / w * (XLIM[1] - XLIM[0])

    def to_y(Y):
        return YLIM[0] + (Y - y0) / h * (YLIM[1] - YLIM[0])

    rec = {t: {} for t in LAYER_TYPES}
    for p in paths:
        pts = p['pts']
        if p['op'] == 'S' and len(pts) == 16 and abs(p['w'] - 2.2) < 1e-6 and p['RG'] in COLOR2TYPE:
            t = COLOR2TYPE[p['RG']]                       # mean curve (linewidth 2.2)
            assert [round(to_x(X)) for X, _ in pts] == list(range(16))
            rec[t]['mean'] = [to_y(Y) for _, Y in pts]
        elif p['op'] == 'B' and len(pts) == 34 and p['gs'] == '/A4' and p['rg'] in COLOR2TYPE:
            t = COLOR2TYPE[p['rg']]                       # fill_between polygon (alpha .12)
            lower, upper = pts[1:17], pts[18:34][::-1]
            assert pts[17] == pts[18] and pts[0] == pts[33]
            assert [round(to_x(X)) for X, _ in lower] == list(range(16))
            assert [round(to_x(X)) for X, _ in upper] == list(range(16))
            rec[t]['p25'] = [to_y(Y) for _, Y in lower]
            rec[t]['p75'] = [to_y(Y) for _, Y in upper]
    for t in LAYER_TYPES:
        assert set(rec[t]) == {'mean', 'p25', 'p75'}, (pdf, t, sorted(rec[t]))
        assert all(l - 1e-6 <= m <= u + 1e-6
                   for l, m, u in zip(rec[t]['p25'], rec[t]['mean'], rec[t]['p75'])), (pdf, t)
    return rec


# ---- plotting (mirrors result.ipynb, cell 7, correlation block) ---------------------
annot_bbox = dict(boxstyle='round,pad=0.35', facecolor='white',
                  edgecolor='#CC0000', linewidth=1.2, alpha=0.95)
annot_arrow = dict(arrowstyle='->', color='#CC0000', lw=1.8,
                   connectionstyle='arc3,rad=-0.2',
                   shrinkA=0, shrinkB=14)
circle_color = '#CC0000'


def plot_correlation(rec, out_pdf, png=False, label=r'$\rho_S$'):
    blocks = list(range(16))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    for t in LAYER_TYPES:
        ls = '-' if t in attn_markers else '--'
        ax.plot(blocks, rec[t]['mean'], color=type_colors[t], linewidth=2.2,
                marker=all_markers[t], markersize=6, linestyle=ls)
        ax.fill_between(blocks, rec[t]['p25'], rec[t]['p75'], color=type_colors[t], alpha=0.12)

    down_corr = rec['down_proj']['mean']
    peak_block = int(np.argmax(down_corr))
    peak_val = down_corr[peak_block]
    ax.scatter([peak_block], [peak_val], s=450, facecolors='none',
               edgecolors=circle_color, linewidths=2.5, zorder=10)
    ax.annotate(
        f'{label} = {peak_val:.2f}',
        xy=(peak_block, peak_val),
        xytext=(peak_block + 4, peak_val - 0.18),
        fontsize=16, fontfamily='serif',
        color='#CC0000',
        arrowprops=annot_arrow,
        ha='center', va='top',
        bbox=annot_bbox,
    )

    ax.set_xlabel('Transformer Block', fontsize=24)
    ax.set_ylabel('Rank Correlation', fontsize=24)
    ax.set_xticks(blocks)
    ax.set_ylim(*YLIM)
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.3)
    style_ax(ax)
    plt.tight_layout()
    fig.savefig(out_pdf, format='pdf', bbox_inches='tight', facecolor='white')
    if png:
        fig.savefig(out_pdf[:-4] + '.png', format='png', dpi=110, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return peak_block, peak_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='dir with the archived correlation_*.pdf')
    ap.add_argument('--out', required=True, help='output dir')
    ap.add_argument('--png', action='store_true', help='also write PNG previews')
    ap.add_argument('--dump-json', default=None, help='write recovered statistics here')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    stats = {}
    for pdf in sorted(glob.glob(os.path.join(args.src, 'correlation_*.pdf'))):
        key = os.path.basename(pdf)[len('correlation_'):-4]
        rec = recover_stats(pdf)
        stats[key] = rec
        out_pdf = os.path.join(args.out, os.path.basename(pdf))
        pb, pv = plot_correlation(rec, out_pdf, png=args.png)
        print(f'{key}: down_proj peak at block {pb}, rho_S = {pv:.3f} -> {out_pdf}')
    if args.dump_json:
        json.dump(stats, open(args.dump_json, 'w'), indent=1)
        print('recovered statistics written to', args.dump_json)


if __name__ == '__main__':
    main()
