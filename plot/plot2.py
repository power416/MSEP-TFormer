import numpy as np
import matplotlib.pyplot as plt
import os


image_path = './image2/'

if not os.path.exists(image_path):
    os.mkdir(image_path)


def plot_img(ax, x, ys, names, colors, labels, ids=['(a)', '(b)', '(c)', '(d)']):
    k = 0
    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            col2 = col.twinx()  # this is the important function

            col.set_zorder(1)  
            col2.set_zorder(0)
            col.patch.set_visible(False) 
        
            col.text(0., 1.1, ids[k], transform=col.transAxes, fontweight='bold', va='top')
            col.scatter(x, ys[k][0], s=3, marker='o', color=colors[0], edgecolors='dimgray', alpha=0.7, linewidth=0.1, label=labels[k][0])
            col.scatter(x, ys[k][1], s=3, marker='o', color=colors[1], edgecolors='dimgray', alpha=0.7, linewidth=0.1, label=labels[k][1])
            col.scatter(x, ys[k][2], s=3, marker='o', color=colors[2], edgecolors='dimgray', alpha=0.7, linewidth=0.1, label=labels[k][2])
            col.scatter(x, ys[k][3], s=3, marker='o', color=colors[3], edgecolors='dimgray', alpha=0.7, linewidth=0.1, label=labels[k][3])
            col.grid(color=plt.cm.cividis(0.5), linestyle='--', linewidth=0.5)
            if k == 0:
                col.set_ylabel('%s Error' % names[k], fontsize=9)
            elif k == 1:
                col.set_ylabel('%s Error (km)' % names[k], fontsize=9)          
            elif k == 2:
                col.set_ylabel('%s Error (km)' % names[k], fontsize=9)       
            elif k == 3:
                col.set_ylabel('%s Error (s)' % names[k], fontsize=9)

            col.set_xlabel('SNR (dB)', fontsize=9)
            col.tick_params(axis='both', labelsize=9)

            col2.hist(x, bins=15, rwidth=0.8, color='silver', alpha=0.4)
            col2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            col2.set_ylabel('Frequency', fontsize=9)
            col2.tick_params(axis='both', labelsize=9)

            col.legend(loc='upper right', borderaxespad=0.3, markerscale=2, prop={'size': 8})
            plt.tight_layout()

            k += 1
    plt.savefig(image_path + '2.png', dpi=300, bbox_inches='tight')
    plt.close()


cn_dicts = list(np.load('../Compared/CNN/results/dicts.npy', allow_pickle=True))
ma_dicts = list(np.load('../Compared/MagNet/results/dicts.npy', allow_pickle=True))
eq_dicts = list(np.load('../Compared/EQConvMixer/results/dicts.npy', allow_pickle=True))
ba_dicts = list(np.load('../Compared/Bayesian/results/dicts.npy', allow_pickle=True))
mf_dicts = list(np.load('../Compared/MFTnet/results/dicts.npy', allow_pickle=True))
ou_dicts = list(np.load('../MSEP-TFormer/results/dicts.npy', allow_pickle=True))

snrs = []
cn_error_ms = []
ma_error_ms = []
mf_error_ms = []
ou_error_ms = []

cn_error_es = []
eq_error_es = []
mf_error_es = []
ou_error_es = []

cn_error_ds = []
eq_error_ds = []
mf_error_ds = []
ou_error_ds = []

cn_error_ts = []
ba_error_ts = []
mf_error_ts = []
ou_error_ts = []

for dict in cn_dicts:
    snrs.append(dict['snr'])
    cn_error_ms.append(dict['error_m'])
    cn_error_ds.append(dict['error_d'])
    cn_error_es.append(dict['error_e'])
    cn_error_ts.append(dict['error_t'])

for dict in ma_dicts:
    ma_error_ms.append(dict['error_m'])

for dict in eq_dicts:
    eq_error_ds.append(dict['error_d'])
    eq_error_es.append(dict['error_e'])

for dict in ba_dicts:
    ba_error_ts.append(dict['error_t'])

for dict in mf_dicts:
    mf_error_ms.append(dict['error_m'])
    mf_error_ds.append(dict['error_d'])
    mf_error_es.append(dict['error_e'])
    mf_error_ts.append(dict['error_t'])

for dict in ou_dicts:
    ou_error_ms.append(dict['error_m'])
    ou_error_ds.append(dict['error_d'])
    ou_error_es.append(dict['error_e'])
    ou_error_ts.append(dict['error_t'])

error_ms = [cn_error_ms, ma_error_ms, mf_error_ms, ou_error_ms]
error_ds = [cn_error_ds, eq_error_ds, mf_error_ds, ou_error_ds]
error_es = [cn_error_es, eq_error_es, mf_error_es, ou_error_es]
error_ts = [cn_error_ts, ba_error_ts, mf_error_ts, ou_error_ts]

errors = [error_ms, error_ds, error_es, error_ts]

colors = ['moccasin', 'skyblue', 'thistle', 'lightsalmon']

label1 = ['CNN', 'MagNet', 'MFTnet', 'MSEP-TFormer']
label2 = ['CNN', 'EQConvMixer', 'MFTnet', 'MSEP-TFormer']
label3 = ['CNN', 'EQConvMixer', 'MFTnet', 'MSEP-TFormer']
label4 = ['CNN', 'Bayesian', 'MFTnet', 'MSEP-TFormer']
labels = [label1, label2, label3, label4]

names = ['Magnitude', 'Depth', 'Epicenter', 'P Travel Time']

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
plot_img(ax, snrs, errors, names, colors, labels)
