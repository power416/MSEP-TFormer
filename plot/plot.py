import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import os
from functions import *


image_path = './image/'

if not os.path.exists(image_path):
    os.mkdir(image_path)

def cal_metric(preds_path, trues_path):
    errors = []
    r2, mean, rmse, mae, mape = 0., 0., 0., 0., 0.
    if preds_path:
        preds = np.load(preds_path, allow_pickle=True)
        trues = np.load(trues_path, allow_pickle=True)
        errors = trues - preds
        r2, mean, rmse, mae, mape = calculate_metric(preds, trues)

    return preds, trues, errors, r2, mean, rmse, mae, mape

def get_center(true):
    true_max = np.ceil(max(true))
    true_min = np.floor(min(true))
    true_c = []
    while True:
        true_c.append(true_min)
        true_min += 1
        if true_min > true_max:
            break

    return true_c

def plot_img(ax, preds, trues, errors, names, units, diffs, colors, titles, r2s, means, rmses, maes, mapes, type_name):
    k = 0
    id = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)', '(m)', '(n)', '(o)', '(p)']
    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            pred = preds[i//2][j]
            true = trues[i//2][j]
            true_c = get_center(true)
            error = errors[i//2][j]
            r2 = r2s[i//2][j]
            mean = means[i//2][j]
            rmse = rmses[i//2][j]
            mae = maes[i//2][j]
            mape = mapes[i//2][j]
            # print(error)
            if i % 2 == 0:
                col.text(0., 1.1, id[k], transform=col.transAxes, fontsize=16, fontweight='bold', va='top')

                col.scatter(true, pred, s=20, marker='o', color=colors[j], edgecolors='dimgray', alpha=0.7, linewidth=0.3, label='Predicted %s' % type_name)
                col.plot(true_c, true_c, linestyle='--', color='dimgray', alpha=0.8, linewidth=1.5)
                col.grid(color=plt.cm.cividis(0.5), linestyle='--', linewidth=0.5)
                if len(pred) != 0:
                    col.text(0.1, 0.93, 'R2=%.4f' % r2, transform=col.transAxes, ha='left', va='center', bbox={'facecolor': 'whitesmoke', 'edgecolor': 'dimgrey', 'alpha': 0.5, 'boxstyle': 'round'})
                if j == 0:
                    col.set_ylabel('Predicted %s' % names[i//2] + units, fontsize=12)
                col.set_xlabel('Catalog %s' % names[i//2] + units, fontsize=12)
                if i == 0:
                    col.set_title('%s' % titles[j], fontsize=14)
                plt.tight_layout()
            else:
                y_major_locator=MultipleLocator(diffs[i//2])
                col.yaxis.set_major_locator(y_major_locator)
                col.text(0., 1.1, id[k], transform=col.transAxes, fontsize=16, fontweight='bold', va='top')
                col.hist(error, bins=35, color=colors[j], edgecolor='black')
                col.grid(axis='y',color=plt.cm.cividis(0.5), linestyle='--', linewidth=0.5)
                if len(error) != 0:
                    col.text(0.6, 0.85, 'μ=%.4f\nσ=%.4f\nmae=%.4f\nmape=%.4f' %(mean, rmse, mae, mape), transform=col.transAxes, ha='left', va='center', bbox={'facecolor': 'whitesmoke', 'edgecolor': 'dimgrey', 'alpha': 0.5, 'boxstyle': 'round'})
                if j == 0:
                    col.set_ylabel('Frequency', fontsize=12)
                col.set_xlabel(names[i//2]+' Residuals' + units, fontsize=12)
                plt.tight_layout()
            # col.xlabel('Amplitude counts', fontsize=12)
            k += 1
    plt.savefig(image_path + '%s.png' % type_name, dpi=600, bbox_inches='tight')
    plt.close()

# colors = ['turquoise', 'lightpink', 'skyblue', 'bisque']
colors = ['moccasin', 'skyblue', 'thistle', 'lightsalmon']

### m
pred_m1, true_m1, error_m1, r2_m1, mean_m1, rmse_m1, mae_m1, mape_m1 = cal_metric('../Compared/CNN/results/preds_m.npy', '../Compared/CNN/results/trues_m.npy')
pred_m2, true_m2, error_m2, r2_m2, mean_m2, rmse_m2, mae_m2, mape_m2 = cal_metric('../Compared/MagNet/results/preds_m.npy', '../Compared/MagNet/results/trues_m.npy')
pred_m3, true_m3, error_m3, r2_m3, mean_m3, rmse_m3, mae_m3, mape_m3 = cal_metric('../Compared/MFTnet/results/preds_m.npy', '../Compared/MFTnet/results/trues_m.npy')
pred_m4, true_m4, error_m4, r2_m4, mean_m4, rmse_m4, mae_m4, mape_m4 = cal_metric('../MSEP-TFormer/results/preds_m.npy', '../MSEP-TFormer/results/trues_m.npy')

pred_m = [pred_m1, pred_m2, pred_m3, pred_m4]
true_m = [true_m1, true_m2, true_m3, true_m4]
error_m = [error_m1, error_m2, error_m3, error_m4]
r2_m = [r2_m1, r2_m2, r2_m3, r2_m4]
mean_m = [mean_m1, mean_m2, mean_m3, mean_m4]
rmse_m = [rmse_m1, rmse_m2, rmse_m3, rmse_m4]
mae_m = [mae_m1, mae_m2, mae_m3, mae_m4]
mape_m = [mape_m1, mape_m2, mape_m3, mape_m4]

preds_m = [pred_m]
trues_m = [true_m]
errors_m = [error_m]
r2s_m = [r2_m]
means_m = [mean_m]
rmses_m = [rmse_m]
maes_m = [mae_m]
mapes_m = [mape_m]

fig1, ax1 = plt.subplots(2, 4, sharey='row', figsize=(14, 8))
names1 = ['Magnitude']
units1 = ''
diffs1 = [4000]
plot_img(ax1, preds_m, trues_m, errors_m, names1, units1, diffs1, colors, ['CNN', 'MagNet', 'MFTnet', 'MSEP-TFormer'], r2s_m, means_m, rmses_m, maes_m, mapes_m, 'm')

### l
pred_d1, true_d1, error_d1, r2_d1, mean_d1, rmse_d1, mae_d1, mape_d1 = cal_metric('../Compared/CNN/results/preds_d.npy', '../Compared/CNN/results/trues_d.npy')
pred_d2, true_d2, error_d2, r2_d2, mean_d2, rmse_d2, mae_d2, mape_d2 = cal_metric('../Compared/EQConvMixer/results/preds_d.npy', '../Compared/EQConvMixer/results/trues_d.npy')
pred_d3, true_d3, error_d3, r2_d3, mean_d3, rmse_d3, mae_d3, mape_d3 = cal_metric('../Compared/MFTnet/results/preds_d.npy', '../Compared/MFTnet/results/trues_d.npy')
pred_d4, true_d4, error_d4, r2_d4, mean_d4, rmse_d4, mae_d4, mape_d4 = cal_metric('../MSEP-TFormer/results/preds_d.npy', '../MSEP-TFormer/results/trues_d.npy')

pred_e1, true_e1, error_e1, r2_e1, mean_e1, rmse_e1, mae_e1, mape_e1 = cal_metric('../Compared/CNN/results/preds_e.npy', '../Compared/CNN/results/trues_e.npy')
pred_e2, true_e2, error_e2, r2_e2, mean_e2, rmse_e2, mae_e2, mape_e2 = cal_metric('../Compared/EQConvMixer/results/preds_e.npy', '../Compared/EQConvMixer/results/trues_e.npy')
pred_e3, true_e3, error_e3, r2_e3, mean_e3, rmse_e3, mae_e3, mape_e3 = cal_metric('../Compared/MFTnet/results/preds_e.npy', '../Compared/MFTnet/results/trues_e.npy')
pred_e4, true_e4, error_e4, r2_e4, mean_e4, rmse_e4, mae_e4, mape_e4 = cal_metric('../MSEP-TFormer/results/preds_e.npy', '../MSEP-TFormer/results/trues_e.npy')

pred_d = [pred_d1, pred_d2, pred_d3, pred_d4]
true_d = [true_d1, true_d2, true_d3, true_d4]
error_d = [error_d1, error_d2, error_d3, error_d4]
r2_d = [r2_d1, r2_d2, r2_d3, r2_d4]
mean_d = [mean_d1, mean_d2, mean_d3, mean_d4]
rmse_d = [rmse_d1, rmse_d2, rmse_d3, rmse_d4]
mae_d = [mae_d1, mae_d2, mae_d3, mae_d4]
mape_d = [mape_d1, mape_d2, mape_d3, mape_d4]

pred_e = [pred_e1, pred_e2, pred_e3, pred_e4]
true_e = [true_e1, true_e2, true_e3, true_e4]
error_e = [error_e1, error_e2, error_e3, error_e4]
r2_e = [r2_e1, r2_e2, r2_e3, r2_e4]
mean_e = [mean_e1, mean_e2, mean_e3, mean_e4]
rmse_e = [rmse_e1, rmse_e2, rmse_e3, rmse_e4]
mae_e = [mae_e1, mae_e2, mae_e3, mae_e4]
mape_e = [mape_e1, mape_e2, mape_e3, mape_e4]

preds_l = [pred_d, pred_e]
trues_l = [true_d, true_e]
errors_l = [error_d, error_e]
r2s_l = [r2_d, r2_e]
means_l = [mean_d, mean_e]
rmses_l = [rmse_d, rmse_e]
maes_l = [mae_d, mae_e]
mapes_l = [mape_d, mape_e]

fig2, ax2 = plt.subplots(4, 4, sharey='row', figsize=(14, 16))
names2 = ['Depth', 'Epicenter']
units2 = ' (km)'
diffs2 = [6000, 10000]
plot_img(ax2, preds_l, trues_l, errors_l, names2, units2, diffs2, colors, ['CNN', 'EQConvMixer', 'MFTnet', 'MSEP-TFormer'], r2s_l, means_l, rmses_l, maes_l, mapes_l, 'l')

### t
pred_t1, true_t1, error_t1, r2_t1, mean_t1, rmse_t1, mae_t1, mape_t1 = cal_metric('../Compared/CNN/results/preds_t.npy', '../Compared/CNN/results/trues_t.npy')
pred_t2, true_t2, error_t2, r2_t2, mean_t2, rmse_t2, mae_t2, mape_t2 = cal_metric('../Compared/Bayesian/results/preds_t.npy', '../Compared/Bayesian/results/trues_t.npy')
pred_t3, true_t3, error_t3, r2_t3, mean_t3, rmse_t3, mae_t3, mape_t3 = cal_metric('../Compared/MFTnet/results/preds_t.npy', '../Compared/MFTnet/results/trues_t.npy')
pred_t4, true_t4, error_t4, r2_t4, mean_t4, rmse_t4, mae_t4, mape_t4 = cal_metric('../MSEP-TFormer/results/preds_t.npy', '../MSEP-TFormer/results/trues_t.npy')

pred_t = [pred_t1, pred_t2, pred_t3, pred_t4]
true_t = [true_t1, true_t2, true_t3, true_t4]
error_t = [error_t1, error_t2, error_t3, error_t4]
r2_t = [r2_t1, r2_t2, r2_t3, r2_t4]
mean_t = [mean_t1, mean_t2, mean_t3, mean_t4]
rmse_t = [rmse_t1, rmse_t2, rmse_t3, rmse_t4]
mae_t = [mae_t1, mae_t2, mae_t3, mae_t4]
mape_t = [mape_t1, mape_t2, mape_t3, mape_t4]

preds_t = [pred_t]
trues_t = [true_t]
errors_t = [error_t]
r2s_t = [r2_t]
means_t = [mean_t]
rmses_t = [rmse_t]
maes_t = [mae_t]
mapes_t = [mape_t]

names3 = ['P Travel Time']
units3 = ' (s)'
diffs3 = [10000]

fig3, ax3 = plt.subplots(2, 4, sharey='row', figsize=(14, 8))
plot_img(ax3, preds_t, trues_t, errors_t, names3, units3, diffs3, colors, ['CNN', 'Bayesian', 'MFTnet', 'MSEP-TFormer'], r2s_t, means_t, rmses_t, maes_t, mapes_t, 't')
