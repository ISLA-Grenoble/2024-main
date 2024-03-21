import matplotlib.pyplot as plt
import pandas as pd
from palmerpenguins import load_penguins
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_proportions_total(df):
    p_adelie = np.mean(df["species"] == "Adelie")
    p_gentoo = np.mean(df["species"] == "Gentoo")
    props = {'adelie': p_adelie, 'gentoo': p_gentoo}
    return props


def get_proportions_split(df, feature, thr):
    df_L = df[df[feature] < thr]
    p_adelie_L = np.mean(df_L["species"] == "Adelie")
    p_gentoo_L = np.mean(df_L["species"] == "Gentoo")
    df_R = df[df[feature] >= thr]
    p_adelie_R = np.mean(df_R["species"] == "Adelie")
    p_gentoo_R = np.mean(df_R["species"] == "Gentoo")
    props = {}
    props['L'] = {'adelie': p_adelie_L, 'gentoo': p_gentoo_L}
    props['R'] = {'adelie': p_adelie_R, 'gentoo': p_gentoo_R}
    return props


def get_info_gain(df, feature, thr=None):

    if thr is None:
        thr = df[feature].median()

    props_total = get_proportions_total(df)
    entropy_total = -np.sum([p * np.log(p) for p in props_total.values()])

    props_split = get_proportions_split(df, feature, thr)

    if np.prod(list(props_split['L'].values())) == 0.0:
        entropy_L = 0
    else:
        entropy_L = -np.sum([p * np.log(p) for p in props_split['L'].values()])

    if np.prod(list(props_split['R'].values())) == 0.0:
        entropy_R = 0
    else:
        entropy_R = -np.sum([p * np.log(p) for p in props_split['R'].values()])

    N = len(df)
    N_L = len(df[df[feature] < thr])
    N_R = len(df[df[feature] >= thr])
    info_gain = entropy_total - (N_L/N*entropy_L + N_R/N*entropy_R)

    return info_gain


plt.style.use("plcr_stylesheet.txt")

penguins = load_penguins().dropna()
penguins = penguins[penguins["species"] != "Chinstrap"]
df = pd.DataFrame()
df["bill_depth"] = penguins["bill_depth_mm"]
df["bill_length"] = penguins["bill_length_mm"]
df["species"] = penguins["species"]

info_gain_bill_depth = []
idxsort = df["bill_depth"].values.argsort()
bill_depth_min, bill_depth_max = df["bill_depth"].values[[idxsort[1], idxsort[-2]]]
thr_bill_depth_array = np.linspace(bill_depth_min, bill_depth_max, 100)
for thr_bill_depth in thr_bill_depth_array:
    info_gain = get_info_gain(df, "bill_depth", thr_bill_depth)
    info_gain_bill_depth.append(info_gain)
info_gain_bill_depth = np.array(info_gain_bill_depth)

info_gain_bill_length = []
idxsort = df["bill_length"].values.argsort()
bill_length_min, bill_length_max = df["bill_length"].values[[idxsort[1], idxsort[-2]]]
thr_bill_length_array = np.linspace(bill_length_min, bill_length_max, 100)
for thr_bill_length in thr_bill_length_array:
    info_gain = get_info_gain(df, "bill_length", thr_bill_length)
    info_gain_bill_length.append(info_gain)
info_gain_bill_length = np.array(info_gain_bill_length)

fig, ax = plt.subplots(figsize=(10.5, 10.0))
sel_adelie = df["species"] == "Adelie"
ax.scatter(df[sel_adelie]["bill_depth"], df[sel_adelie]["bill_length"], s=80, c='C0')
sel_gentoo = df["species"] == "Gentoo"
ax.scatter(df[sel_gentoo]["bill_depth"], df[sel_gentoo]["bill_length"], s=80, c='C1')
ax.set_xlabel("bill depth (mm)")
ax.set_ylabel("bill length (mm)")

# create new axes on the right and on the top of the current axes
divider = make_axes_locatable(ax)
ax_top = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
ax_right = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)

# make some labels invisible
ax_top.xaxis.set_tick_params(labelbottom=False)
ax_right.yaxis.set_tick_params(labelleft=False)

ax_top.plot(thr_bill_depth_array, info_gain_bill_depth, c='k', lw=2.0)
ax_right.plot(info_gain_bill_length, thr_bill_length_array, c='k', lw=2.0)
thr_bill_depth_info_max = thr_bill_depth_array[info_gain_bill_depth.argmax()]
bill_depth_info_max = max(info_gain_bill_depth)
ax_top.set_title(f"Information gain for bill depth (max of {bill_depth_info_max:.2f} at {thr_bill_depth_info_max:.2f})",
                 fontsize=16)
thr_bill_length_info_max = thr_bill_length_array[info_gain_bill_length.argmax()]
bill_length_info_max = max(info_gain_bill_length)
ax_right.set_ylabel(f"Information gain for bill length (max of {bill_length_info_max:.2f} at {thr_bill_length_info_max:.2f})",
                    fontsize=16, rotation=270, labelpad=-95)

# now determine nice limits by hand:
# ax_top.set_yticks([0, 50, 100])
# ax_right.set_xticks([0, 50, 100])

fig.savefig('decision_trees_algorithm.svg', format='svg')

