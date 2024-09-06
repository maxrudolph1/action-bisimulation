import tensorboard_reducer as tbr
import matplotlib.pyplot as plt
import glob

plt.rcParams.update({
#     "text.usetex": True,
    "font.family": "Times",
    # "axes.labelsize": 30,
    # "axes.titlesize": 30,
    # "xtick.labelsize": 30,
    # "ytick.labelsize": 30,
    # "legend.fontsize": 30,
})

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink', 'tab:brown', 'tab:cyan', 'tab:olive', 'tab:gray']
labels = [
    # 'online/neweps_nooracle_convenc_e1.0_nogoal',
    'online/baseline',
    # 'online/singlestep_e1.0_norelu',
    # 'online/auto_nofc_goal',
    'online/auto_nofc_goal_sidetune',
    'online/auto_nofc_goal_sidetune_b16',
    # 'online/singlestep_e1.0',
    'online/singlestep_e1.0_sidetune',
    'online/singlestep_e1.0_sidetune_b16',
    'online/ftarget_e1.0_sidetune',
    'online/ftarget_e1.0_sidetune_b16',
    # 'online/neweps_ftarget_convenc_e1.0_nogoal'
]

for label, color in zip(labels, colors):
    dirs = glob.glob(f"experiments/{label}/*")
    events_dict = tbr.load_tb_events(dirs, strict_steps=False, min_runs_per_step=1)
    success_rate = events_dict['success_rate']
    success_rate = success_rate[success_rate.index <= 1000000]
    success_rate = success_rate.interpolate(method="index")
    mean = success_rate.mean(axis=1)
    std = success_rate.std(axis=1)
    if "baseline" in label:
        new_label = "No pretraining"
    else:
        eps = label.split("_")[-2][1:]
        new_label = rf"$\epsilon = {eps}$"
    new_label = label
    # if "ftarget" in label:
    #     new_label += " ftarget"
    # if "doublef" in label:
    #     new_label += " doublef"
    print(new_label)
    plt.plot(mean, label=new_label, color=color)
    plt.fill_between(mean.index, mean - std, mean + std, facecolor=color, alpha=0.5)

plt.legend(loc="lower right")
plt.xlabel("Timestep")
plt.ylabel("Success rate")
plt.ylim(0, 1)
plt.xlim(0, 1000000)
plt.title("Online Fine-Tuning Performance")
plt.show()
