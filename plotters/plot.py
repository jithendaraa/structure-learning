import wandb, pdb
import matplotlib.pyplot as plt
import numpy as np

colors = ['orange', 'blue', 'red', 'purple', 'yellow', 'black']

api = wandb.Api(timeout=30)

# Project is specified by <entity/project-name>
runs = api.runs("structurelearning/structure-learning")

def get_plotting_data(reqd_runs, reqd_keys):
    seed_data = {}
    for key in reqd_keys: seed_data[key] = []

    for run in reqd_runs:
        plotting_data = run.scan_history(reqd_keys, max_steps)

        for key in reqd_keys:
            seed_data[key].append([data[key] for data in plotting_data])
        
    
    for key in reqd_keys:
        seed_data[key] = [x for x in seed_data[key] if x]

    return seed_data

def get_reqd_runs(exp_config):
    reqd_runs = []
    for run in runs:
        reqd_run = True
        for k,v in exp_config.items():
            if run.config[k] != v: 
                reqd_run = False
                break
        if reqd_run is False: continue
        
        reqd_runs.append(run)   # This is a required run
    return reqd_runs

# ! 'linear_decoder_bcd'
exp_config = {
    'exp_name': 'linear_decoder_bcd',
    'fix_decoder': False,
    'learn_L': 'partial',
    'Z_KL': True,
    'decoder_layers': 'linear',
    'train_loss': 'mse',
    'num_samples': 500
}

reqd_keys = ['_step', 'Evaluations/SHD', 'Evaluations/AUROC', 'Evaluations/SHD_C', 'ELBO']
max_steps = 20000


def plot_data(key, seed_data, ax, label = None, color='blue'):
    if label is None: label = ''
    x_axis = np.array(seed_data['_step'][0])
    y_axis_seeds = np.array(seed_data[key])
    yaxis = y_axis_seeds.mean(0)
    fill = y_axis_seeds.std(0)
    ax.plot(x_axis, yaxis, label=label, color=color)
    ax.fill_between(x_axis, yaxis - fill, yaxis + fill, alpha=0.3, color=color)

def set_axes_details(ax, xlabel, ylabel, title=None, set_legend=True):
    if xlabel: ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # if set_legend: ax.legend()
    if title: ax.title.set_text(f"{title}")
    ax.grid()
    


# ! linear decoder dibs, observational data, supervised.
exp_config = {
    'exp_name': 'linear_decoder_dibs',
    'supervised': True,
    'num_samples': 'partial',
    'num_updates': 4000,
    'steps': 14000,
    'across_interv': False,
    'num_samples': 300,
    'obs_data': 300
}

reqd_keys = ['_step', 'Evaluations/AUROC (empirical)', 
            'Evaluations/AUROC (marginal)', 'Evaluations/MEC or GT recovery %', 
            'Evaluations/Exp. SHD (Empirical)', 'Evaluations/Exp. SHD (Marginal)', 
            'Evaluations/CPDAG SHD']
max_steps = 4000


def plot_metrics_for_nodes(num_nodes, proj_dims, reqd_keys, exp_config):
    num_subplots = len(reqd_keys) - 1
    h = int(np.sqrt(num_subplots))
    w = int(num_subplots / h)
    f, axes = plt.subplots(h, w, figsize = (12, 5) )

    idxs = []
    for i in range(h):
        for j in range(w):
            idxs.append((i, j))
    exp_config['num_nodes'] = num_nodes

    exp_config['exp_edges'] = 0.5
    reqd_runs = get_reqd_runs(exp_config)
    print(f'Fetched {len(reqd_runs)} runs')
    plotting_data = get_plotting_data(reqd_runs, reqd_keys)

    for i in range(num_subplots):
        ax = axes[idxs[i]]
        key = reqd_keys[1:][i]
        plot_data(key, plotting_data, ax, label = 'ER-1', color='blue')

    exp_config['exp_edges'] = 1.0
    reqd_runs = get_reqd_runs(exp_config)
    print(f'Fetched {len(reqd_runs)} runs')
    plotting_data = get_plotting_data(reqd_runs, reqd_keys)

    for i in range(num_subplots):
        ax = axes[idxs[i]]
        key = reqd_keys[1:][i]
        plot_data(key, plotting_data, ax, label = 'ER-2', color='green')
        set_axes_details(ax, xlabel=None, ylabel=key.split('/')[-1])

    lines_labels = [axes[idxs[i]].get_legend_handles_labels() for i in range(num_subplots)]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    f.legend(lines[:2], labels[:2])
    f.text(0.5, 0.02, 'Num. Iterations', ha='center')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    f.savefig(f'plot_n{num_nodes}p{proj_dims}_er12.png')

plot_metrics_for_nodes(4, 10, reqd_keys, exp_config)
plot_metrics_for_nodes(5, 10, reqd_keys, exp_config)
plot_metrics_for_nodes(10, 20, reqd_keys, exp_config)
plot_metrics_for_nodes(20, 50, reqd_keys, exp_config)
