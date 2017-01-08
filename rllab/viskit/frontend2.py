import glob
import os.path as osp
import sys
import numpy as np
import json
import matplotlib
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=40)
matplotlib.rc('ytick', labelsize=40)
import matplotlib.pyplot as plt
import pprint
import seaborn as sns
import os
from copy import deepcopy
plt.ion()

def sliding_mean(data_array, window=5):
    data_array = np.array(data_array)
    new_list = []
    for i in range(len(data_array)):
        indices = list(range(max(i - window + 1, 0),
                        min(i + window + 1, len(data_array))))
        avg = 0
        for j in indices:
            avg += data_array[j]
        avg /= float(len(indices))
        new_list.append(avg)

    return np.array(new_list)

def recursive_glob(path, ext, depth=1):
    files = []
    mid = ''
    for i in range(depth):
        pattern = osp.join(path, '%s*.%s'%(mid, ext))
        print('Globbing %s...'%pattern)
        files += glob.glob(pattern, recursive=True)
        mid += '**/'
    return files

color_maps = dict(
    qprop="red",
    qvpg="maroon",
    trpo="blue",
    vpg="teal",
    ddpg="green",
)

def get_default_linestyle(exp_names):
    linestyle_map = dict()
    for exp_name in exp_names:
        algo = exp_name.split('-al-',1)[1].split('--',1)[0]
        if algo in ['qprop', 'qvpg']:
            linestyle_map[exp_name] = 'solid'
        else:
            linestyle_map[exp_name] = 'dashdot'
    return linestyle_map

def get_default_color(exp_names):
    exps = dict()
    color_map = dict()
    for exp_name in exp_names:
        algo = exp_name.split('-al-',1)[1].split('--',1)[0]
        batch_size = int(exp_name.split('-v1-',1)[1].split('-',1)[0])
        attr = '%05d'%batch_size
        if algo == 'qprop' or algo == 'qvpg':
            if 'adapt1' in exp_name: attr += '-c'
            elif 'adapt2' in exp_name: attr += '-a'
        elif algo == 'ddpg':
            attr = exp_name
        if algo in exps:
            exps[algo].append((attr, exp_name))
        else:
            exps[algo] = [(attr, exp_name)]
    for algo, items in exps.items():
        if algo == 'ddpg': print(items)
        items = sorted(items, key=lambda x: x[0])
        colors = sns.dark_palette(color_maps[algo], (len(items)+1), reverse=False)
        colors += sns.light_palette(color_maps[algo], (len(items)), reverse=True)
        for i, item in enumerate(items):
            color_map[item[1]] = colors[i*2+1]
    return color_map

def get_default_legend(legend_map, exp_names):
    for exp_name in exp_names:
        if exp_name in legend_map: continue
        algo = exp_name.split('-al-',1)[1].split('--',1)[0]
        batch_size = int(exp_name.split('-v1-',1)[1].split('-',1)[0])
        if algo == 'qprop':
            if 'adapt1' in exp_name: algo_name = 'TR-c-Q-Prop'
            elif 'adapt2' in exp_name: algo_name = 'TR-a-Q-Prop'
            else: algo_name = 'TR-Q-Prop'
            legend_map[exp_name] = '%s-%05d'%(algo_name, batch_size)
        elif algo == 'qvpg':
            if 'adapt1' in exp_name: algo_name = 'v-c-Q-Prop'
            elif 'adapt2' in exp_name: algo_name = 'v-a-Q-Prop'
            else: algo_name = 'v-Q-Prop'
            legend_map[exp_name] = '%s-%05d'%(algo_name, batch_size)
        elif algo == 'ddpg':
            algo_name = algo.upper()
            legend_map[exp_name] = '%s'%(algo_name)
        else:
            algo_name = algo.upper()
            legend_map[exp_name] = '%s-%05d'%(algo_name, batch_size)
    return legend_map

def load_csv(files):
    data = dict()
    for f in files:
        print('Loading %s...'%f)
        json_file = osp.join(osp.dirname(f), 'params.json')
        if not osp.isfile(json_file):
            print('WARNING: json file %s does not exist. Skipping %s...'%(
                json_file, f))
            continue
        with open(json_file) as json_io:
            json_data = json.load(json_io)
        exp_name = json_data["exp_name"]
        json_args = json_data["json_args"]
        datum = np.genfromtxt(f, dtype=float, delimiter=',', names=True)
        assert exp_name not in data, 'ERROR: exp_name=%s loaded twice!'%exp_name
        keys = datum.dtype.names
        data[exp_name] = {k: datum[k] for k in keys}
        data[exp_name]['_keys'] = keys
        data[exp_name]['_json_args'] = json_args
        if 'al-ddpg' in exp_name:
            batch_size = int(exp_name.split('-v1-',1)[1].split('--',1)[0])
        else:
            batch_size = json_args["algo"]["batch_size"]
        data[exp_name]['_batch_size'] = batch_size
        print('Loaded exp_name=%s, batch_size=%d, %d columns, %d rows.'%(exp_name, data[exp_name]["_batch_size"], len(keys), datum.size))
    return data

def plot_data(data, paths, window_size=1, normalize_batch_size=True,
        base_batch_size=1000, keys=['AverageReturn'], legend_map=dict(),
        x_cutoff=1e10, reorder=None, color_palette='coolwarm_r',
        use_default_legend=False, use_default_color=False, threshold=None,
        ylim=None, xlim=None, use_default_linestyle=True, n_per_line=None):

    default_linestyle = {"linewidth":3, "markeredgewidth":5,
                }
    #fillstyle = {'alpha':0.2, 'linewidth':4, 'antialiased':True}
    fontsize = 20
    prop = {'size':fontsize-4, 'weight': 'heavy'}
    legendstyle = {'loc':'lower right',
        'ncol':1, 'fancybox':True, 'shadow':True,
        'labelspacing':0.2, 'prop':prop}
    exp_names = data.keys()
    if use_default_legend: legend_map = get_default_legend(legend_map, exp_names)
    def map_exp_name(x):
        if x in legend_map: return legend_map[x]
        else: return x
    sorted_exp_names = sorted(exp_names, key=map_exp_name)
    for i, key in enumerate(keys):
        fig = plt.figure(i)
        valid_exp_names = [exp_name for exp_name in sorted_exp_names if key in data[exp_name]]
        if reorder is not None:
            assert(len(reorder) == len(valid_exp_names))
            valid_exp_names = [valid_exp_names[i] for i in reorder]
        ax = plt.subplot(111)
        if use_default_color:
            color_map = get_default_color(valid_exp_names)
        else:
            ax.set_color_cycle(sns.color_palette(color_palette,len(valid_exp_names)))
        if use_default_linestyle:
            linestyle_map = get_default_linestyle(valid_exp_names)
        for j, exp_name in enumerate(valid_exp_names):
            xkey = "Iteration"
            if 'ddpg' in exp_name: xkey = "Epoch"
            x = data[exp_name][xkey].astype(int)
            window=window_size
            if normalize_batch_size:
                x = x.astype(float)/base_batch_size*data[exp_name]["_batch_size"]
                x = x.astype(int)
                window = int(float(window)*base_batch_size/data[exp_name]["_batch_size"])
            y = data[exp_name][key]
            ind = x <= x_cutoff
            x = x[ind]
            y = y[ind]
            if window_size > 1: y = sliding_mean(y, window=window)
            label = map_exp_name(exp_name)
            print('INFO[%s]: max %s=%f'%(label, key, y.max()))
            if threshold is not None:
                ind = y > threshold
                x_crossed = x[ind]
                if len(x_crossed) > 0:
                    print('INFO[%s]: cross %s=%f at %d'%(label, key, threshold, x_crossed[0]))
            linestyle = deepcopy(default_linestyle)
            if use_default_color: linestyle['color'] = color_map[exp_name]
            if use_default_linestyle: linestyle['linestyle'] = linestyle_map[exp_name]
            if n_per_line is not None:
                div = float(len(x))/n_per_line
                if div >=2:
                    x = x[::int(div)]
                    y = y[::int(div)]
            print('INFO[%s]: plotting %d points.'%(label, len(x)))
            plt.plot(x, y, label=label.replace('_','-'), **linestyle)
            #plt.fill_between(x, y-std, y+std, edgecolor=colors[i], facecolor=colors[i], **fillstyle)
        plt.xlabel('Episodes', fontsize=fontsize, fontweight='bold')
        plt.ylabel(key, fontsize=fontsize, fontweight='bold')
        if ylim is not None:
            plt.ylim(ylim)
        if xlim is not None:
            plt.xlim(xlim)
        plt.setp(ax.get_xticklabels(), fontsize=20)
        plt.setp(ax.get_yticklabels(), fontsize=20)
        plt.legend(**legendstyle).draggable(True)
        plt.draw()
        sys.stdout.flush()
        plt.tight_layout()
        fig.savefig('%s/Desktop/fig-%d.png'%(os.getenv('HOME'),i))
    print('Ctrl-C to exit.')
    while 1:
        plt.pause(.1)

def main(argv=None):

    if len(sys.argv) == 1:
        print('Usage %s [json config file] <path1> [<path2> <path3> ...]'%sys.argv[0])
        sys.exit(0)

    configs = dict(
        paths = [],
    )

    if osp.isfile(sys.argv[1]) and osp.splitext(sys.argv[1])[1] == '.json':
        with open(sys.argv[1]) as f:
            load_configs = json.load(f)
            configs.update(load_configs)
            print('Loaded configs from %s: %s'%(sys.argv[1], pprint.pformat(configs,indent=1)))
        paths = sys.argv[2:] + configs["paths"]
    else:
        paths = sys.argv[1:]

    files = []
    for path in paths:
        if osp.isfile(path) and osp.splitext(path)[1] == '.csv': files.append(path); continue
        files += recursive_glob(path=path, ext='csv', depth=2)
    files = sorted(list(set(files)))
    print('Located %d csv files, %s...'%(len(files),pprint.pformat(files,indent=1)))

    data = load_csv(files)

    plot_data(data, **configs)

if __name__ == '__main__':
    main()
