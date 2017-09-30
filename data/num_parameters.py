import numpy as np
import matplotlib.pyplot as plt

nums = [5184, 64, 221184, 128, 884736, 256, 1769472, 256, 3538944, 512, 7077888, 512, 7077888, 512, 7077888, 512, 33554432, 4096, 16777216, 4096]

params_with_biases = [nums[i] + nums[i+1] for i in np.arange(0,len(nums),2)]
print(params_with_biases)
print('all', np.sum(nums))

def plot_it():
    labels = ('conv1a', 'conv2a', 'conv3a', 'conv3b', 'conv4a', 'conv4b', 'conv5a', 'conv5b', 'fc6', 'fc7')
    xticks = list(range(len(params_with_biases)))
    yticks = params_with_biases[4:]
    yticklabels = ['%2.3f' % (a / 1e6) for a in yticks]

    fig = plt.figure(dpi=200, figsize=(10,6))
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Layer', size=16)
    ax.yaxis.grid(linestyle='--', zorder=0)
    ax.set_ylabel('#Parameters / 1e06', size=16)
    ax.yaxis.get_major_formatter().set_powerlimits((0,6))
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels,size=12)
    ax.set_xticklabels(labels, size=12)
    ax.text(0, params_with_biases[0] + 100000, '0.005', horizontalalignment='center')
    ax.text(1, params_with_biases[1] + 100000, '0.221', horizontalalignment='center')
    ax.text(2, params_with_biases[2] + 100000, '0.885', horizontalalignment='center')
    ax.text(3, params_with_biases[3] + 100000, '1.770', horizontalalignment='center')
    ax.bar(xticks, params_with_biases, zorder=3)
    fig.savefig('../img_approach/parameters_plot.png', dpi=200, bbox_inches='tight')

if __name__ == '__main__':
    #plot_it()
    pass

