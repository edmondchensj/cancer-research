import math
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.cm as cm
from matplotlib.colors import Normalize

def set_plot_style(): 
    plt.style.use('ggplot')
    plt.rcParams['axes.labelsize'] = 'small'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 'xx-small'
    plt.rcParams['ytick.labelsize'] = 'xx-small'
    plt.rcParams['legend.fontsize'] = 'x-small'
    plt.rcParams['figure.autolayout'] = True # equiv to plt.tight_layout()
    plt.rcParams['savefig.bbox'] = 'tight' # might be bbox_inches instead
    plt.rcParams['savefig.dpi'] = 1000

def _sort_by_date(path):
    return list(sorted(os.listdir(path), key=lambda f: os.stat(os.path.join(path, f)).st_mtime))

def retrieve_wordclouds(wordcloud_dir):
    return [str(wordcloud_dir + '/' + img) for img in _sort_by_date(wordcloud_dir) if 'wordcloud_topic' in img]

def initialize_grid(num_topics,gradient=False):
    row = math.ceil(math.sqrt(num_topics))
    col = row
    fig = plt.figure(figsize=(col-0.35,row))
    if gradient and (math.ceil(num_topics/row) < row): # auto adjust to rectangle 
        col = col - 1
        fig.set_size_inches(col-1,row)
    gs = gridspec.GridSpec(row, col, wspace=0, hspace=0.10)
    return fig, gs

def _save_and_close(fig,figpath):
    fig.savefig(figpath)
    plt.close()
    print('* [wordcloudgrid.py] File saved: %s.' %figpath)
    return figpath

def get_dynamic_colors(data,cmap_relative=False,cmap=None):
    if cmap_relative: 
        max_val = abs(max(data,key=abs))
        bound = max_val + max_val/10
        norm = Normalize(vmin=-bound, vmax=bound)
        color = 'RdYlGn' if cmap is None else cmap
    else:
        norm = Normalize(vmin=min(data), vmax=max(data)+(max(data)/7))
        color = 'Oranges' if cmap is None else cmap
    cmap = cm.get_cmap(color)
    colors = list(map(lambda x: cmap(norm(x)), data))
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
    return colors,sm

def generate_wc_grid(fig,gs,num_topics,wordcloud_dir,target_dir,dynamic_color=None,sm=None,cbar_label=None,tag='',save=True,figpath=None):
    images = retrieve_wordclouds(wordcloud_dir)

    for i in range(num_topics):
        img = mpimg.imread(images[i])
        ax = plt.subplot(gs[i])
        ax.imshow(img)
        for pos in ['top', 'bottom', 'right', 'left']:
            if gradient:
                ax.spines[pos].set_color(dynamic_color[i])
            else:
                ax.spines[pos].set_color('#E5E7E9')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.annotate('%s'%(i+1), xy=(0,0), xytext=(0.0175,0.875), textcoords='axes fraction',fontsize='x-small', color='gray')

    if dynamic_color is not None: # Make colorbar
        cbar_ax = fig.add_axes([0.99,0.12,0.02,0.76])
        sm.set_array([])
        cbar = plt.colorbar(sm,cax=cbar_ax)
        cbar.set_label('%s' %cbar_label,size='xx-small',labelpad=0.2)
        cbar.ax.tick_params(labelsize='xx-small',pad=0)

    if save: figpath = _save_and_close(fig,f'{target_dir}/wordcloud_grid{"_"+tag}.png')
    return figpath

def gradientGrid(data,wordcloud_dir,target_dir,tag='',cmap_relative=False,cbar_label=False):
    print("* [wordcloudgrid.py] Now making wordcloud grid with color gradients ...")
    set_plot_style()
    num_topics = len(data)
    fig, gs = initialize_grid(num_topics,gradient=True)
    dynamic_color,sm = get_dynamic_colors(data,cmap_relative)
    figpath = generate_wc_grid(fig,gs,num_topics,wordcloud_dir,target_dir,dynamic_color=dynamic_color,sm=sm,cbar_label=cbar_label,tag=tag)
    return figpath

def basicGrid(num_topics,wordcloud_dir,target_dir):
    print("* [wordcloudgrid.py] Now visualizing model as wordcloud grid...")
    set_plot_style()
    fig, gs = initialize_grid(num_topics)
    figpath = generate_wc_grid(fig,gs,num_topics,wordcloud_dir,target_dir)
    return figpath