import math
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from itertools import product

'''
Function Name Style Guide:
1. no underscores ('_'): Main Functions     e.g. gradientGrid()
2. words separated by underscores: Sub Functions
3. prefix underscore: weak internal use.    e.g. _sort_by_date()
'''

# Styling
matplotlib.style.use('seaborn-notebook')
plt.rcParams['axes.facecolor']='#FDF6D8'
plt.rcParams['savefig.facecolor']='#FDF6D8'
plt.rcParams['axes.labelweight'] = 'bold'

def _sort_by_date(path):
    return list(sorted(os.listdir(path), key=lambda f: os.stat(os.path.join(path, f)).st_mtime))

def initialize_grid(model,gradient):
    num_topics = model.num_topics
    row = math.ceil(math.sqrt(num_topics))
    col = row
    fig = plt.figure(figsize=(row,col))

    # auto adjust to rectangle 
    if gradient and (math.ceil(num_topics/row) < row): 
        col = col - 1
        fig.set_size_inches(col,row)
    
    gs = gridspec.GridSpec(row, col, wspace=0, hspace=0.10)
    return fig, gs, row, col

def retrieve_wordclouds(wordcloud_dir):
    '''
    Returns list of filepaths for saved wordclouds
    '''
    return [str(wordcloud_dir + '/' + img) for img in _sort_by_date(wordcloud_dir) if 'wordcloud_topic' in img]

def get_dynamic_colors(data,cmap_relative):
    if cmap_relative: 
        max_val = abs(max(data,key=abs))
        bound = max_val + max_val/10
        norm = Normalize(vmin=-bound, vmax=bound)
        color = 'RdYlGn'
    else:
        norm = Normalize(vmin=min(data), vmax=max(data)+(max(data)/8))
        color = 'Oranges'
    cmap = cm.get_cmap(color)
    colors = list(map(lambda x: cmap(norm(x)), data))
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
    return colors,sm

def generate_wc_grid(model,wordcloud_dir,target_dir,gradient=False,dynamic_color=None,sm=None,cbar_label=None,tag=''):
    '''
    If gradient=False, this function will generate a basic grid with fixed box color.
    '''
    fig, gs, row, col = initialize_grid(model,gradient)
    images = retrieve_wordclouds(wordcloud_dir)

    idx=0
    for i,j in product(range(row),range(col)):
        if idx==model.num_topics:
            break
        else:
            img = mpimg.imread(images[idx])
            ax = plt.subplot(gs[i,j])
            ax.imshow(img)
            for pos in ['top', 'bottom', 'right', 'left']:
                if gradient:
                    ax.spines[pos].set_color(dynamic_color[idx])
                else:
                    ax.spines[pos].set_color('#E5E7E9')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.annotate('%s'%(idx+1), xy=(0,0), xytext=(0.0175,0.875), textcoords='axes fraction',fontsize='x-small', color='gray')
            idx += 1

    if gradient:
        if row == col:
            gs.update(left=0,right=0.92,wspace=-0.3)
        else:
            gs.update(left=0,right=0.99,wspace=0.1)
        cbar_ax = fig.add_axes([0.9,0.12,0.02,0.76])
        sc = plt.gca().get_children()[0]
        cbar = plt.colorbar(sc,cax=cbar_ax)
        cbar.set_label('%s' %cbar_label,size='xx-small',labelpad=-0.05)
        cbar.ax.tick_params(labelsize='xx-small',pad=-0.15)

    fig_path = '%s/wordcloud_grid%s.png'%(target_dir,'_'+tag)
    fig.savefig(fig_path,bbox_inches='tight',dpi=1000)
    print('* [wordcloudgrid.py] File saved: %s.' %fig_path)
    return fig_path

def gradientGrid(model,data,wordcloud_dir,target_dir,tag='',cmap_relative=False,cbar_label=False):
    '''
    Takes in an array 'data' of length model.num_topics that changes color of each topic bounding box. 
    This is useful for side-by-side graphs, where the left shows a bar or line plot, and the right is this wordcloud grid.
    '''
    print("* [wordcloudgrid.py] Now making wordcloud grid with color gradients ...")
    dynamic_color,sm = get_dynamic_colors(data,cmap_relative)
    fig_path = generate_wc_grid(model,wordcloud_dir,target_dir,gradient=True,dynamic_color=dynamic_color,sm=sm,cbar_label=cbar_label,tag=tag)
    return fig_path

def basicGrid(model,wordcloud_dir,target_dir):
    print("* [wordcloudgrid.py] Now visualizing model as wordcloud grid...")
    fig_path = generate_wc_grid(model,wordcloud_dir,target_dir)
    return fig_path