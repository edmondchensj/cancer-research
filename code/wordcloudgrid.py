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

def _sort_by_date(path):
    return list(sorted(os.listdir(path), key=lambda f: os.stat(os.path.join(path, f)).st_mtime))

def initialize_grid(model):
    gridwidth = math.ceil(math.sqrt(model.num_topics))
    fig = plt.figure(figsize=(gridwidth,gridwidth))
    gs = gridspec.GridSpec(gridwidth, gridwidth, wspace=0, hspace=0.10)
    return fig, gs, gridwidth

def retrieve_wordclouds(wordcloud_dir):
    '''
    Returns list of filepaths for saved wordclouds
    '''
    return [str(wordcloud_dir + '/' + img) for img in _sort_by_date(wordcloud_dir) if 'wordcloud_topic' in img]

def get_dynamic_colors(data,cmap_relative):
    if ~cmap_relative:
        norm = Normalize(vmin=min(data), vmax=max(data)+(max(data)/8))
        color = 'Oranges'
    else:
        max_val = max(data,key=abs)
        norm = Normalize(vmin=-max_val, vmax=max_val)
        color = 'RdYlGn'
    cmap = cm.get_cmap(color)
    colors = list(map(lambda x: cmap(norm(x)), data))
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
    print('* [wordcloudgrid.py] cmap color: %s' %color)
    return colors,sm

def generate_wc_grid(fig,gs,gridwidth,model,images,target_dir,sm=False,cbar_label=False,dynamic_color=False,tag=''):
    '''
    If dynamic_color=False, this function will generate a basic grid with fixed box color.
    '''
    idx=0
    for i,j in product(range(gridwidth),range(gridwidth)):
        if idx==model.num_topics:
            break
        else:
            img = mpimg.imread(images[idx])
            ax = plt.subplot(gs[i,j])
            ax.imshow(img)
            for pos in ['top', 'bottom', 'right', 'left']:
                if dynamic_color==False:
                    ax.spines[pos].set_color('#E5E7E9')
                else:
                    ax.spines[pos].set_color(dynamic_color[idx])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.annotate('%s'%(idx+1), xy=(0,0), xytext=(0.0175,0.875), textcoords='axes fraction',fontsize='x-small', color='gray')
            idx += 1
    if sm != False:
        gs.update(left=0,right=0.92,wspace=-0.23)
        cbar_ax = fig.add_axes([0.92,0.12,0.03,0.76])
        sm.set_array([])
        cbar = plt.colorbar(sm,cax=cbar_ax)
        print('* [wordcloudgrid.py] cbar label: %s' %cbar_label)
        cbar.set_label('%s' %cbar_label,size='x-small')
        cbar.ax.tick_params(labelsize='x-small',pad=-0.5)

    fig_path = '%s/wordcloud_grid%s.png'%(target_dir,'_'+tag)
    fig.savefig(fig_path,dpi = 1000, bbox_inches='tight')
    print('* [wordcloudgrid.py] Wordcloud grid saved.')
    return fig_path

def basic_grid_backup(model,wordcloud_dir,target_dir):
    print("\nNow combining wordclouds to grid...")
    gridwidth = math.ceil(math.sqrt(model.num_topics))
    print("Gridsize: %s x %s"%(gridwidth,gridwidth))
    fig = plt.figure(figsize=(gridwidth,gridwidth))
    gs = gridspec.GridSpec(gridwidth, gridwidth, wspace=0, hspace=0.10)
    images = [img for img in _sort_by_date(wordcloud_dir) if 'wordcloud_topic' in img]
    print("List of wordclouds:")
    print(images)

    idx = 0
    for i,j in product(range(gridwidth),range(gridwidth)):
        if idx==model.num_topics:
            break
        else:
            img = mpimg.imread(wordcloud_dir + '/' + images[idx])
            ax = plt.subplot(gs[i,j])
            ax.imshow(img)
            for pos in ['top', 'bottom', 'right', 'left']:
                ax.spines[pos].set_color('#E5E7E9')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.annotate('%s'%(idx+1), xy=(0,0), xytext=(0.0175,0.875), textcoords='axes fraction',fontsize='x-small', color='#E5E7E9')
            idx += 1

    fig_path = '%s/wordcloud_grid.png'%target_dir
    fig.savefig(fig_path,dpi = 1000)
    print('* [wordcloudgrid.py] Wordcloud grid saved.')
    return fig_path

def gradientGrid(model,data,wordcloud_dir,target_dir,tag='',cmap_relative=False,cbar_label=False):
    '''
    Takes in an array 'data' of length model.num_topics that changes color of each topic bounding box. 
    This is useful for side-by-side graphs, where the left shows a bar or line plot, and the right is this wordcloud grid.
    '''
    print("* [wordcloudgrid.py] Now making wordcloud grid with color gradients ...")
    fig, gs, gridwidth = initialize_grid(model)
    images = retrieve_wordclouds(wordcloud_dir)
    colors,sm = get_dynamic_colors(data,cmap_relative)
    fig_path = generate_wc_grid(fig,gs,gridwidth,model,images,target_dir,sm=sm,cbar_label=cbar_label,dynamic_color=colors,tag=tag)
    return fig_path

def basicGrid(model,wordcloud_dir,target_dir):
    print("* [wordcloudgrid.py] Now combining wordclouds to grid...")
    fig, gs, gridwith = initialize_grid(model)
    images = retrieve_wordclouds(wordcloud_dir)
    fig_path = generate_wc_grid(fig,gs,gridwidth,model,images,target_dir)
    return fig_path