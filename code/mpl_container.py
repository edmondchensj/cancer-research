import pandas as pd
import numpy as np
import math
import os
# Plotting libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

def set_footnote(fig,footnote):
    fig.text(0.15,0.05, footnote, size='xx-small', ha="left")

def set_title(fig,title):
    fig.suptitle(title,x=0.15,y=1.03,ha='left')

def set_caption(fig,caption):
    fig.text(0.15,0.99, caption, size='x-small', ha="left")

def set_container(figpath,footnote=None,title=None,caption=None):
    fig,ax = plt.subplots()
    ax.imshow(mpimg.imread(figpath))
    ax.axis('off')
    ax.autoscale_view('tight')

    if footnote is not None: set_footnote(fig,footnote)
    if title is not None: set_title(fig,title)
    if caption is not None: set_caption(fig,caption)

    plt.tight_layout()
    fig.savefig(figpath.split('.png')[0]+f'_txt'+'.png',bbox_inches='tight',dpi=1000)

def main():
    figpath = 'data/visualize/basic_4dr.png'
    set_container(figpath,footnote='hello',title='title',caption='caption')

if __name__ == '__main__':
    main()



