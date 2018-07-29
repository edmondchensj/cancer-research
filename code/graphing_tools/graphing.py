import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from matplotlib_venn import venn3, venn3_circles
import graphing_tools.wordcloudgrid as wcg

def styling(): 
    plt.style.use('ggplot')
    plt.rcParams['axes.facecolor']='#EBEDEF' ##EBEBEB
    plt.rcParams['axes.labelsize'] = 'small'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['text.color'] = '#4C4C4B'
    plt.rcParams['axes.grid'] = False
    plt.rcParams['xtick.labelsize'] = 'xx-small'
    plt.rcParams['ytick.labelsize'] = 'xx-small'
    plt.rcParams['legend.fontsize'] = 'x-small'
    plt.rcParams['figure.autolayout'] = True # equiv to plt.tight_layout()
    plt.rcParams['savefig.bbox'] = 'tight' # might be bbox_inches instead
    plt.rcParams['savefig.dpi'] = 1000

def _init_plot():
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_axes([0.2,0.1,0.63,0.8])
    return fig, ax

def _save_and_close(fig,figpath):
    fig.savefig(figpath)
    plt.close()
    print(f'* [graphing.py] File saved: {figpath}.')
    return figpath

def _style_venn(v):
    patches = ['100','010','001','110','101','011','111']
    color = {'100':'#E6B0AA','010':'#ABEBC6','001':'#D6EAF8','110':'#F4D03F','101':'#BB8FCE','011':'#73C6B6','111':'#E5E8E8'}
    # color: light red, light green, light blue, yel, magenta, teal, light gray. 
    for patch in patches:
        if v.get_patch_by_id(patch) is not None: # certain patches may not show if data is heavily skewed. 
            v.get_patch_by_id(patch).set_color(color[patch])
    return v

''' Venn Diagram '''
def _venn_conditions(df,topicA,topicB,topicC,threshold):
    Abc = (df[f'Topic_{topicA}']>=threshold) & (df[f'Topic_{topicB}']<threshold) & (df[f'Topic_{topicC}']<threshold)
    aBc = (df[f'Topic_{topicA}']<threshold) & (df[f'Topic_{topicB}']>=threshold) & (df[f'Topic_{topicC}']<threshold)
    ABc = (df[f'Topic_{topicA}']>=threshold) & (df[f'Topic_{topicB}']>=threshold) & (df[f'Topic_{topicC}']<threshold)
    abC = (df[f'Topic_{topicA}']<threshold) & (df[f'Topic_{topicB}']<threshold) & (df[f'Topic_{topicC}']>=threshold)
    AbC = (df[f'Topic_{topicA}']>=threshold) & (df[f'Topic_{topicB}']<threshold) & (df[f'Topic_{topicC}']>=threshold)
    aBC = (df[f'Topic_{topicA}']<threshold) & (df[f'Topic_{topicB}']>=threshold) & (df[f'Topic_{topicC}']>=threshold)
    ABC = (df[f'Topic_{topicA}']>=threshold) & (df[f'Topic_{topicB}']>=threshold) & (df[f'Topic_{topicC}']>=threshold)
    return [Abc, aBc, ABc, abC, AbC, aBC, ABC]

def _nan_to_zeros(subsets):
    return list(map(lambda x: 0 if math.isnan(x) else x,subsets))

def _venn(df,topicA,topicB,topicC,threshold,current_dir,tag=''):
    conditions = _venn_conditions(df,topicA,topicB,topicC,threshold)
    subsets = [df[c].shape[0] for c in conditions]
    s = _nan_to_zeros(subsets)

    fig = plt.figure(figsize=(8,8))
    v = venn3(subsets=(s[0],s[1],s[2],s[3],s[4],s[5],s[6]),set_labels=(f'Topic {topicA}',f'Topic {topicB}',f'Topic {topicC}'))
    v = _style_venn(v)
    figpath = _save_and_close(fig,f'{current_dir}/venn_{threshold}_{tag}.png')

def venn_top3(df,year_trend,threshold,current_dir):
    ''' Add one to each because list indexing trails topic indexing by one. '''
    top3 = _highest_absolute(year_trend,n=3)
    _venn(df,top3[0]+1,top3[1]+1,top3[2]+1,threshold,current_dir,tag='top3')

def venn_growth(df,year_trend,total_growth,threshold,current_dir):
    ''' Compares topics with highest, lowest growth with topic with highest absolute. '''
    high_growth,low_growth,[high_idx,low_idx] = _high_low_growth(total_growth)
    top1 = _highest_absolute(year_trend,n=1)
    _venn(df,top1[0]+1,high_idx+1,low_idx+1,threshold,current_dir,tag='growth')

''' Trends '''
def show_trend(year_trend,total_growth,current_dir,wordcloud_dir,relative=False):
    styling()
    tag = f'trend_{"rel" if relative else "abs"}'
    graph = _plot_trend(year_trend,total_growth,current_dir,relative)
    wcgrid = wcg.gradientGrid(total_growth,
                            wordcloud_dir,
                            current_dir,
                            tag=tag,
                            cmap_relative=relative,
                            cbar_label='Total Growth')
    _merge_graph_wcgrid(graph,wcgrid,current_dir,tag=tag)

def _high_low_growth(total_growth):
    high_growth,high_idx = max([(v,i) for i,v in enumerate(total_growth)])
    low_growth,low_idx = min([(v,i) for i,v in enumerate(total_growth)])
    return high_growth,low_growth,[high_idx,low_idx]

def _highest_absolute(year_trend,n=3):
    final_year_values = [topic.values[-1] for topic in year_trend]
    topn = sorted(range(len(year_trend)), key=lambda i:final_year_values[i], reverse=True)[:n]
    return topn

def _auto_adjust(ann_list,x,y):
    ''' A function that prevents overlapping annotations '''
    ann_x,ann_y = x[-1], y.values[-1]
    if ann_list:
        nearest = (np.abs(np.asarray(ann_list) - ann_y)).argmin()
        distance = ann_y - ann_list[nearest]
        min_pad = 0.04
        if abs(distance) < min_pad:
            print(f'Overlap! {distance}')
            pad = min_pad-abs(distance)
            if (distance > 0): # shift up
                ann_y += pad
            else:  # shift down
                ann_y -= pad
    ann_list.append(ann_y)
    return ann_x,ann_y

def _plot_trend(year_trend,total_growth,current_dir,relative=False):
    fig,ax = _init_plot()
    colors,sm = wcg.get_dynamic_colors(total_growth,cmap_relative=relative)

    high_growth,low_growth,hlg = _high_low_growth(total_growth)
    topn = _highest_absolute(year_trend,n=5)
    ann_list = []

    for i in range(len(year_trend)):
        x,y = year_trend[i].index, year_trend[i]
        if i not in (hlg+topn):
            ax.plot(x,y,color=colors[i],zorder=1)
            continue

        ann_x,ann_y = _auto_adjust(ann_list,x,y)
        if i in hlg:
            ax.plot(x,y,color=colors[i],linewidth=2.5,zorder=2)
            growth = high_growth if i==hlg[0] else low_growth
            growth = f'{growth:+.2f}' if relative else f'{growth:+d}'
            ax.annotate(f'Topic {i+1} ({growth})',xy=(ann_x,ann_y),fontsize='x-small',fontweight='bold',zorder=3)
        elif i in topn:
            ax.plot(x,y,color=colors[i],zorder=1)
            ax.annotate(f'Topic {i+1}',xy=(ann_x,ann_y),fontsize='x-small',zorder=3)

    ax.xaxis.set_ticks(list(year_trend[0].index)[::2])
    ax.set_xlabel("Year")
    ax.set_ylabel('Proportion of Papers' if relative else 'Number of Papers')
    fig.savefig(f'{current_dir}/topic_trend{"_rel" if relative else "_abs"}.png')
    figpath = _save_and_close(fig,f'{current_dir}/topic_trend{"_rel" if relative else "_abs"}.png')
    return figpath

''' Distributions '''
def show_distribution(data,model,corpus,current_dir,wordcloud_dir,dominant=False):
    styling()
    tag = 'dominant' if dominant else 'mention'
    graph = _plot_graph(data,
                        _topic_keywords(model,corpus),
                        current_dir,
                        tag=tag)
    wcgrid = wcg.gradientGrid(data,wordcloud_dir,current_dir,tag=tag,cbar_label='No. of Papers')
    _merge_graph_wcgrid(graph,wcgrid,current_dir,tag=tag)

def _topic_keywords(model,corpus,n=1):
    topic_keywords = []
    if n==0:
        topic_keywords = range(1,model.num_topics+1,1)
    else:
        for i in range(model.num_topics):
            words = list(dict(model.show_topic(i, n)).keys())
            words_as_string = '%s (%s,...)' %(i+1,','.join(words))
            topic_keywords.append(words_as_string)
    return topic_keywords

def _plot_graph(data,topic_keywords,current_dir,tag):
    fig,ax = _init_plot()
    y = np.arange(len(data))
    ax.barh(y,data,align='center')
    ax.set_yticks(y)
    ax.set_yticklabels(topic_keywords)
    ax.invert_yaxis()
    ax.set_xlabel('No. of papers')
    ax.set_ylabel('Topic')
    figpath = _save_and_close(fig,f'{current_dir}/topic_{tag}.png')
    return figpath

''' Merge with Wordcloud Grid '''
def _merge_graph_wcgrid(graph,wcgrid,current_dir,tag):
    print('* - - -> Merging graph and wordcloud grid')
    fig = plt.figure(figsize=(12,5))
    #graph_ax = plt.axes([0,0,0.58,1])
    #wc_ax = plt.axes([0.58,0,0.47,1])
    graph_ax = fig.add_axes([0,0,0.55,1])
    wc_ax = fig.add_axes([0.55,0,0.43,1])
    graph_ax.imshow(mpimg.imread(graph))
    wc_ax.imshow(mpimg.imread(wcgrid))
    graph_ax.axis('off')
    #graph_ax.autoscale_view('tight')
    wc_ax.axis('off')
    #wc_ax.autoscale_view('tight')
    figpath = _save_and_close(fig,f'{current_dir}/topic_{tag}_wc.png')