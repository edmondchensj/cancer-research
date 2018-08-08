import numpy as np
import pandas as pd
import math
import pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import (OffsetImage,AnnotationBbox)
from matplotlib_venn import venn3, venn3_circles
import graphing_tools.wordcloudgrid as wcg

''' Table of Contents:
    
    1. General Plotting
    2. Distribution
    3. Trend
    4. Venn
    5. Coherence Graph
'''

# 1. General Plotting
def set_plot_style(): 
    plt.style.use('ggplot')

    # Background colors
    plt.rcParams['axes.facecolor'] = '#FDFDFD' ##EBEBEB
    plt.rcParams['axes.edgecolor'] = '#FDFDFD'
    plt.rcParams['figure.facecolor'] = '#FDFDFD' # off white background. 
    plt.rcParams['savefig.facecolor'] = '#FDFDFD'

    # Text
    plt.rcParams['text.color'] = '#666666'
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.labelsize'] = 'x-small'
    plt.rcParams['axes.labelcolor'] = '#8F8F8F'
    plt.rcParams['xtick.color'] = '#8F8F8F'
    plt.rcParams['xtick.labelsize'] = 'xx-small'
    plt.rcParams['ytick.color'] = '#8F8F8F'
    plt.rcParams['ytick.labelsize'] = 'xx-small'
    plt.rcParams['legend.fontsize'] = 'x-small'

    # Grid
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = '#EBEBEB'
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['grid.alpha'] = 0.75

    # Layout
    plt.rcParams['figure.autolayout'] = True # equiv to plt.tight_layout()
    plt.rcParams['savefig.bbox'] = 'tight' # might be bbox_inches instead
    plt.rcParams['savefig.dpi'] = 750

def _make_plot(num_topics,narrow=False):
    set_plot_style()
    fig = plt.figure(figsize=(6,4.25))
    gs1 = gridspec.GridSpec(1,1)
    if narrow: # for topic distribution. shift chart to the right and make more narrow. 
        gs1.update(left=0.08,right=0.48) # width=0.4
    else:
        gs1.update(left=0,right=0.44) # width=0.44

    col = math.floor(math.sqrt(num_topics))
    row = math.ceil(num_topics/col)
    width = math.sqrt(col/row)*0.42+0.04
    gs2 = gridspec.GridSpec(row,col,wspace=0.02,hspace=0.02)
    gs2.update(left=0.98-width,right=0.98)
    return fig,gs1,gs2

def _save_and_close(fig,figpath):
    fig.savefig(figpath,bbox_inches='tight')
    plt.close()
    print(f'* [graphing.py] File saved: {figpath}.')
    return figpath

def _make_colorbar(fig,sm,cbar_label):
    cbar_ax = fig.add_axes([0.99,0.1,0.01,0.77])
    sm.set_array([])
    cbar = plt.colorbar(sm,cax=cbar_ax)
    cbar.set_label(cbar_label)
    cbar.ax.tick_params(size=0)
    cbar.outline.set_visible(False)

def _set_title_footnote(fig,plot):
    fig.text(-0.02,0.945,_title()[plot],size='x-large',color='#101010')
    fig.text(-0.02,-0.025,f'{_footnote()["std"]}',size='xx-small')

def _title():
    return {'topic_mention':'What topics in breast cancer research ...'
                            '\n                 ... were most prevalent in the past 20 years?',
            'trend_rel':'What topics in breast cancer research ...'
                            '\n                 ... gained the most popularity in the past 20 years?'}

def _footnote():
    return {'std':'Based on a minimum topic contribution of 10%, by Latent Dirichlet Allocation (LDA).'
            '\nData Source: Abstracts from 12,951 review papers with keyword "breast cancer",'
            ' published between 1997 to 2017. Retrieved from PubMed.'
            '\nTopic Modeling algorithm: LDA with Term Frequency-Inverse Document Frequency (TFIDF).'}
# 2. Distribution
def show_distribution(data,model,current_dir,wordcloud_dir,dominant=False):
    num_topics = model.num_topics
    fig,gs1,gs2 = _make_plot(num_topics,narrow=True)
    dynamic_color,sm = wcg.get_dynamic_colors(data,cmap_relative=False)

    _plot_graph(gs1,data,_topic_keywords(model),current_dir)
    wcg.get_wordcloud_grid(fig,gs2,num_topics,wordcloud_dir,current_dir,dynamic_color=dynamic_color,sm=sm,save=False)
    _make_colorbar(fig,sm,cbar_label='Total Papers')
    if not dominant: _set_title_footnote(fig,'topic_mention')

    _save_and_close(fig,f'{current_dir}/topic_{"dominant" if dominant else "mention"}.png')

def _plot_graph(gs,data,topic_keywords,current_dir):
    ax = plt.subplot(gs[0])
    y = np.arange(len(data))
    ax.barh(y,data,align='center')
    ax.set_yticks(y)
    ax.set_yticklabels(topic_keywords)
    ax.invert_yaxis()
    ax.tick_params(axis=u'both', which=u'both',length=0) # remove tick marks.
    ax.set_xlabel('Total Papers')

def _topic_keywords(model,n=1):
    topic_keywords = []
    if n==0:
        topic_keywords = range(1,model.num_topics+1,1)
    else:
        for i in range(model.num_topics):
            words = list(dict(model.show_topic(i, n)).keys())
            words_as_string = '%s (%s,...)' %(i+1,','.join(words))
            topic_keywords.append(words_as_string)
    return topic_keywords


# 3. Trend
def show_trend(year_trend,total_growth,current_dir,wordcloud_dir,relative=False):
    num_topics = len(year_trend)
    fig,gs1,gs2 = _make_plot(num_topics)
    dynamic_color,sm = wcg.get_dynamic_colors(total_growth,cmap_relative=relative)

    _plot_trend(gs1,year_trend,total_growth,dynamic_color,relative)
    wcg.get_wordcloud_grid(fig,gs2,num_topics,wordcloud_dir,current_dir,dynamic_color=dynamic_color,sm=sm,save=False)
    _make_colorbar(fig,sm,cbar_label='Total Growth (%)')
    if relative: _set_title_footnote(fig,'trend_rel')

    _save_and_close(fig,f'{current_dir}/topic_trend{"_rel" if relative else "_abs"}.png')

def _plot_trend(gs,year_trend,total_growth,colors,relative=False):
    ax = plt.subplot(gs[0])

    high_growth,low_growth,hlg = _high_low_growth(total_growth)
    topn = _highest_absolute(year_trend,n=5)
    ann_list = []

    for i in range(len(year_trend)): # we can get number of topics via len(year_trend).
        x,y = year_trend[i].index, year_trend[i]
        if i not in (hlg+topn):
            ax.plot(x,y,color=colors[i],zorder=1)
            continue

        ann_x,ann_y = _auto_adjust(ann_list,x,y)
        if i in hlg:
            ax.plot(x,y,color=colors[i],linewidth=2.5,zorder=2)
            growth = high_growth if i==hlg[0] else low_growth
            growth = f'{growth:+.1f}%' if relative else f'{growth:+.0f}%'
            ax.annotate(f'Topic {i+1} ({growth})',xy=(ann_x,ann_y),fontsize='x-small',fontweight='bold',zorder=3)
        elif i in topn:
            ax.plot(x,y,color=colors[i],zorder=1)
            ax.annotate(f'Topic {i+1}',xy=(ann_x,ann_y),fontsize='xx-small',zorder=3)

    ax.xaxis.set_ticks(list(year_trend[0].index)[::4])
    ax.tick_params(axis=u'both', which=u'both',length=0) # remove tick marks.
    ax.set_xlabel("Year")
    ax.set_ylabel('Proportion of Papers' if relative else 'Number of Papers')

def _high_low_growth(total_growth):
    high_growth,high_idx = max([(v,i) for i,v in enumerate(total_growth)])
    low_growth,low_idx = min([(v,i) for i,v in enumerate(total_growth)])
    return high_growth,low_growth,[high_idx,low_idx]

def _highest_absolute(year_trend,n=5,avoid=[]):
    final_year_values = [topic.values[-1] for topic in year_trend]
    topn = sorted(range(len(year_trend)), key=lambda i:final_year_values[i], reverse=True)[:n+len(avoid)]
    topn = [x for x in topn if x not in avoid]
    return topn

def _auto_adjust(ann_list,x,y):
    ''' A function that prevents overlapping annotations '''
    ann_x,ann_y = x[-1], y.values[-1]
    if ann_list:
        nearest = (np.abs(np.asarray(ann_list) - ann_y)).argmin()
        distance = ann_y - ann_list[nearest]
        min_pad = 0.018
        if abs(distance) < min_pad:
            pad = min_pad-abs(distance)
            if (distance > 0): # shift up
                ann_y += pad
            else:  # shift down
                ann_y -= pad
    ann_list.append(ann_y)
    return ann_x,ann_y


# 4. Venn Diagram
def venn_top3(df,year_trend,threshold,current_dir,wordcloud_dir):
    print('* - - > Getting Venn for top 3 topics (based on year 2017)')
    top3 = _highest_absolute(year_trend,n=3)
    figpath = _show_venn(df,top3[0]+1,top3[1]+1,top3[2]+1,threshold,current_dir,wordcloud_dir,tag='top3') #Note: List index trails topic index by one.
    return figpath

def venn_growth(df,year_trend,total_growth,threshold,current_dir,wordcloud_dir):
    print('* - - > Getting Venn to compare (i) highest absolute (ii) highest growth (iii) lowest growth')
    _,_,[high_idx,low_idx] = _high_low_growth(total_growth)
    top1 = _highest_absolute(year_trend,n=1,avoid=[high_idx,low_idx])
    figpath = _show_venn(df,top1[0]+1,high_idx+1,low_idx+1,threshold,current_dir,wordcloud_dir,tag='growth') #Note: List index trails topic index by one.
    return figpath

def merge_two_venns(venn1_path,venn2_path,current_dir):
    set_plot_style()
    fig = plt.figure(figsize=(7,3))
    ax1 = fig.add_axes([0,0,0.5,1])
    ax2 = fig.add_axes([0.5,0,0.5,1])
    ax1.imshow(mpimg.imread(venn1_path))
    ax1.axis('off')
    ax2.imshow(mpimg.imread(venn2_path))
    ax2.axis('off')
    figpath = _save_and_close(fig,f'{current_dir}/venn_merged.png')

def _style_venn(v,ax,s,topicA,topicB,topicC,wordcloud_dir):
    cmap,__ = wcg.get_dynamic_colors(s,cmap='Reds')
    color = {'100':'#EBEDEF','010':'#EBEDEF','110':cmap[2],'001':'#EBEDEF','101':cmap[4],'011':cmap[5],'111':cmap[6]}
    venn_patches = ['100','010','001','110','101','011','111'] # First three are arranged so that they represent Abc, aBc and abC. 

    for patch in venn_patches:
        if v.get_patch_by_id(patch) is not None: # certain patches may not show if data is heavily skewed. 
            v.get_patch_by_id(patch).set_color(color[patch])

    for text in v.set_labels:
        if text is not None: 
            text.set_fontweight('bold')
    return v

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

def _show_venn(df,topicA,topicB,topicC,threshold,current_dir,wordcloud_dir,tag=''):
    set_plot_style()
    conditions = _venn_conditions(df,topicA,topicB,topicC,threshold)
    subsets = [df[c].shape[0] for c in conditions]
    s = _nan_to_zeros(subsets)
    fig,ax = plt.subplots()
    fig.set_size_inches(6,7)
    v = venn3(subsets=(s[0],s[1],s[2],s[3],s[4],s[5],s[6]),set_labels=(f'Topic {topicA}',f'Topic {topicB}',f'Topic {topicC}'))
    v = _style_venn(v,ax,s,topicA,topicB,topicC,wordcloud_dir)
    fig.text(0.5,0,f'Threshold: {threshold}',ha='center',fontsize='15')
    figpath = _save_and_close(fig,f'{current_dir}/venn_{threshold}_{tag}.png')
    return figpath


# 5. Coherence Graph
def show_coherence_graph(model_dir):
    print("\n* Showing coherence graph ...")
    coherence_values,num_topics_range = _load_cv_ntr(model_dir)
    set_plot_style()
    fig = plt.figure(figsize=(4,3))
    ax = plt.subplot()
    x = num_topics_range

    for i in range(len(coherence_values)):
        ax.plot(x,coherence_values[i],color='#C5C5C5',linewidth=0.5,zorder=1)    # edit
    avg_coherence = np.mean(np.array(coherence_values),axis=0)
    ax.plot(x,avg_coherence,linewidth=1.75,zorder=2)    # edit color

    ax.xaxis.set_ticks(list(num_topics_range)[::2])
    ax.set_xlabel("Number of Topics")
    ax.set_ylabel("Coherence Score")
    ax.tick_params(axis=u'both', which=u'both',length=0) # remove tick marks.
    fig.text(0.1,0,f'The bold line shows the average scores from {len(coherence_values)} runs of Topic Modeling.',size='xx-small')
    fig.savefig(f'{model_dir}/coherence_graph.png')
    print('* Coherence graph saved.')

def _load_cv_ntr(model_dir):
    with open(f'{model_dir}/cv.pkl','rb') as f:
        coherence_values = pickle.load(f)
    with open(f'{model_dir}/num_topic_range.pkl','rb') as f:
        num_topics_range = pickle.load(f)
    return coherence_values,num_topics_range