import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import pandas as pd
import gensim.corpora as corpora
import wordcloudgrid as wcg

styling()

def styling(): 
    plt.style.use('ggplot')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.facecolor']='#FDF6D8'
    plt.rcParams['axes.labelsize'] = 'small'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 'xx-small'
    plt.rcParams['ytick.labelsize'] = 'xx-small'
    plt.rcParams['legend.fontsize'] = 'x-small'
    plt.rcParams['figure.autolayout'] = True # equiv to plt.tight_layout()
    plt.rcParams['savefig.facecolor']='#FDF6D8'
    plt.rcParams['savefig.bbox'] = 'tight' # might be bbox_inches instead
    plt.rcParams['savefig.dpi'] = 1000

def _init_plot():
    fig = plt.figure() # (width,height)
    ax = fig.add_axes((0.2, 0.18, 0.68, 0.7))
    return fig, ax

def _save_and_close(fig,figpath):
    fig.savefig(fig_path)
    plt.close()

''' Trends '''
def show_trend(year_trend,total_growth,current_dir,wordcloud_dir,relative=False):
    graph = _plot_trend(year_trend,total_growth,current_dir,relative)
    tag = f'trend_{"rel" if relative else "abs"}'
    wcgrid = wcg.gradientGrid(total_growth,
                            wordcloud_dir,
                            current_dir,
                            tag=tag
                            cmap_relative=relative,
                            cbar_label='Total growth (%)')
    _merge_graph_wcgrid(graph,wcgrid,current_dir,tag=tag)

def _plot_trend(year_trend,total_growth,current_dir,relative=False):
    fig,ax = _init_plot()

    colors,sm = wcg.get_dynamic_colors(total_growth,cmap_relative=relative)
    max_val,max_idx = max([(v,i) for i,v in enumerate(total_growth)])
    min_val,min_idx = min([(v,i) for i,v in enumerate(total_growth)])
    for i in range(len(year_trend)):
        x,y = year_trend[i].index, year_trend[i]
        if i in [max_idx,min_idx]:
            ax.plot(x,y,color=colors[i],label=f"Topic {i+1}",linewidth='5')
            ax.annotate(f'{max_val if i==max_idx else min_val}',xy=(x+0.01,y+0.01),size='x-small',fontweight='bold')
        else:
            ax.plot(x,y,color=colors[i],label=f"Topic {i+1}")

    ax.xaxis.set_ticks(list(year_trend[0].index) del xticks[1::2])
    ax.set_xlabel("Year")
    ax.set_ylabel("Proportion of total yearly papers (%)")
    _save_and_close(fig,f'{current_dir}/topic_trend{"_rel" if relative else "_abs"}.png')
    return fig_path

''' Distributions '''
def show_distribution(data,model,corpus,current_dir,wordcloud_dir,dominant=False):
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
    _save_and_close(fig,f'{current_dir}/topic_{tag}.png')
    return fig_path

''' Merge with Wordcloud Grid '''
def _merge_graph_wcgrid(graph,wcgrid,current_dir,tag):
    print('* - - -> Merging graph and wordcloud grid')
    fig = plt.figure()
    graph_ax = plt.axes([0,0.15,0.53,1])
    wc_ax = plt.axes([0.53,0.15,0.47,1])
    graph_ax.imshow(mpimg.imread(graph))
    wc_ax.imshow(mpimg.imread(wcgrid))
    graph_ax.axis('off')
    graph_ax.autoscale_view('tight')
    wc_ax.axis('off')
    wc_ax.autoscale_view('tight')
    _save_and_close(fig,f'{current_dir}/topic_{tag}_wc.png')