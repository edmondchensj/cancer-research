import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import pandas as pd
import gensim.corpora as corpora

def total_growth(year_trend):
    func = lambda x: (x.values[-1] - x.values[0])/x.values[0]*100
    total_growth = list(map(func,year_trend))
    return total_growth

def trend(df,topic_range,topic_sensitivity,relative=False):
    year_trend = []
    papers_per_year = df.groupby('Year').size()
    for i in topic_range:
        topic_mentions = df[df['Topic_%s'%(i+1)]>=topic_sensitivity].groupby('Year').size()
        if relative:
            topic_mentions = topic_mentions/papers_per_year
        year_trend.append(topic_mentions)
    return year_trend

def plot_trend(year_trend,year_range,model,current_dir,relative=False):
    fig = plt.figure()
    ax = fig.add_axes((0.2, 0.18, 0.65, 0.7)) # (left,bottom,width,height)
    plt.style.use('seaborn-notebook')

    linestyle = ['-','--',':','-.'] # alternate linestyles to differentiate topics better.
    for i in range(model.num_topics):
        ax.plot(year_range,year_trend[i],label="Topic %s"%(i+1),linestyle=linestyle[i%4])
    xticks = list(year_range)
    del xticks[1::2]
    ax.xaxis.set_ticks(xticks)
    ax.set_xlabel("Year")
    ax.set_ylabel("Proportion of total yearly papers (%)" if relative else "No. of Papers")
    ax.set_title("Topic Mentions across years")
    ax.grid(linestyle='--')
    legend_elements = []
    ax.legend(loc="upper left",fontsize='small')
    fig_path = '%s/topic_trend%s.png' %(current_dir,'_rel' if relative else '_abs')
    fig.savefig(fig_path,bbox_inches='tight',dpi = 750)
    plt.close()
    return fig_path

def merge_graph_wcgrid(graph,wcgrid,model,current_dir,tag):
    print('* - - -> Merging graph and wordcloud grid')
    # set axes:
    fig = plt.figure()
    graph_ax = plt.axes([0,0,0.53,1])
    wc_ax = plt.axes([0.53,0,0.47,1])
    graph_ax.imshow(mpimg.imread(graph))
    wc_ax.imshow(mpimg.imread(wcgrid))
    graph_ax.axis('off')
    graph_ax.autoscale_view('tight')
    wc_ax.axis('off')
    wc_ax.autoscale_view('tight')
    plt.tight_layout()
    fig_path = '%s/topic_%s_wc.png' %(current_dir,tag)
    fig.savefig(fig_path,bbox_inches='tight',dpi = 1000)
    plt.close()

def plot_graph(topic_dist,topic_keywords,model,current_dir,tag,title,footnote):
    fig = plt.figure()
    ax = fig.add_axes((0.2, 0.18, 0.65, 0.7)) # (left,bottom,width,height)
    y = np.arange(model.num_topics)
    ax.barh(y,topic_dist,align='center')
    ax.set_yticks(y)
    ax.set_yticklabels(topic_keywords) # might have to set fontsize
    ax.invert_yaxis() #reorder topics to start from top to bottom. 
    ax.set_xlabel('Number of papers')
    ax.set_title(title)
    ax.grid(axis='x',linestyle='--')
    plt.style.use('seaborn-notebook')
    plt.tight_layout()
    fig.text(0.5,0.025, footnote, size=8, ha="center")
    fig_path = '%s/topic_%s.png' %(current_dir,tag)
    fig.savefig(fig_path,bbox_inches='tight',dpi = 750)
    plt.close()
    return fig_path