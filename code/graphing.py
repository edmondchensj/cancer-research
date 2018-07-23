import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import pandas as pd
import gensim.corpora as corpora
import wordcloudgrid as wcg

#def set_ax_bg_color(ax):
#    ax.set_facecolor('#FDF6D8')

#def set_fig_bg_color(fig):
#    fig.patch.set_facecolor('#FDF6D8')

'''
For Trends
'''
def show_trend(year_trend,total_growth,model,topic_sensitivity,current_dir,wordcloud_dir,std_footer,relative=False):
    footnote = _footnote(std_footer,topic_sensitivity,dominant=False)
    graph = _plot_trend(year_trend,
                        model,
                        current_dir,
                        title="Trends in Breast Cancer Research (%s)" %('Relative' if relative else 'Absolute'),
                        footnote=footnote,
                        relative=relative)
    wcgrid = wcg.gradientGrid(model,
                            total_growth,
                            wordcloud_dir,
                            current_dir,
                            tag='trend_%s'%('rel' if relative else 'abs'),
                            cmap_relative=relative,
                            cbar_label='Total growth (%)')
    _merge_graph_wcgrid(graph,wcgrid,model,current_dir,tag='trend_%s'%('rel' if relative else 'abs'))

def _plot_trend(year_trend,model,current_dir,title,footnote,relative=False):
    fig = plt.figure()
    ax = fig.add_axes((0.2, 0.18, 0.68, 0.7)) # (left,bottom,width,height)
    plt.style.use('seaborn-notebook')
    linestyle = ['-','--',':','-.'] # alternate linestyles to differentiate topics better.
    for i in range(model.num_topics):
        ax.plot(year_trend[i].index,year_trend[i],label="Topic %s"%(i+1),linestyle=linestyle[i%4])
    xticks = list(year_trend[0].index)
    del xticks[1::2]
    ax.xaxis.set_ticks(xticks)
    ax.set_xlabel("Year")
    ax.set_ylabel("Proportion of total yearly papers (%)" if relative else "No. of Papers")
    ax.set_title(title)
    ax.grid(linestyle='--')
    legend_elements = []
    ax.legend(loc="upper left",fontsize='small')

    fig.text(0.2,0.025, footnote, size='xx-small', ha="left")

    fig_path = '%s/topic_trend%s.png' %(current_dir,'_rel' if relative else '_abs')
    fig.savefig(fig_path,bbox_inches='tight',dpi = 750)
    plt.close()
    return fig_path

'''
For Topic Distribution
'''
def show_distribution(data,model,corpus,topic_sensitivity,current_dir,wordcloud_dir,std_footer,dominant=False):
    footnote = _footnote(std_footer,topic_sensitivity,dominant=dominant)
    topic_keywords = _topic_keywords(model,corpus)
    tag = 'dominant' if dominant else 'mention'
    graph = _plot_graph(data,
                        topic_keywords,
                        model,
                        current_dir,
                        tag=tag,
                        title='Distribution$\mathregular{^{1}}$ of Topics in Breast Cancer Research$\mathregular{^{2}}$',
                        footnote=footnote)
    wcgrid = wcg.gradientGrid(model,data,wordcloud_dir,current_dir,tag=tag,cbar_label='No. of Papers')
    _merge_graph_wcgrid(graph,wcgrid,model,current_dir,tag=tag)

def _footnote(std_footer,topic_sensitivity,dominant=False):
    if dominant:
        footnote = '$\mathregular{^{1}}$ Based on highest contributing topic by Latent Dirichlet Allocation model.\nEach paper is assigned only one topic.'
    else:
        footnote = "$\mathregular{^{1}}$ Based on topic contribution of at least %s by Latent Dirichlet Allocation model. Each paper can feature several topics.\n" %topic_sensitivity
    return footnote + "$\mathregular{^{2}}$ " + std_footer

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

def _plot_graph(topic_dist,topic_keywords,model,current_dir,tag,title,footnote):
    fig = plt.figure()
    ax = fig.add_axes((0.22, 0.18, 0.65, 0.7)) # (left,bottom,width,height)
    y = np.arange(model.num_topics)
    ax.barh(y,topic_dist,align='center')
    ax.set_yticks(y)
    ax.set_yticklabels(topic_keywords) # might have to set fontsize
    ax.invert_yaxis() #reorder topics to start from top to bottom. 
    ax.set_xlabel('No. of papers')
    ax.set_title(title)
    ax.grid(axis='x',linestyle='--')
    
    fig.text(0.12,0.025, footnote, size='xx-small', ha="left")

    plt.style.use('seaborn-notebook')
    plt.tight_layout()
    fig_path = '%s/topic_%s.png' %(current_dir,tag)
    fig.savefig(fig_path,bbox_inches='tight',dpi = 750)
    plt.close()
    return fig_path

'''
For Merging Wordcloud Grid and Graphs
'''
def _merge_graph_wcgrid(graph,wcgrid,model,current_dir,tag):
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

    plt.tight_layout()
    fig_path = '%s/topic_%s_wc.png' %(current_dir,tag)
    fig.savefig(fig_path,bbox_inches='tight',dpi = 1000)
    plt.close()