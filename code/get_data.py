from Bio import Entrez
from pprint import pprint
import pandas as pd
import time
from lxml import etree
from datetime import timedelta

def get_ids(query,api,batch_size,batch_start):
	handle = Entrez.esearch(db="pubmed", 
							term=query, 
							api_key=api, 
							retmax=batch_size, 
							retstart=batch_start,
							usehistory='y',
							sort="Journal")
	record = Entrez.read(handle)
	pmids = record["IdList"]
	return pmids

def get_text(record):
	#categories = ['BACKGROUND','OBJECTIVE','CONCLUSIONS']
	full_text = ""
	#for category in categories:
	texts = record.xpath("//AbstractText/text()")
	full_text = ' '.join(texts)
	return full_text

def get_year(record):
	year = record.xpath("//PubMedPubDate[@PubStatus='pubmed']/Year/text()")[0]
	if year == "":
		year = float('NaN')
	return year

def get_title(record):
	title = ""
	title = record.xpath("//ArticleTitle/text()")
	if not title:
		title = record.xpath("//BookTitle/text()")
	return title[0]

def get_citedby(pmid,api):
	handle = Entrez.elink(dbfrom="pubmed", id=pmid, linkname="pubmed_pubmed_citedin")
	record = etree.parse(handle)
	cited_by = record.xpath("//Link/Id/text()")
	return cited_by

def get_all_data(query,api,limit):
	start_time = time.time()

	cols = ['Year','Title','Abstract','Cited_by','Total_times_cited']
	df = pd.DataFrame(columns=cols).rename_axis('PMID')
	pd.options.display.max_colwidth = 12
	batch_size = 20
	batch_start = 0

	while df.shape[0] < limit: 
		pmids = get_ids(query,api,batch_size,batch_start)
		# To preview a specific publication, replace ID in this url: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=26239398&retmode=xml
		for pmid in pmids:
			handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml", api_key=api)
			record = etree.parse(handle)
			full_text = get_text(record)
			year = get_year(record)
			title = get_title(record)
			cited_by = get_citedby(pmid,api)
			new_row = pd.Series([year,title,full_text,cited_by,len(cited_by)], index=cols).rename(pmid)
			df.loc[pmid] = new_row # Indexing necessary to avoid duplicates. 

		print('Got %s publications, total: %s' %(len(pmids), df.shape[0]))
		elapsed = time.time() - start_time
		print('Time elapsed: %s' %(str(timedelta(seconds=elapsed))))
		batch_start += batch_size

		if (df.shape[0]%100)==0:
			print('Checkpoint reached. Saving to csv...')
			save_to_csv(df)
		if len(pmids)==0:
			print('Retrieved maximum publications.')
			break

	elapsed = time.time() - start_time
	print('Total time to retrieve data: %s' %(str(timedelta(seconds=elapsed))))
	return df

def save_to_csv(df):
	print('\nPreviewing data:')
	print(df.head(3))
	print('...')
	print(df.tail(3))
	df.to_csv('data/breastcancer_reviews_hasabstract_1997to2017.csv')
	print('Saved to csv.')

def main(query,api,limit):
	df = get_all_data(query,api,limit)
	save_to_csv(df)
	print('Download complete.')

if __name__ == "__main__":
	api = 'c1da8a7e6a61c63d1540db9488d64e22c208'
	email = "chensje@hotmail.com"
	Entrez.email = email
	keyword = "breast cancer"
	year_from = "1997"
	year_to = "2017"
	query = "%s[title] review[publication type] %s:%s[PDAT] hasabstract" % (keyword, year_from, year_to)
	limit = 13100
	main(query,api,limit)