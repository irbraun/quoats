import streamlit as st
import pandas as pd
import numpy as np
import sys
import re
import itertools
import nltk
import re
import base64
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from string import punctuation
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import strip_non_alphanum, stem_text, preprocess_string, strip_tags, strip_punctuation
from PIL import Image
from textwrap import wrap
import plotly
import plotly.graph_objects as go



sys.path.append("../oats")
import oats
from oats.utils.utils import load_from_pickle, save_to_pickle, flatten
from oats.biology.dataset import Dataset
from oats.annotation.ontology import Ontology








from token_similarities import TokenSimilarities, LossLogger
import query_handlers as qh



nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)











# Path to the documentation and dataset of genes, phenotype descriptions and annotations to be used. 
DOC_PARAGRAPH_PATH = "documentation/paragraph.txt"
DOC_TABLE_PATH = "documentation/table.txt"
DATASET_PATH = "data/genes_texts_annots.csv"

# Names and paths specific to the ontologies used.
ONTOLOGY_NAMES = ["PATO","PO","GO"]
ONTOLOGY_OBO_PATHS = ["ontologies/pato.obo", "ontologies/po.obo", "ontologies/go.obo"]
ONTOLOGY_PICKLE_PATHS = ["ontologies/pato.pickle", "ontologies/po.pickle", "ontologies/go.pickle"]

# These mappings are necessary if the internal strings used for species and how they should be displayed are different.
SPECIES_STRINGS_IN_DATA = ["ath", "zma", "sly", "gmx", "osa", "mtr"]
SPECIES_STRINGS_TO_DISPLAY = ["Arabidopsis", "Maize", "Tomato", "Soybean", "Rice", "Medicago"]
TO_SPECIES_DISPLAY_NAME = {i:d for i,d in zip(SPECIES_STRINGS_IN_DATA,SPECIES_STRINGS_TO_DISPLAY)}



# Paths relevent to the saved machine learning models or classes.
DOC2VEC_MODEL_PATH = "models/doc2vec_model_trained_on_plant_phenotypes.model"
WORD2VEC_MODEL_PATH = "models/word2vec_model_trained_on_plant_phenotypes.model"






WORD_EMBEDDINGS_PICKLE_PATH = "stored/stored_token_similarities.pickle"
SENT_EMBEDDINGS_FROM_WORD2VEC_PATH = "stored/stored_sent_embeddings_from_word2vec"
SENT_EMBEDDINGS_FROM_DOC2VEC_PATH = "stored/stored_sent_embeddings_from_doc2vec"
SENT_EMBEDDINGS_FROM_TFIDF_PATH = "stored/stored_sent_embeddings_from_tfidf"



















# Constants that help define how the tables look and how the text wraps within the table cells.
TABLE_HEADER_COLOR = "#808080"
TABLE_HEIGHT = 1500
HEADER_HEIGHT = 30
RESULT_COLUMN_WIDTH = 55
MAX_LINES_IN_RESULT_COLUMN = 100
DESCRIPTION_COLUMN_WIDTH = 90
NEWLINE_TOKEN = "[NEWLINE]"

# What color the alternating rows should be, and what column will indicate when they should alternate?
TABLE_ROWS_COLOR_1 = "#FFFFFF" # light gray
TABLE_ROWS_COLOR_2 = "#F1F2F6" # slighly darker gray
ALTERNATE_ROW_COLOR_BASED_ON_COLUMN_KEY = "gene"









# Constants that help define how the columns appear in the plottly tables. 
# The first value is the universal string key used throughout the script, so leave that alone.
# The second is how the column is titled in the presented tables so that can be changed just here and the change will take effect throughout.
# The third is the relative width of the column to all the other columns. Leave the rank column as 1 (the smallest), and change all othere
# with respect to that column.
COLUMN_SETTINGS = [
	("rank", "<b>Rank<b>", 0.9),
	("score", "<b>Score<b>", 1),
	("result", "Result", 1),
	("keywords", "<b>Query Keywords<b>", 8),
	("sentences", "Query Sentences", 8),
	("terms", "Ontology Terms", 6),
	("species", "<b>Species<b>", 2),
	("gene", "<b>Gene<b>", 2),
	("model", "<b>Gene Model<b>", 2),
	("phenotype", "<b>Phenotype Description<b>", 12),
	("query_sentence", "<b>Query<b>", 5),
	("matching_sentence", "<b>Matches<b>", 12),
	("query_term_id", "<b>Query Term ID", 2),
	("query_term_name", "<b>Query Term Name", 4),
	("annotated_term_id", "<b>Annotated Term ID", 2),
	("annotated_term_name", "<b>Annotated Term Name", 4),
	("internal_id", "Internal ID", 1),
]

COLUMN_NAMES = {x[0]:x[1] for x in COLUMN_SETTINGS}
COLUMN_NAMES_TO_OUTPUT_COLUMN_NAME = {v:v.replace("<b>","") for k,v in COLUMN_NAMES.items()}
COLUMN_WIDTHS = {x[0]:x[2] for x in COLUMN_SETTINGS}








# How should keywords and phrases be cleaned and handled as far as preprocessing or stemming goes?
KEYWORD_DELIM = "[DELIM]"
KEYWORD_PREPROCESSING_FILTERS = [lambda x: x.lower(), strip_non_alphanum, strip_tags, strip_punctuation, stem_text]
PREPROCESSING_FOR_KEYWORD_SEARCH_FUNCTION = lambda x: "{}{}{}".format(KEYWORD_DELIM, KEYWORD_DELIM.join([token for token in preprocess_string(x, KEYWORD_PREPROCESSING_FILTERS)]), KEYWORD_DELIM)










# Initial configuration and the header image at the top of the page.
st.set_page_config(page_title="QuOATS", layout="wide", initial_sidebar_state="expanded")
PATH_TO_LOGO_PNG = "images/header.png"
st.image(Image.open(PATH_TO_LOGO_PNG), caption=None, width=500, output_format="png")
st.markdown("### A tool for **Qu**erying phenotype descriptions with **O**ntology **A**nnotations and **T**ext **S**imilarity")


hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 













with open(DOC_PARAGRAPH_PATH,"r") as f:
	doc_paragraph = f.read()

with open(DOC_TABLE_PATH,"r") as f:
	table_string = f.read()
	lines = table_string.split("\n")
	lines = ["|{}|".format(line.replace("\t","|")) for line in lines]
	lines = [line.replace('"','') for line in lines]
	header_separation_line = "-".join(["|"]*lines[0].count("|"))
	all_lines = []
	all_lines.append(lines[0])
	all_lines.append(header_separation_line)
	all_lines.extend(lines[1:])
	doc_table = "\n".join(all_lines)

doc_expander = st.beta_expander(label="Show/Hide Documentation", expanded=True)
with doc_expander:
	st.markdown("## Overview")
	st.markdown(doc_paragraph)
	st.markdown("## Details")
	st.markdown(doc_table)






# Setting some of the color schemes and formatting of the page. 
# E7FD8E light green
# 90918A middle gray
# 952A53 eggplant
# B3DE98 green
# FF8B00 orange
# E4F084 yellow
st.markdown(
	"""
	<style>
	body {
		color: #111;
		background-color: #fff;
	}
	.reportview-container .markdown-text-container {
		font-family: arial;
	}
	.sidebar .sidebar-content {
		background-image: linear-gradient(#FD951A,#E4F084);
		color: black;
	}
	.Widget>label {
		color: black;
		font-family: arial;
	}
	[class^="st-b"]  {
		color: black;
		font-family: arial;
	}
	.st-bb {
		background-color: transparent;
	}
	.st-at {
		background-color: transparent;
	}
	footer {
		color: black;
		font-family: times;
	}
	.reportview-container .main footer, .reportview-container .main footer a {
		color: #fff;
	}
	header .decoration {
		background-image: none;
	}
	""",
	unsafe_allow_html=True,
)















def truncate_string(text, char_limit):
	"""Return a truncated version of the text and adding elipses if it's longer than the character limit.
	
	Args:
		text (TYPE): Description
		char_limit (TYPE): Description
	
	Returns:
		TYPE: Description
	"""
	truncated_text = text[:char_limit]
	if len(text)>char_limit:
		# Make extra room for the "..." string and then add it to the end.
		truncated_text = text[:char_limit-3]
		truncated_text = "{}...".format(truncated_text)
	return(truncated_text)









def gene_name_search(dataset, gene_name):
	"""Helper function for searching the dataset for a gene identifier.
	
	Args:
	    dataset (TYPE): Description
	    gene_name (TYPE): Description
	
	Returns:
	    TYPE: Description
	"""
	gene_name = gene_name.lower().strip()
	species_to_gene_id_list = defaultdict(list)
	for species in dataset.get_species():
		gene_ids = dataset.get_species_to_name_to_ids_dictionary(include_synonyms=True, lowercase=True)[species][gene_name]
		for gene_id in gene_ids:
			species_to_gene_id_list[species].append(gene_id)
	species_to_gene_id_list = {s:list(set(l)) for s,l in species_to_gene_id_list.items()}
	return(species_to_gene_id_list)



	

































@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def initial_setup():


	# Reading in the pickled ontology objects.
	ontologies = {name:load_from_pickle(path) for name,path in zip(ONTOLOGY_NAMES,ONTOLOGY_PICKLE_PATHS)}



	# Loading the dataset from the oats object and retaining all the mappings that are specified there.
	dataset = Dataset(data=DATASET_PATH, keep_ids=True)
	dataset.filter_has_description()


	df = dataset.to_pandas()



	# Building the annotations dictionary. This should actually be done here outside the search sections and only run once.
	gene_id_to_annotations = dataset.get_annotations_dictionary()





	# Load mappings that are specified in this dataset. These are used to put the results from different types of queries together below.
	# The way this is done here is dependent on how the dataset object is constructed here but the resulting dictionaries could be 
	# constructed any other way, what matters is that they exist below this point. 
	df[COLUMN_NAMES["gene"]] = df["id"].map(lambda x: dataset.get_gene_dictionary()[x].primary_identifier)
	df[COLUMN_NAMES["model"]] = df["id"].map(lambda x: dataset.get_gene_dictionary()[x].gene_models[0] if len(dataset.get_gene_dictionary()[x].gene_models)>0 else "")
	df[COLUMN_NAMES["species"]] = df["species"].map(TO_SPECIES_DISPLAY_NAME)


	gene_id_to_gene_identifier = dict(zip(df["id"].values, df[COLUMN_NAMES["gene"]].values))
	gene_id_to_gene_model = dict(zip(df["id"].values, df[COLUMN_NAMES["model"]].values))
	gene_id_to_species = dict(zip(df["id"].values, df[COLUMN_NAMES["species"]].values))





	gene_id_to_description = dataset.get_description_dictionary()




	# Create mappings between genes and unique sentences in the dataset and also the reverse mappings.
	sentences = set()
	for gene_id,desciption in gene_id_to_description.items():
		sentences_in_text = sent_tokenize(desciption)
		sentences.update(sentences_in_text)
	sentences = list(sentences)
	sentence_id_to_sentence = {i:s for i,s in enumerate(sentences)}
	sentence_to_sentence_id = {s:i for i,s in sentence_id_to_sentence.items()}
	gene_id_to_sentences_ids = defaultdict(list)
	for gene_id,desciption in gene_id_to_description.items():
		sentences_in_text = sent_tokenize(desciption)
		gene_id_to_sentences_ids[gene_id].extend([sentence_to_sentence_id[s] for s in sentences_in_text])



	# Mapping to text strings that have already been preprocessed in some specified way.
	gene_id_to_preprocessed_description = {i:" ".join(preprocess_string(s)) for i,s in gene_id_to_description.items()}
	sentence_id_to_preprocessed_sentence = {i:" ".join(preprocess_string(s)) for i,s in sentence_id_to_sentence.items()}

	# Processing the full descriptions or sentences in a way that allows for keyword matching to be done very fast.
	gene_id_to_description_for_keyword_matching  = {i:PREPROCESSING_FOR_KEYWORD_SEARCH_FUNCTION(s) for i,s in gene_id_to_description.items()}
	sentence_id_to_sentence_for_keyword_matching = {i:PREPROCESSING_FOR_KEYWORD_SEARCH_FUNCTION(s) for i,s in sentence_id_to_sentence.items()}

	# These dictionaries have noresults to do with processing, but are used for displaying text in a readable and controlled way in the plottly table. 
	gene_id_to_descriptions_one_line_truncated = {i:truncate_string(d, DESCRIPTION_COLUMN_WIDTH) for i,d in gene_id_to_description.items()}
	gene_id_to_descriptions_with_newline_tokens = {i:NEWLINE_TOKEN.join(wrap(d, DESCRIPTION_COLUMN_WIDTH)) for i,d in gene_id_to_description.items()}
	sentence_id_to_sentences_one_line_truncated = {i:truncate_string(s, DESCRIPTION_COLUMN_WIDTH) for i,s in sentence_id_to_sentence.items()}
	sentence_id_to_sentences_with_newline_tokens = {i:NEWLINE_TOKEN.join(wrap(s, DESCRIPTION_COLUMN_WIDTH)) for i,s in sentence_id_to_sentence.items()}









	# Load the model either from a pickle or have to run the preprocessing steps again.
	if os.path.exists(WORD_EMBEDDINGS_PICKLE_PATH):
		model = load_from_pickle(WORD_EMBEDDINGS_PICKLE_PATH)
	else:
		model = TokenSimilarities(WORD2VEC_MODEL_PATH)
		save_to_pickle(model, WORD_EMBEDDINGS_PICKLE_PATH)




	# Load or find the document embeddings from the mean word embeddings.
	sentence_id_to_mean_word2vec_embedding = {i:model.get_mean_embedding(text.split()) for i,text in sentence_id_to_preprocessed_sentence.items()}
	#if os.path.exists(SENT_EMBEDDINGS_FROM_WORD2VEC_PATH):
	#	sentence_id_to_mean_word2vec_embedding = load_from_pickle(SENT_EMBEDDINGS_FROM_WORD2VEC_PATH)
	#else:
	#	sentence_id_to_mean_word2vec_embedding = {i:model.get_mean_embedding(text.split()) for i,text in sentence_id_to_preprocessed_sentence.items()}
	#	save_to_pickle(sentence_id_to_mean_word2vec_embedding, SENT_EMBEDDINGS_FROM_WORD2VEC_PATH)




	# Load or find the document embeddings inferred using the Doc2Vec model.
	doc2vec_model = gensim.models.Doc2Vec.load(DOC2VEC_MODEL_PATH)
	sentence_id_to_doc2vec_embedding = {i:doc2vec_model.infer_vector(text.lower().split()) for i,text in sentence_id_to_preprocessed_sentence.items()}
	#if os.path.exists(SENT_EMBEDDINGS_FROM_DOC2VEC_PATH):
	#	sentence_id_to_doc2vec_embedding = load_from_pickle(SENT_EMBEDDINGS_FROM_DOC2VEC_PATH)
	#else:
	#	sentence_id_to_doc2vec_embedding = {i:doc2vec_model.infer_vector(text.lower().split()) for i,text in sentence_id_to_preprocessed_sentence.items()}
	#	save_to_pickle(sentence_id_to_doc2vec_embedding, SENT_EMBEDDINGS_FROM_DOC2VEC_PATH)
		
	

	# Load or find the document embeddings inferred using tf-idf.
	tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)
	tfidf_vectorizer.fit(gene_id_to_preprocessed_description.values())
	sentence_id_to_tfidf_embedding = {i:tfidf_vectorizer.transform([text]).toarray()[0] for i,text in sentence_id_to_preprocessed_sentence.items()}
	#if os.path.exists(SENT_EMBEDDINGS_FROM_TFIDF_PATH):
	#	sentence_id_to_tfidf_embedding = load_from_pickle(SENT_EMBEDDINGS_FROM_TFIDF_PATH)
	#else:
	#	sentence_id_to_tfidf_embedding = {i:tfidf_vectorizer.transform([text]).toarray()[0] for i,text in sentence_id_to_preprocessed_sentence.items()}
	#	save_to_pickle(sentence_id_to_tfidf_embedding, SENT_EMBEDDINGS_FROM_TFIDF_PATH)





	approaches = {
		"tfidf":{
			"name":"TF-IDF",
			"sentence_id_to_embedding":sentence_id_to_tfidf_embedding, 
			"vectorization_function":lambda x: tfidf_vectorizer.transform([x]).toarray()[0],
			"preprocessing_function":lambda x: " ".join(preprocess_string(x)),
			"threshold":0.6, 
		},
		"doc2vec":{
			"name":"Doc2Vec",
			"sentence_id_to_embedding":sentence_id_to_doc2vec_embedding, 
			"vectorization_function":lambda x: doc2vec_model.infer_vector(x.lower().split()), 
			"preprocessing_function":lambda x: " ".join(preprocess_string(x)),
			"threshold":0.6,
		},
		"word2vec":{
			"name":"Word2Vec",
			"sentence_id_to_embedding":sentence_id_to_mean_word2vec_embedding,
			"vectorization_function":lambda x: model.get_mean_embedding(x),
			"preprocessing_function":lambda x: preprocess_string(x),
			"threshold":0.6,
		}
	}








	# What are the results that need to be used later? These are all the dictionaries or objects that just need to be created once during initial setup and 
	# then remain unchanged when queries are processed or when results are displayed, they just need to be referenced.
	return(
		dataset,
		ontologies,
		gene_id_to_description,
		gene_id_to_annotations,
		gene_id_to_gene_identifier,
		gene_id_to_gene_model,
		gene_id_to_species,
		sentence_id_to_sentence,
		sentence_to_sentence_id,
		gene_id_to_sentences_ids,
		gene_id_to_preprocessed_description,
		sentence_id_to_preprocessed_sentence,
		gene_id_to_description_for_keyword_matching,
		sentence_id_to_sentence_for_keyword_matching,
		gene_id_to_descriptions_one_line_truncated,
		gene_id_to_descriptions_with_newline_tokens,
		sentence_id_to_sentences_one_line_truncated,
		sentence_id_to_sentences_with_newline_tokens,
		model,
		approaches,
		)









# Make sure to just copy and paste this directly from the returned values from the cached setup function, they can be identically names to make this easier.
(
dataset,
ontologies,
gene_id_to_description,
gene_id_to_annotations,
gene_id_to_gene_identifier,
gene_id_to_gene_model,
gene_id_to_species,
sentence_id_to_sentence,
sentence_to_sentence_id,
gene_id_to_sentences_ids,
gene_id_to_preprocessed_description,
sentence_id_to_preprocessed_sentence,
gene_id_to_description_for_keyword_matching,
sentence_id_to_sentence_for_keyword_matching,
gene_id_to_descriptions_one_line_truncated,
gene_id_to_descriptions_with_newline_tokens,
sentence_id_to_sentences_one_line_truncated,
sentence_id_to_sentences_with_newline_tokens,
model,
approaches,
) = initial_setup()















############# The Sidebar ###############

# Option for subsetting the data to only include certain species.
st.sidebar.markdown("### Filtering by Species")
species_display_names = [TO_SPECIES_DISPLAY_NAME[x] for x in dataset.get_species()]
species_list = st.sidebar.multiselect(label="Filter to only include certain species", options=species_display_names, default=species_display_names)
if len(species_list) == 0:
	species_list = species_display_names


# Options that are general and apply to all the types of queries.
st.sidebar.markdown("### General Options for all Queries")
TRUNCATED_TEXT_LABEL = "Truncate phenotype descriptions" 
truncate = st.sidebar.checkbox(label=TRUNCATED_TEXT_LABEL, value=True)
max_number_of_genes_to_show = st.sidebar.number_input("Maximum number of genes to include in results", min_value=1, max_value=None, value=50, step=50)
TABLE_WIDTH = st.sidebar.slider(label="Table width in pixels", min_value=400, max_value=8000, value=2000, step=100, format=None, key=None)


# Options that are specific to a particular query should go in their own section to make that clear.
st.sidebar.markdown("### Gene Identifier Query Options")
synonyms = st.sidebar.checkbox(label="Show possible gene synonyms", value=False)



approach = st.sidebar.selectbox("Method of comparing phenotype descriptions", tuple(approaches.keys()), format_func=lambda x: {k:v["name"] for k,v in approaches.items()}[x])



ref_text = "If you find this tool useful in your own research, please cite this page."

contact_text = "If you have feedback about the tool or experience issues while using it, please contact irbraun@iastate.edu or szarecor@iastate.edu."


st.sidebar.markdown("### Citation")
st.sidebar.markdown(ref_text)
st.sidebar.markdown("### Contact")
st.sidebar.markdown(contact_text)



















############# The Searchbar Section ###############


# Display the search section of the main page.
st.markdown("## Search")

search_types = ["freetext", "keywords", "terms", "identifiers"]
search_type_labels = ["Free Text", "Keywords & Keyphrases", "Ontology Terms", "Gene Identifiers"]
search_type_label_map = {t:l for t,l in zip(search_types,search_type_labels)}
search_types_format_func = lambda x: search_type_label_map[x]
search_type = st.radio(label="Select a type of search", options=search_types, index=0, format_func=search_types_format_func)
input_text = st.text_input(label="Enter text here")



















def get_column_explanation_table(column_keys_and_explanations):
	"""Formats a string for a markdown table explaining columns in the results.
	
	Args:
	    column_keys_and_explanations (TYPE): Description
	
	Returns:
	    TYPE: Description
	"""
	table_rows = []
	table_rows.append("|Column|Contents")
	table_rows.append("|-|-|")
	for (key,explanation) in column_keys_and_explanations:
		table_rows.append("|{}|{}|".format(COLUMN_NAMES[key].replace("<b>",""),explanation))
	table_string = "\n".join(table_rows)
	return(table_string)










def display_download_links(df, column_keys, column_keys_to_unwrap, column_keys_to_list, num_rows):
	"""Formats and presents the links for downloading results.
	
	Args:
	    df (TYPE): Description
	    column_keys (TYPE): Description
	    column_keys_to_unwrap (TYPE): Description
	    column_keys_to_list (TYPE): Description
	    num_rows (TYPE): Description
	"""


	# Subsetting the dataframe to only contain the indicated number of rows, assumes it is already in the desired order.
	subsets_df = df[[COLUMN_NAMES[x] for x in column_keys]].head(num_rows)

	# If there were columns that used newline tokens to wrap lines, remove those tokens before downloading.
	for key in column_keys_to_unwrap:
		subsets_df[COLUMN_NAMES[key]] = subsets_df[COLUMN_NAMES[key]].map(lambda x: x.replace(NEWLINE_TOKEN," "))

	# If there were columns that were using newline tokens to separate lists of something like that, represent those with a semicolon instead.
	for key in column_keys_to_list:
		subsets_df[COLUMN_NAMES[key]] = subsets_df[COLUMN_NAMES[key]].map(lambda x: "{}".format("; ".join(x.split(NEWLINE_TOKEN))))



	# Presenting a download link for a tsv file that contains everything in the output table.
	full = subsets_df.copy(deep=True)
	full.rename(COLUMN_NAMES_TO_OUTPUT_COLUMN_NAME, inplace=True, axis="columns")
	tsv = full.to_csv(index=False, sep="\t")
	b64 = base64.b64encode(tsv.encode()).decode() 
	link = f'<a href="data:file/tsv;base64,{b64}" download="query_results.tsv">Complete tab-separated dataset</a>'
	st.markdown(link, unsafe_allow_html=True)


	# Presenting a download link for a tsv file that contains just the list of gene identifiers in the ranked order.
	genes_only = subsets_df.copy(deep=True)
	column_keys_to_keep = ["rank","species","gene","model"]
	genes_only = genes_only[[COLUMN_NAMES[x] for x in column_keys_to_keep]]
	genes_only = genes_only.drop_duplicates(inplace=False)
	genes_only.rename(COLUMN_NAMES_TO_OUTPUT_COLUMN_NAME, inplace=True, axis="columns")
	tsv = genes_only.to_csv(index=False, sep="\t")
	b64 = base64.b64encode(tsv.encode()).decode() 
	link = f'<a href="data:file/tsv;base64,{b64}" download="query_results.tsv">List of genes only</a>'
	st.markdown(link, unsafe_allow_html=True)






def download_output_table(df, column_keys, column_keys_to_unwrap, column_keys_to_list, num_rows, output_path):
	"""Download an output results table to a specified output path, used when running as a script.
	
	Args:
	    df (TYPE): Description
	    column_keys (TYPE): Description
	    column_keys_to_unwrap (TYPE): Description
	    column_keys_to_list (TYPE): Description
	    num_rows (TYPE): Description
	    output_path (TYPE): Description
	"""

	# Subsetting the dataframe to only contain the indicated number of rows, assumes it is already in the desired order.
	subsets_df = df[[COLUMN_NAMES[x] for x in column_keys]].head(num_rows)

	# If there were columns that used newline tokens to wrap lines, remove those tokens before downloading.
	for key in column_keys_to_unwrap:
		subsets_df[COLUMN_NAMES[key]] = subsets_df[COLUMN_NAMES[key]].map(lambda x: x.replace(NEWLINE_TOKEN," "))

	# If there were columns that were using newline tokens to separate lists of something like that, represent those with a semicolon instead.
	for key in column_keys_to_list:
		subsets_df[COLUMN_NAMES[key]] = subsets_df[COLUMN_NAMES[key]].map(lambda x: "{}".format("; ".join(x.split(NEWLINE_TOKEN))))


	# Presenting a download link for a tsv file that contains everything in the output table.
	full = subsets_df.copy(deep=True)
	full.rename(COLUMN_NAMES_TO_OUTPUT_COLUMN_NAME, inplace=True, axis="columns")
	full.to_csv(output_path, index=False, sep="\t")











def display_plottly_dataframe(df, column_keys, column_keys_to_wrap, num_rows):
	"""Formats and presents the plotly table.
	
	Args:
	    df (TYPE): Description
	    column_keys (TYPE): Description
	    column_keys_to_wrap (TYPE): Description
	    num_rows (TYPE): Description
	"""
	my_df = df[[COLUMN_NAMES[x] for x in column_keys]].head(num_rows)

	header_values = my_df.columns
	cell_values = []
	for index in range(0, len(my_df.columns)):
		cell_values.append(my_df.iloc[:,index:index+1])




	# Shouldn't have to do it this way, but we do. There is a bug with inserting the <br> tags any other way than in strings specified in this way.
	# For some reason, HTML tags present before this point are not recognized, I haven't figured out why. The newline token is a special token in 
	# this script that has multiple uses, and that token needs to translated to the tag that actually produces a newline in the plotly table here.
	indices_of_columns_to_wrap = [column_keys.index(x) for x in column_keys_to_wrap]
	for col_idx,col_key in zip(indices_of_columns_to_wrap,column_keys_to_wrap):
		contents = list(cell_values[col_idx][COLUMN_NAMES[col_key]].values)
		contents = [x.replace(NEWLINE_TOKEN,"<br>") for x in contents]
		cell_values[col_idx] = contents



	# Pick background colors for the rows by alternating with repsect to a value in one of the columns.
	# This is intended to be used for making the rows that refer to a particular gene all be the same background color.
	# This way when the data is sorted so all those rows are together there will be an alternating effect.
	values_in_column_to_base_row_color_on = my_df[COLUMN_NAMES[ALTERNATE_ROW_COLOR_BASED_ON_COLUMN_KEY]].values
	fill_colors = []
	num_to_row_color = {0:TABLE_ROWS_COLOR_1, 1:TABLE_ROWS_COLOR_2}
	last = values_in_column_to_base_row_color_on[0]
	ctr = 2
	for v in values_in_column_to_base_row_color_on:
		if v != last:
			ctr = ctr+1
		last = v
		fill_colors.append(num_to_row_color[ctr%2])
	fill_colors = [fill_colors*len(column_keys)]



	# Display the table as a plotly figure with the provided formatting, with dimensions specified by constants and the dynamically generated color lists.
	fig = go.Figure(data=[go.Table(
		columnorder = list(range(len(column_keys))),
		columnwidth = [COLUMN_WIDTHS[x] for x in column_keys],
		header=dict(values=header_values, fill_color=TABLE_HEADER_COLOR, align="left", font=dict(color='black', size=14), height=HEADER_HEIGHT),
		cells=dict(values=cell_values, fill_color=fill_colors, align="left", font=dict(color='black', size=14)),
		)])
	fig.update_layout(width=TABLE_WIDTH, height=TABLE_HEIGHT)
	st.plotly_chart(fig)
























threshold = 0.51


############### Allowing this file to be run as a normal Python script instead of a streamlit application for testing #############




if not st._is_running_with_streamlit:

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--type", "-t", dest="type", required=True, choices=search_types)
	parser.add_argument("--query", "-q", dest="query", required=True)
	parser.add_argument("--limit", "-l", dest="limit", required=True, type=int)
	parser.add_argument("--output", "-o", dest="output", required=True)
	parser.add_argument("--species", "-s", dest="species", required=False, choices=species_display_names)
	parser.add_argument("--threshold", "-r", dest="threshold", required=False, type=float)
	parser.add_argument("--algorithm", "-a", dest="algorithm", required=False, type=str)

	args = parser.parse_args()
	search_type = args.type
	input_text = args.query
	output_path = args.output
	max_number_of_genes_to_show = args.limit
	if args.species is None:
		species_list = species_display_names
	else:
		species_list = [args.species]

	# Overwrite all the thresholds if one was provided.
	if args.threshold is not None:
		threshold = args.threshold
		for approach in approaches:
			approaches[approach]["threshold"] = threshold
	if args.algorithm is not None:
		approach = args.algorithm.lower()


	# Making some changes that are specific to the running this as a script.
	truncate = False





























############### Listening for something to be typed and entered into the search field #############











#    ********  ******** ****     ** ********   ** *******   ******** ****     ** ********** ** ******** ** ******** *******  
#   **//////**/**///// /**/**   /**/**/////   /**/**////** /**///// /**/**   /**/////**/// /**/**///// /**/**///// /**////** 
#  **      // /**      /**//**  /**/**        /**/**    /**/**      /**//**  /**    /**    /**/**      /**/**      /**   /** 
# /**         /******* /** //** /**/*******   /**/**    /**/******* /** //** /**    /**    /**/******* /**/******* /*******  
# /**    *****/**////  /**  //**/**/**////    /**/**    /**/**////  /**  //**/**    /**    /**/**////  /**/**////  /**///**  
# //**  ////**/**      /**   //****/**        /**/**    ** /**      /**   //****    /**    /**/**      /**/**      /**  //** 
#  //******** /********/**    //***/********  /**/*******  /********/**    //***    /**    /**/**      /**/********/**   //**
#   ////////  //////// //      /// ////////   // ///////   //////// //      ///     //     // //       // //////// //     // 



if search_type == "identifiers" and input_text != "":


	# Start the results section of the page.
	st.markdown("## Results")
	

	# There is a different way of finding the specific gene if this is being run as a script.
	if not st._is_running_with_streamlit:


		# Do the actual processing of the search against the full dataset here.
		gene_search_string = input_text
		assert len(gene_search_string.split(":")) == 2
		species_string = gene_search_string.split(":")[0]
		gene_identifier_string = gene_search_string.split(":")[1]
		gene_matches = gene_name_search(dataset=dataset, gene_name=gene_identifier_string)
		gene_matches = {species:id_list for species,id_list in gene_matches.items() if TO_SPECIES_DISPLAY_NAME[species].lower() == species_string.lower()}
		assert len(gene_matches) == 1
		matching_gene_ids_list = gene_matches[list(gene_matches.keys())[0]]
		if len(matching_gene_ids_list) > 1:
			print("ambiguous")
		else:
			i = matching_gene_ids_list[0]


			# Get information about which gene from the dataset was selected.
			selected_gene_primary_name = dataset.get_gene_dictionary()[i].primary_identifier
			selected_gene_other_names = dataset.get_gene_dictionary()[i].all_identifiers
			selected_gene_phenotype_description = dataset.get_description_dictionary()[i]
		

			search_string = gene_id_to_description[i]

			ids_subset = [i for i,species in gene_id_to_species.items() if species in species_list]

			with st.spinner("Searching..."):
				results = qh.handle_free_text_query_with_precomputed_sentence_embeddings(
					search_string = search_string,
					max_genes = max_number_of_genes_to_show,
					sent_tokenize_f = nltk.sent_tokenize,
					preprocess_f = approaches[approach]["preprocessing_function"],
					vectorization_f = approaches[approach]["vectorization_function"],
					s_id_to_s = sentence_id_to_sentence,
					s_id_to_s_embedding = approaches[approach]["sentence_id_to_embedding"],
					g_id_to_s_ids = gene_id_to_sentences_ids,
					threshold = approaches[approach]["threshold"],
					first=i,
					ids_subset=ids_subset)



				# Creating the formatted version of each column that are pulled either directly from the ones used for processing the query or 
				# the previously constructed dictionaries that map gene IDs to other information.
				results[COLUMN_NAMES["rank"]] = results["rank"]
				results[COLUMN_NAMES["query_sentence"]] = results["q"] 
				results[COLUMN_NAMES["score"]] = results["score"].map(lambda x: "{:.3f}".format(x))
				results[COLUMN_NAMES["matching_sentence"]] = results["sentence"] 
				results[COLUMN_NAMES["gene"]] = results["gene_id"].map(gene_id_to_gene_identifier)
				results[COLUMN_NAMES["species"]] = results["gene_id"].map(gene_id_to_species)
				results[COLUMN_NAMES["model"]] = results["gene_id"].map(gene_id_to_gene_model)
				results[COLUMN_NAMES["internal_id"]] = results["gene_id"]



				# Subsetting to only include a particular set of species.
				results = results[results[COLUMN_NAMES["species"]].isin(species_list)]


				# Formatting the columns correctly and truncating ones that wrap on multiple lines if the table is compressed.
				results[COLUMN_NAMES["matching_sentence"]] = results["sentence_id"].map(sentence_id_to_sentences_with_newline_tokens)
				if truncate:
					results[COLUMN_NAMES["matching_sentence"]] = results["sentence_id"].map(sentence_id_to_sentences_one_line_truncated)




				# Check to make sure that atleast some rows are left after all the filtering steps are complete.
				if results.shape[0] == 0:
					st.markdown("No genes were found that have phenotypes described similarly to this query.")


				
				# Display the download options and the plottly table of the results if there were any rows remaining.
				else:
					
					# Show the subset of columns that is relevant to this search.
					columns_to_include_keys = ["rank", "species", "gene", "model", "score", "query_sentence", "matching_sentence"]
					columns_to_include_keys_and_wrap = ["matching_sentence"]
					column_keys_to_unwrap = []
					if truncate == False:
						column_keys_to_unwrap = ["matching_sentence"]
					column_keys_to_list = []
					num_rows = results.shape[0]




	# The regular version of this query, done inside the streamlit application.
	else:


		# Do the actual processing of the search against the full dataset here.
		gene_search_string = input_text
		gene_matches = gene_name_search(dataset=dataset, gene_name=gene_search_string)

		# Search text was entered and the search was processed but the list of relevant IDs found is empty.
		if len(gene_matches)==0:
			st.markdown("No genes were found for '{}'.".format(gene_search_string))

		# Search text was entered and the search was processed and atleast one matching ID was found.
		gene_buttons_dict = {}
		if len(gene_matches)>0:
			st.markdown("Genes matching '{}' are shown below. Select one to see other genes with similarly described phenotypes.".format(gene_search_string))
			unique_button_key = 1
			for species,id_list in gene_matches.items():
				for i in id_list:
					primary_gene_name = dataset.get_gene_dictionary()[i].primary_identifier
					other_names = dataset.get_gene_dictionary()[i].all_identifiers
					button_label = "{}: {}".format(TO_SPECIES_DISPLAY_NAME[species], primary_gene_name)
					gene_buttons_dict[i] = st.button(label=button_label, key=unique_button_key)
					unique_button_key = unique_button_key+1
					if synonyms:
						synonyms_field_char_limit = 150
						synonyms_field_str = truncate_string(", ".join(other_names), synonyms_field_char_limit)
						st.markdown("(Other synonyms include {})".format(synonyms_field_str))





		# Handle what should happen if any of the previously presented gene buttons was clicked.
		# Has to be a loop because we need to check all the presented buttons, and there might be more than one.
		for i,gene_button in gene_buttons_dict.items():
			if gene_button:
				
				# Get information about which gene from the dataset was selected.
				selected_gene_primary_name = dataset.get_gene_dictionary()[i].primary_identifier
				selected_gene_other_names = dataset.get_gene_dictionary()[i].all_identifiers
				selected_gene_phenotype_description = dataset.get_description_dictionary()[i]
				




				# Add an expander section for how the view of the results can be customized. This includes the number of genes shown.
				expander = st.beta_expander(label="Show/Hide Gene Information")
				with expander:
					st.markdown("**Identifier:** {}".format(selected_gene_primary_name))
					st.markdown("**Possible Synonym(s):** {}".format(", ".join(selected_gene_other_names)))
					st.markdown("**Phenotype Description(s):** {}".format(selected_gene_phenotype_description))

				




				search_string = gene_id_to_description[i]

				ids_subset = [i for i,species in gene_id_to_species.items() if species in species_list]

				with st.spinner("Searching..."):
					results = qh.handle_free_text_query_with_precomputed_sentence_embeddings(
						search_string = search_string,
						max_genes = max_number_of_genes_to_show,
						sent_tokenize_f = nltk.sent_tokenize,
						preprocess_f = approaches[approach]["preprocessing_function"],
						vectorization_f = approaches[approach]["vectorization_function"],
						s_id_to_s = sentence_id_to_sentence,
						s_id_to_s_embedding = approaches[approach]["sentence_id_to_embedding"],
						g_id_to_s_ids = gene_id_to_sentences_ids,
						threshold = approaches[approach]["threshold"],
						first=i,
						ids_subset=ids_subset)






					# Creating the formatted version of each column that are pulled either directly from the ones used for processing the query or 
					# the previously constructed dictionaries that map gene IDs to other information.
					results[COLUMN_NAMES["rank"]] = results["rank"]
					results[COLUMN_NAMES["query_sentence"]] = results["q"] 
					results[COLUMN_NAMES["score"]] = results["score"].map(lambda x: "{:.3f}".format(x))
					results[COLUMN_NAMES["matching_sentence"]] = results["sentence"] 
					results[COLUMN_NAMES["gene"]] = results["gene_id"].map(gene_id_to_gene_identifier)
					results[COLUMN_NAMES["species"]] = results["gene_id"].map(gene_id_to_species)
					results[COLUMN_NAMES["model"]] = results["gene_id"].map(gene_id_to_gene_model)
					results[COLUMN_NAMES["internal_id"]] = results["gene_id"]



					# Subsetting to only include a particular set of species.
					results = results[results[COLUMN_NAMES["species"]].isin(species_list)]




					# Formatting the columns correctly and truncating ones that wrap on multiple lines if the table is compressed.
					results[COLUMN_NAMES["matching_sentence"]] = results["sentence_id"].map(sentence_id_to_sentences_with_newline_tokens)
					if truncate:
						results[COLUMN_NAMES["matching_sentence"]] = results["sentence_id"].map(sentence_id_to_sentences_one_line_truncated)




					# Check to make sure that atleast some rows are left after all the filtering steps are complete.
					if results.shape[0] == 0:
						st.markdown("No genes were found that have phenotypes described similarly to this query.")


					
					# Display the download options and the plottly table of the results if there were any rows remaining.
					else:
						
						# Show the subset of columns that is relevant to this search.
						columns_to_include_keys = ["rank", "species", "gene", "model", "score", "query_sentence", "matching_sentence"]
						columns_to_include_keys_and_wrap = ["matching_sentence"]
						column_keys_to_unwrap = []
						if truncate == False:
							column_keys_to_unwrap = ["matching_sentence"]
						column_keys_to_list = []
						num_rows = results.shape[0]





						# Create the expanded section for explaning what the columns are.
						expander = st.beta_expander(label="Show/Hide Explanation of Columns", expanded=True)
						with expander:
							column_keys_and_explanations = [
							("rank", "Genes are ranked first by the maximum score for any query sentence to a sentence associated with this gene, and then by the mean score for all query sentences."),
							("score","The average similarity between each word in the query sentence and the most similar word in the phenotype description sentence."),
							("query_sentence","A sentence from the query."),
							("matching_sentence","A sentence from the phenotype descriptions associated with this gene. These sentences are truncated by default so as to only take up one line in the table. Uncheck the '{}' option to expand them.".format(TRUNCATED_TEXT_LABEL)),
							]
							st.markdown(get_column_explanation_table(column_keys_and_explanations))




						# Create the expanded section for presenting download options.
						expander = st.beta_expander(label="Show/Hide Download Options", expanded=False)
						with expander:
							display_download_links(results, columns_to_include_keys, column_keys_to_unwrap, column_keys_to_list, num_rows)

						# Show the main plottly table.
						display_plottly_dataframe(results, columns_to_include_keys, columns_to_include_keys_and_wrap, num_rows)
















#  ********** ******** *******   ****     ****  ********
# /////**/// /**///// /**////** /**/**   **/** **////// 
#     /**    /**      /**   /** /**//** ** /**/**       
#     /**    /******* /*******  /** //***  /**/*********
#     /**    /**////  /**///**  /**  //*   /**////////**
#     /**    /**      /**  //** /**   /    /**       /**
#     /**    /********/**   //**/**        /** ******** 
#     //     //////// //     // //         // ////////  


elif search_type == "terms" and input_text != "":

	st.markdown("## Results")


	term_ids = input_text.replace(","," ").split()
	term_ids = list(set(term_ids))
	pattern = re.compile("[A-Z]{2,}[:_][0-9]{7}$")
	term_ids_are_valid = [bool(pattern.match(term_id)) for term_id in term_ids]
	
	if False in term_ids_are_valid:
		st.markdown("Invalid ontology term identifiers.")


	else:

		st.markdown("**Terms(s) searched**: {}".format(", ".join(term_ids)))

		# Linking out to other resources like Ontobee and Planteome.
		# TODO Verify that the links lead somewhere valid before displaying them in the application.
		ontobee_url_template = "http://www.ontobee.org/ontology/{}?iri=http://purl.obolibrary.org/obo/{}_{}"
		planteome_url_template = "http://browser.planteome.org/amigo/term/{}:{}"
		lines_with_terms_and_links = []
		for term_id in term_ids:
			term_id_str = term_id.replace(":","_")
			ontology_name, term_number = tuple(term_id_str.split("_"))
			try:
				term_label = ontologies[ontology_name.upper()][term_id].name
				ontobee_url = ontobee_url_template.format(ontology_name, ontology_name, term_number)
				planteome_url = planteome_url_template.format(ontology_name, term_number)
				line = "{} ({}, [Ontobee]({}), [Planteome]({}))".format(term_id, term_label, ontobee_url, planteome_url)
				lines_with_terms_and_links.append(line)
			except:
				pass
		lines_with_terms_and_links_str = "\n\n".join(lines_with_terms_and_links)


		expander = st.beta_expander(label="Show/Hide Additional Resources", expanded=True)
		with expander:
			st.markdown(lines_with_terms_and_links_str)





		ids_subset = [i for i,species in gene_id_to_species.items() if species in species_list]
		

		with st.spinner("Searching..."):
			results = qh.handle_annotation_query(
				term_ids = term_ids,
				max_genes = max_number_of_genes_to_show,
				g_id_to_annots = gene_id_to_annotations,
				ontologies = ontologies,
				ids_subset=ids_subset)





			# Creating the formatted version of each column that are pulled either directly from the ones used for processing the query or 
			# the previously constructed dictionaries that map gene IDs to other information.
			results[COLUMN_NAMES["rank"]] = results["rank"]
			results[COLUMN_NAMES["gene"]] = results["gene_id"].map(gene_id_to_gene_identifier)
			results[COLUMN_NAMES["species"]] = results["gene_id"].map(gene_id_to_species)
			results[COLUMN_NAMES["model"]] = results["gene_id"].map(gene_id_to_gene_model)
			results[COLUMN_NAMES["query_term_id"]] = results["query_term_id"]
			results[COLUMN_NAMES["query_term_name"]] = results["query_term_name"]
			results[COLUMN_NAMES["annotated_term_id"]] = results["annotated_term_id"]
			results[COLUMN_NAMES["annotated_term_name"]] = results["annotated_term_name"]
			results[COLUMN_NAMES["internal_id"]] = results["gene_id"]

			# Formatting the columns correctly and truncating ones that wrap on multiple lines if the table is compressed.
			if truncate:
				results[COLUMN_NAMES["phenotype"]] = results["gene_id"].map(gene_id_to_descriptions_one_line_truncated)
			else:
				results[COLUMN_NAMES["phenotype"]] = results["gene_id"].map(gene_id_to_descriptions_with_newline_tokens)



			# Filtering to include on a specific set of species in the presented rows.
			results = results[results[COLUMN_NAMES["species"]].isin(species_list)]


			# Check to see if there are no rows left after all those filtering steps.
			if results.shape[0] == 0:
				st.markdown("No genes were found for '{}'. Make sure the ontology term IDs are separated by commas or spaces and formatted like the examples.".format(term_id))

			
			else:

				# Show the subset of columns that is relevant to this search.
				columns_to_include_keys = ["rank", "species", "gene", "model", "query_term_id", "query_term_name", "annotated_term_id", "annotated_term_name"]
				columns_to_include_keys_and_wrap = []
				column_keys_to_unwrap = []
				column_keys_to_list = []
				num_rows = results.shape[0]


				# Create the expanded section for explaning what the columns are.
				expander = st.beta_expander(label="Show/Hide Explanation of Columns", expanded=True)
				with expander:
					column_keys_and_explanations = [
					("rank", "Genes are ranked by the number of differerent query terms that map to annotations for a given gene."),
					("query_term_id","An ontology term identifier from the query."),
					("query_term_name","The corresponding name of that term in the ontology."),
					("annotated_term_id","The identifier of an ontology term annotated to this gene, that is either equal to or inherits the queried term."),
					("annotated_term_name","The corresponding name of that term in the ontology.")
					]
					st.markdown(get_column_explanation_table(column_keys_and_explanations))


				# Create the expanded section for presenting download options.
				expander = st.beta_expander(label="Show/Hide Download Options", expanded=False)
				with expander:
					display_download_links(results, columns_to_include_keys, column_keys_to_unwrap, column_keys_to_list, num_rows)
				
				# Show the main dataframe of results.
				display_plottly_dataframe(results, columns_to_include_keys, columns_to_include_keys_and_wrap, num_rows)














#  **   ** ******** **    ** **       **   *******   *******   *******    ********
# /**  ** /**///// //**  ** /**      /**  **/////** /**////** /**////**  **////// 
# /** **  /**       //****  /**   *  /** **     //**/**   /** /**    /**/**       
# /****   /*******   //**   /**  *** /**/**      /**/*******  /**    /**/*********
# /**/**  /**////     /**   /** **/**/**/**      /**/**///**  /**    /**////////**
# /**//** /**         /**   /**** //****//**     ** /**  //** /**    **        /**
# /** //**/********   /**   /**/   ///** //*******  /**   //**/*******   ******** 
# //   // ////////    //    //       //   ///////   //     // ///////   ////////  




# Handling keyword and keyphrase queries.
elif search_type == "keywords" and input_text != "":


	# Modifying the keywords and phrases in way that makes them easy to compare against the preprocessed descripitons.
	search_kws = input_text
	keywords = search_kws.strip().strip(punctuation).split(",")
	raw_keywords = [kw.strip() for kw in keywords]
	modified_keywords = [PREPROCESSING_FOR_KEYWORD_SEARCH_FUNCTION(kw) for kw in raw_keywords]




	# Start the results section of the page, and give a quick summary of what was queried for.
	st.markdown("## Results")
	keywords_str = ", ".join([kw for kw in raw_keywords if len(kw.split())==1])
	phrases_str = ", ".join([kw for kw in raw_keywords if len(kw.split())>1])
	st.markdown("**Word(s) searched**: {}".format(keywords_str))
	st.markdown("**Phrase(s) searched**: {}".format(phrases_str))



	phene_per_line = st.checkbox(label="Show individual matching sentences on each row (instead of showing one gene on each row)", value=False)


	ids_subset = [i for i,species in gene_id_to_species.items() if species in species_list]


	with st.spinner("Searching..."):

		# Call the function that handles free text queries and returns a dataframe, passing in all the required arguments.
		results = qh.handle_keyword_query(
			raw_keywords = raw_keywords,
			modified_keywords = modified_keywords,
			max_genes = max_number_of_genes_to_show,
			phene_per_line = phene_per_line,
			s_id_to_s = sentence_id_to_sentence,
			s_id_to_kw_s = sentence_id_to_sentence_for_keyword_matching,
			g_id_to_d = gene_id_to_description,
			g_id_to_kw_d = gene_id_to_description_for_keyword_matching,
			g_id_to_s_ids = gene_id_to_sentences_ids,
			ids_subset=ids_subset)




		# Adding the formatted columns that are the same no matter what.
		results[COLUMN_NAMES["rank"]] = results["rank"]
		results[COLUMN_NAMES["keywords"]] = results["kw_found"] 
		results[COLUMN_NAMES["gene"]] = results["gene_id"].map(gene_id_to_gene_identifier)
		results[COLUMN_NAMES["species"]] = results["gene_id"].map(gene_id_to_species)
		results[COLUMN_NAMES["model"]] = results["gene_id"].map(gene_id_to_gene_model)
		results[COLUMN_NAMES["internal_id"]] = results["gene_id"]
		


		# Formatting the column that holds the text description or sentence based on what goes on one line, and if it should be truncated.
		if phene_per_line:
			if truncate:
				results[COLUMN_NAMES["matching_sentence"]] = results["sentence_id"].map(sentence_id_to_sentences_one_line_truncated)
			else:
				results[COLUMN_NAMES["matching_sentence"]] = results["sentence_id"].map(sentence_id_to_sentences_with_newline_tokens)
		else:
			if truncate:
				results[COLUMN_NAMES["matching_sentence"]] = results["gene_id"].map(gene_id_to_descriptions_one_line_truncated)
			else:
				results[COLUMN_NAMES["matching_sentence"]] = results["gene_id"].map(gene_id_to_descriptions_with_newline_tokens)




		# Subsetting for a particular set of species.
		results = results[results[COLUMN_NAMES["species"]].isin(species_list)]




		# Checking to see if there are any results at all left after all those filtering steps.
		if results.shape[0] == 0:
			st.markdown("No genes were found for '{}'. Make sure the keywords and keyphrases in this search are separated by commas.".format(search_kws))





		# If there are some rows left, display them in the formatted plottly table and present download links as well.
		else:
			

			# Show the subset of columns that is relevant to this search.
			columns_to_include_keys = ["rank", "species", "gene", "model", "keywords", "matching_sentence"]
			columns_to_include_keys_and_wrap = ["matching_sentence"]
			column_keys_to_unwrap = []
			if truncate == False:
				column_keys_to_unwrap = ["matching_sentence"]
			column_keys_to_list = []
			num_rows = results.shape[0]
			



			# Create the expanded section for explaning what the columns are.
			expander = st.beta_expander(label="Show/Hide Explanation of Columns", expanded=True)
			with expander:

				if phene_per_line:
					column_keys_and_explanations = [
					("rank", "Genes are ranked by the number of matching keywords or keyphrases"),
					("keywords","The words and phrases from the query that are present in this sentence from the phenotype descriptions for this gene."),
					("matching_sentence","A sentence from the phenotype descriptions associated with this gene. These sentences are truncated by default so as to only take up one line in the table. Uncheck the '{}' option to expand them.".format(TRUNCATED_TEXT_LABEL)),
					]
					st.markdown(get_column_explanation_table(column_keys_and_explanations))

				else:
					column_keys_and_explanations = [
					("rank", "Genes are ranked by the number of matching keywords or keyphrases."),
					("keywords","The words and phrases from the query that are present in this sentence from the phenotype descriptions for this gene."),
					("matching_sentence","All of the sentences in the phenotype descriptions associated with this gene. These sentences are truncated by default so as to only take up one line in the table. Uncheck the '{}' option to expand them.".format(TRUNCATED_TEXT_LABEL)),
					]
					st.markdown(get_column_explanation_table(column_keys_and_explanations))





			# Create the expanded section for presenting download options.
			expander = st.beta_expander(label="Show/Hide Download Options", expanded=False)
			with expander:
				display_download_links(results, columns_to_include_keys, column_keys_to_unwrap, column_keys_to_list, num_rows)
			

			# Display the main plottly table.
			display_plottly_dataframe(results, columns_to_include_keys, columns_to_include_keys_and_wrap, num_rows)















#  ******** *******   ******** ********   ********** ******** **     ** **********
# /**///// /**////** /**///// /**/////   /////**/// /**///// //**   ** /////**/// 
# /**      /**   /** /**      /**            /**    /**       //** **      /**    
# /******* /*******  /******* /*******       /**    /*******   //***       /**    
# /**////  /**///**  /**////  /**////        /**    /**////     **/**      /**    
# /**      /**  //** /**      /**            /**    /**        ** //**     /**    
# /**      /**   //**/********/********      /**    /******** **   //**    /**    
# //       //     // //////// ////////       //     //////// //     //     //     

elif search_type == "freetext" and input_text != "":


	search_string = input_text


	st.markdown("## Results")
	st.markdown("**Text searched**: {}".format(search_string))



	ids_subset = [i for i,species in gene_id_to_species.items() if species in species_list]

	with st.spinner("Searching..."):
		

		# Call the function that handles free text queries and returns a dataframe, passing in all the required arguments.
		results = qh.handle_free_text_query(
			search_string = search_string,
			max_genes = max_number_of_genes_to_show,
			sent_tokenize_f = nltk.sent_tokenize,
			preprocess_f = lambda x: " ".join(preprocess_string(x)),
			model = model,
			s_id_to_s = sentence_id_to_sentence,
			s_id_to_preprocessed_s = sentence_id_to_preprocessed_sentence,
			g_id_to_s_ids = gene_id_to_sentences_ids,
			threshold = threshold,
			ids_subset=ids_subset)







		# Creating the formatted version of each column that are pulled either directly from the ones used for processing the query or 
		# the previously constructed dictionaries that map gene IDs to other information.
		results[COLUMN_NAMES["rank"]] = results["rank"]
		results[COLUMN_NAMES["query_sentence"]] = results["q"] 
		results[COLUMN_NAMES["score"]] = results["score"].map(lambda x: "{:.3f}".format(x))
		results[COLUMN_NAMES["matching_sentence"]] = results["sentence"] 
		results[COLUMN_NAMES["gene"]] = results["gene_id"].map(gene_id_to_gene_identifier)
		results[COLUMN_NAMES["species"]] = results["gene_id"].map(gene_id_to_species)
		results[COLUMN_NAMES["model"]] = results["gene_id"].map(gene_id_to_gene_model)
		results[COLUMN_NAMES["internal_id"]] = results["gene_id"]



		# Subsetting to only include a particular set of species.
		results = results[results[COLUMN_NAMES["species"]].isin(species_list)]




		# Formatting the columns correctly and truncating ones that wrap on multiple lines if the table is compressed.
		results[COLUMN_NAMES["matching_sentence"]] = results["sentence_id"].map(sentence_id_to_sentences_with_newline_tokens)
		if truncate:
			results[COLUMN_NAMES["matching_sentence"]] = results["sentence_id"].map(sentence_id_to_sentences_one_line_truncated)




		# Check to make sure that atleast some rows are left after all the filtering steps are complete.
		if results.shape[0] == 0:
			st.markdown("No genes were found that have phenotypes described similarly to this query.")

		
		# Display the download options and the plottly table of the results if there were any rows remaining.
		else:
			
			# Show the subset of columns that is relevant to this search.
			columns_to_include_keys = ["rank", "species", "gene", "model", "score", "query_sentence", "matching_sentence"]
			columns_to_include_keys_and_wrap = ["matching_sentence"]
			column_keys_to_unwrap = []
			if truncate == False:
				column_keys_to_unwrap = ["matching_sentence"]
			column_keys_to_list = []
			num_rows = results.shape[0]


			# Create the expanded section for explaning what the columns are.
			expander = st.beta_expander(label="Show/Hide Explanation of Columns", expanded=True)
			with expander:
				column_keys_and_explanations = [
				("rank", "Genes are ranked first by the maximum score for any query sentence to a sentence associated with this gene, and then by the mean score for all query sentences."),
				("score","The average similarity between each word in the query sentence and the most similar word in the phenotype description sentence."),
				("query_sentence","A sentence from the query.."),
				("matching_sentence","A sentence from the phenotype descriptions associated with this gene. These sentences are truncated by default so as to only take up one line in the table. Uncheck the '{}' option to expand them.".format(TRUNCATED_TEXT_LABEL)),
				]
				st.markdown(get_column_explanation_table(column_keys_and_explanations))


			# Create the expanded section for presenting download options.
			expander = st.beta_expander(label="Show/Hide Download Options", expanded=False)
			with expander:
				display_download_links(results, columns_to_include_keys, column_keys_to_unwrap, column_keys_to_list, num_rows)


			# Show the main plottly table.
			display_plottly_dataframe(results, columns_to_include_keys, columns_to_include_keys_and_wrap, num_rows)














# Nothing was searched. Default to not showing anything and waiting for one of the interactive widgets to change value.
else:
	pass
	# Should the whole dataset be displayed here instead?
	# Keeping the else pass as a reminder that some default information could be displayed rather than nothing.




# If this is being run as a script, send the output table to the specified output path.
if (not st._is_running_with_streamlit) and (results.shape[0]>0):
	columns_to_include_keys.append("internal_id")
	download_output_table(results, columns_to_include_keys, column_keys_to_unwrap, column_keys_to_list, num_rows, output_path)








