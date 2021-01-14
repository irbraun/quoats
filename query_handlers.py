import pandas as pd
import numpy as np
import sys
from scipy.spatial.distance import cosine


sys.path.append("../oats")
import oats
from oats.utils.utils import flatten






def _get_sentence_scores_for_query_string(query_sentence, sentence_id_to_sentence, model):
	"""Supporting function for the free text query function. This uses word embeddings to 
	find the similarity between a given query sentence and all the other sentences in some
	passed in dictionary. The scores are calculated as the average of the maximum similarities
	between each word in the input query and all the words the sentence being compared to it.
	All the text that is passed in should already be whitespace separated and preprocessed.
	The tokens present in the input are looked up in the word embedding model as-is.
	
	Args:
	    query_sentence (str): A string of whitespace separated tokens.
	    sentence_id_to_sentence (dict of int:str): Mapping between IDs and string to be compared to the query.
	    model (TYPE): Description
	
	Returns:
	    TYPE: Description
	"""
	sentence_id_to_score = {}
	for sentence_id,sentence in sentence_id_to_sentence.items():
		tokens_in_sentence = sentence.split()
		tokens_in_query = query_sentence.split()

		maximum_scores_for_each_token = []
		for query_token in tokens_in_query:
			scores_for_this_token = [model.similarity(query_token, token) for token in tokens_in_sentence]
			if len(scores_for_this_token)>0:
				max_score_for_this_token = np.max(scores_for_this_token)
			else:
				max_score_for_this_token = 0.00
			maximum_scores_for_each_token.append(max_score_for_this_token)
		mean_max_score_for_all_tokens_in_query = np.mean(maximum_scores_for_each_token)
		sentence_id_to_score[sentence_id] = mean_max_score_for_all_tokens_in_query
	return(sentence_id_to_score)






def handle_free_text_query(search_string, max_genes, sent_tokenize_f, preprocess_f, model, s_id_to_s, s_id_to_preprocessed_s, g_id_to_s_ids, threshold, ids_subset=None):
	"""Summary
	
	Args:
	    search_string (TYPE): Description
	    max_genes (TYPE): Description
	    sent_tokenize_f (TYPE): Description
	    preprocess_f (TYPE): Description
	    model (TYPE): Description
	    s_id_to_s (TYPE): Description
	    s_id_to_preprocessed_s (TYPE): Description
	    g_id_to_s_ids (TYPE): Description
	    threshold (TYPE): Description
	
	Returns:
	    TYPE: Description
	"""


	search_raw_sentences = sent_tokenize_f(search_string)

	# Generate a list of row tuples that contain the information we want to display. 
	row_list = []
	for search_raw_sentence in search_raw_sentences:
		search_preprocessed_sentence = preprocess_f(search_raw_sentence)
		sentence_id_to_score = _get_sentence_scores_for_query_string(search_preprocessed_sentence, s_id_to_preprocessed_s, model)
		gene_id_to_matches = {gene_id:[(sentence_id,sentence_id_to_score[sentence_id]) for sentence_id in sentence_ids] for gene_id,sentence_ids in g_id_to_s_ids.items()}
		for gene_id,matches in gene_id_to_matches.items():
			for match in matches:
				# What is the relevant information for a single match and score.
				gene_id = gene_id
				sentence_id = match[0]
				sentence_score = match[1]
				sentence_value = s_id_to_s[sentence_id]
				query_sentence = search_raw_sentence
				# Add that row to the growing list of potential rows to include.
				row_values = (gene_id, sentence_id, sentence_score, sentence_value, query_sentence)
				row_list.append(row_values)


	# Create the dataframe from all of those rows and add columns that reflect views of the scores to rank the rows.
	thing = pd.DataFrame(row_list, columns=["gene_id","sentence_id","score","sentence","q"])
	gene_id_to_max_score = dict(thing.groupby("gene_id")["score"].apply(lambda x: np.max(x)))
	gene_id_to_mean_score = dict(thing.groupby("gene_id")["score"].apply(lambda x: np.mean(x)))
	thing["max_score"] = thing["gene_id"].map(gene_id_to_max_score)
	thing["mean_score"] = thing["gene_id"].map(gene_id_to_mean_score)

	# Filtering the rows.
	thing = thing[thing["score"]>=threshold]


	# Sorting the rows.
	thing = thing.sort_values(by=["max_score","mean_score","gene_id","score"], ascending=False)


	# Filtering and column modifying steps that only apply the sorting is done.
	if ids_subset is not None:
		thing = thing[thing["gene_id"].isin(ids_subset)]
	thing = thing[thing["gene_id"].isin(pd.unique(thing["gene_id"].values)[:max_genes])]
	gene_id_to_rank = {gene_id:rank for rank,gene_id in enumerate(pd.unique(thing["gene_id"]),1)}
	thing["rank"] = thing["gene_id"].map(gene_id_to_rank)
	return(thing)










def _get_sentence_scores_for_query_string_sentence_vectors(query_sentence, sentence_id_to_embedding, vectorization_function):

	sentence_id_to_score = {}
	query_embedding = vectorization_function(query_sentence)
	for sentence_id,sentence_embedding in sentence_id_to_embedding.items():
		similarity = 1-cosine(query_embedding,sentence_embedding)
		sentence_id_to_score[sentence_id] = similarity
	return(sentence_id_to_score)



def handle_free_text_query_with_precomputed_sentence_embeddings(search_string, max_genes, sent_tokenize_f, preprocess_f, vectorization_f, s_id_to_s, s_id_to_s_embedding, g_id_to_s_ids, threshold, first=None, ids_subset=None):


	search_raw_sentences = sent_tokenize_f(search_string)

	# Generate a list of row tuples that contain the information we want to display. 
	row_list = []
	for search_raw_sentence in search_raw_sentences:
		search_preprocessed_sentence = preprocess_f(search_raw_sentence)
		sentence_id_to_score = _get_sentence_scores_for_query_string_sentence_vectors(search_preprocessed_sentence, s_id_to_s_embedding, vectorization_f)
		gene_id_to_matches = {gene_id:[(sentence_id,sentence_id_to_score[sentence_id]) for sentence_id in sentence_ids] for gene_id,sentence_ids in g_id_to_s_ids.items()}
		for gene_id,matches in gene_id_to_matches.items():
			for match in matches:
				# What is the relevant information for a single match and score.
				gene_id = gene_id
				sentence_id = match[0]
				sentence_score = match[1]
				sentence_value = s_id_to_s[sentence_id]
				query_sentence = search_raw_sentence
				# Add that row to the growing list of potential rows to include.
				row_values = (gene_id, sentence_id, sentence_score, sentence_value, query_sentence)
				row_list.append(row_values)


	# Create the dataframe from all of those rows and add columns that reflect views of the scores to rank the rows.
	thing = pd.DataFrame(row_list, columns=["gene_id","sentence_id","score","sentence","q"])
	gene_id_to_max_score = dict(thing.groupby("gene_id")["score"].apply(lambda x: np.max(x)))
	gene_id_to_mean_score = dict(thing.groupby("gene_id")["score"].apply(lambda x: np.mean(x)))
	thing["max_score"] = thing["gene_id"].map(gene_id_to_max_score)
	thing["mean_score"] = thing["gene_id"].map(gene_id_to_mean_score)

	# Filtering the rows.
	thing = thing[thing["score"]>=threshold]


	# Sorting the rows.
	if first is not None:
		thing["first"] = thing["gene_id"].map(lambda x: (x==first))
		thing = thing.sort_values(by=["first","max_score","mean_score","gene_id","score"], ascending=False)
	else:
		thing = thing.sort_values(by=["max_score","mean_score","gene_id","score"], ascending=False)


	# Filtering and column modifying steps that only apply the sorting is done.
	if ids_subset is not None:
		thing = thing[thing["gene_id"].isin(ids_subset)]
	thing = thing[thing["gene_id"].isin(pd.unique(thing["gene_id"].values)[:max_genes])]
	gene_id_to_rank = {gene_id:rank for rank,gene_id in enumerate(pd.unique(thing["gene_id"]),1)}
	thing["rank"] = thing["gene_id"].map(gene_id_to_rank)
	return(thing)




































def _keyword_search(id_to_text, raw_keywords, modified_keywords):
	"""Helper function for searching the dataset for keywords and keyphrases.
	
	Args:
	    id_to_text (TYPE): Description
	    raw_keywords (TYPE): Description
	    modified_keywords (TYPE): Description
	
	Returns:
	    TYPE: Description
	"""
	# The raw keywords and modified keywords should be two paired lists where the elements correspond to one another.
	# The modifications done to the keywords should already match the modifications done to the texts in the input dictionary so they can be directly compared.
	assert len(raw_keywords) == len(modified_keywords)
	id_to_found_keywords = {i:[r_kw for r_kw,m_kw in zip(raw_keywords,modified_keywords) if m_kw in text] for i,text in id_to_text.items()}
	id_to_num_found_keywords = {i:len(kw_list) for i,kw_list in id_to_found_keywords.items()}
	return(id_to_found_keywords, id_to_num_found_keywords)







def handle_keyword_query(raw_keywords, modified_keywords, max_genes, phene_per_line, s_id_to_s, s_id_to_kw_s, g_id_to_d, g_id_to_kw_d, g_id_to_s_ids, ids_subset=None):
	"""Summary
	
	Args:
	    raw_keywords (TYPE): Description
	    modified_keywords (TYPE): Description
	    max_genes (TYPE): Description
	    phene_per_line (TYPE): Description
	    s_id_to_s (TYPE): Description
	    s_id_to_kw_s (TYPE): Description
	    g_id_to_d (TYPE): Description
	    g_id_to_kw_d (TYPE): Description
	    g_id_to_s_ids (TYPE): Description
	
	Returns:
	    TYPE: Description
	"""
	# One of the lines should refer to a single sentence from a phenotype description for a gene.
	if phene_per_line:
		sentence_id_to_found_keywords, sentence_id_to_num_found_keywords = _keyword_search(s_id_to_kw_s, raw_keywords, modified_keywords)
		row_list = []
		for gene_id,sentence_ids in g_id_to_s_ids.items():
			for sentence_id in sentence_ids:
				# What is the relevant information that needs to go in its own columns.
				gene_id = gene_id
				sentence_id = sentence_id
				sentence_value = s_id_to_s[sentence_id]
				num_found_keywords = sentence_id_to_num_found_keywords[sentence_id]
				found_keywords_list = sentence_id_to_found_keywords[sentence_id]
				found_keywords_string = ", ".join(found_keywords_list)
				# Add all of this information to the growing list of rows.
				row_values = (gene_id, sentence_id, sentence_value, num_found_keywords, found_keywords_string)
				row_list.append(row_values)
		# Create the dataframe out of all those rows.
		thing = pd.DataFrame(row_list, columns=["gene_id", "sentence_id", "sentence", "num_found", "kw_found"])
		thing = thing[thing["num_found"]>0]



	# One of the lines should refer to a single gene instead of a single sentence.
	else:
		gen_id_to_found_keywords, gene_id_to_num_found_keywords = _keyword_search(g_id_to_kw_d, raw_keywords, modified_keywords)
		row_list = []
		for gene_id,description in g_id_to_d.items():
			# What is the relevant information that needs to go in its own columns.
			gene_id = gene_id
			description = description
			num_found_keywords = gene_id_to_num_found_keywords[gene_id]
			found_keywords_list = gen_id_to_found_keywords[gene_id]
			found_keywords_string = ", ".join(found_keywords_list)
			# Add all of this information to the growing list of rows.
			row_values = (gene_id, description, num_found_keywords, found_keywords_string)
			row_list.append(row_values)
		# Create the dataframe out of all those rows.
		thing = pd.DataFrame(row_list, columns=["gene_id","description","num_found","kw_found"])
		thing = thing[thing["num_found"]>0]



	# At this point, the resulting dataframe looks a little bit different depending on whether rows refer to genes or sentences.
	# This step of totaling up the keywords found for each gene in two different ways is applicable to both dataframes, because these columns are shared.
	gene_id_to_total_found_keywords = dict(thing.groupby("gene_id")["num_found"].apply(lambda x: np.sum(x)))
	gene_id_to_total_unique_found_keywords = dict(thing.groupby("gene_id")["kw_found"].apply(lambda x: len(set(flatten([y.split(",") for y in x])))))
	thing["total_across_sentences"] = thing["gene_id"].map(gene_id_to_total_found_keywords)
	thing["total_unique_across_sentences"] = thing["gene_id"].map(gene_id_to_total_unique_found_keywords)
	thing = thing.sort_values(by=["total_unique_across_sentences", "gene_id", "num_found"], ascending=False)
	


	# Filtering and column modifying steps that only apply the sorting is done, this also only operates on columns that are shared between the two dataframes.
	if ids_subset is not None:
		thing = thing[thing["gene_id"].isin(ids_subset)]
	thing = thing[thing["gene_id"].isin(pd.unique(thing["gene_id"].values)[:max_genes])]
	gene_id_to_rank = {gene_id:rank for rank,gene_id in enumerate(pd.unique(thing["gene_id"]),1)}
	thing["rank"] = thing["gene_id"].map(gene_id_to_rank)
	return(thing)



















def handle_annotation_query(term_ids, max_genes, g_id_to_annots, ontologies, ids_subset=None):
	"""Summary
	
	Args:
	    term_ids (TYPE): Description
	    max_genes (TYPE): Description
	    g_id_to_annots (TYPE): Description
	    ontologies (TYPE): Description
	
	Returns:
	    TYPE: Description
	"""
	row_list = []

	for term_id in term_ids:
		term_id_str = term_id.replace(":","_")
		ontology_name, term_number = tuple(term_id_str.split("_"))
		term_id_str = term_id.replace(":","_")
		inherited_terms = ontologies[ontology_name.upper()].inherited(term_id)
		descendant_terms = ontologies[ontology_name.upper()].descendants(term_id)


		for gene_id,annotations in g_id_to_annots.items():

			for annotated_id in annotations:
				if annotated_id in descendant_terms:

					gene_id = gene_id
					query_term_id = term_id
					query_term_name = ontologies[ontology_name.upper()][term_id].name
					annotated_term_id = annotated_id
					annotated_term_name = ontologies[ontology_name.upper()][annotated_id].name
					direct = (query_term_id==annotated_term_id)


					row_values = (gene_id, query_term_id, query_term_name, annotated_term_id, annotated_term_name, direct)
					row_list.append(row_values)


	results = pd.DataFrame(row_list, columns=["gene_id", "query_term_id", "query_term_name", "annotated_term_id", "annotated_term_name", "direct"])




	gene_id_to_total_matches = dict(results.groupby("gene_id")["query_term_id"].apply(lambda x: len(set(x))))
	results["total_matches"] = results["gene_id"].map(gene_id_to_total_matches)
	results = results.sort_values(by=["total_matches", "gene_id"], ascending=[False,False])



	# Filtering and column modifying steps that only apply the sorting is done, this also only operates on columns that are shared between the two dataframes.
	if ids_subset is not None:
		results = results[results["gene_id"].isin(ids_subset)]
	results = results[results["gene_id"].isin(pd.unique(results["gene_id"].values)[:max_genes])]
	gene_id_to_rank = {gene_id:rank for rank,gene_id in enumerate(pd.unique(results["gene_id"]),1)}
	results["rank"] = results["gene_id"].map(gene_id_to_rank)
	return(results)
















