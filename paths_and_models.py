


# The path to the csv file that has the complete dataset we want to be able to search.
DATASET_PATH = "data/genes_texts_annots.csv"



# Names and paths specific to the ontologies used.
ONTOLOGY_NAMES = ["PATO","PO","GO"]
ONTOLOGY_OBO_PATHS = ["ontologies/pato.obo", "ontologies/po.obo", "ontologies/go.obo"]
ONTOLOGY_PICKLE_PATHS = ["ontologies/pato.pickle", "ontologies/po.pickle", "ontologies/go.pickle"]




# These mappings are necessary if the internal strings used for species and how they should be displayed are different.
SPECIES_STRINGS_IN_DATA = ["ath", "zma", "sly", "gmx", "osa", "mtr"]
SPECIES_STRINGS_TO_DISPLAY = ["Arabidopsis", "Maize", "Tomato", "Soybean", "Rice", "Medicago"]
TO_SPECIES_DISPLAY_NAME = {i:d for i,d in zip(SPECIES_STRINGS_IN_DATA, SPECIES_STRINGS_TO_DISPLAY)}



# Paths relevent to the saved machine learning models or classes.
WORD2VEC_MODEL_PATH = "models/word2vec_model_trained_on_plant_phenotypes.model"
WORD_EMBEDDINGS_MODEL_PATH = "models/word2vec_model_trained_on_plant_phenotypes.model"
WORD_EMBEDDINGS_PICKLE_PATH = "stored/stored_token_similarities.pickle"
from gensim.parsing.preprocessing import preprocess_string
WORD_EMBEDDINGS_PREPROCESSING_FUNCTION = lambda x: " ".join(preprocess_string(x))


















def get_methods(**kwargs):
	"""
	This function should be changed based on what approaches are used with the particular dataset.
	This function accepts any keyword arguments, they just have to match what is used in the call
	from the main script. No matter what is used to construct the dictionary of approaches the 
	return dictionary should have unique keys for each approach, and then the values are a name
	that is how the approach should be specified in the drop-down menu, a mapping from sentence IDs
	to vector embeddings, a function to take a text string and produce a format that is ready to be
	vectorized, a function that gets that preprocessed form and does the vectorization, and a
	threshold value to associate with approach, that specifies minimum relevant cosine similarity
	between vector embeddings for the given method.
	
	
	Args:
	    **kwargs: Description
	
	Returns:
	    TYPE: Description
	"""

	# The keyword arguments used in this case are 
	# gene_id_to_preprocessed_description
	# sentence_id_to_preprocessed_sentence
	# word_embedding_object


	# Imports that are needed.
	import gensim
	from gensim.parsing.preprocessing import preprocess_string
	from sklearn.feature_extraction.text import TfidfVectorizer


	# Paths that are needed.
	DOC2VEC_MODEL_PATH = "models/doc2vec_model_trained_on_plant_phenotypes.model"


	# Load the document embeddings from the mean word embeddings.
	sentence_id_to_mean_word2vec_embedding = {i:kwargs["word_embedding_object"].get_mean_embedding(text.split()) for i,text in kwargs["sentence_id_to_preprocessed_sentence"].items()}
	# Load the document embeddings inferred using the Doc2Vec model.
	doc2vec_model = gensim.models.Doc2Vec.load(DOC2VEC_MODEL_PATH)
	sentence_id_to_doc2vec_embedding = {i:doc2vec_model.infer_vector(text.lower().split()) for i,text in kwargs["sentence_id_to_preprocessed_sentence"].items()}
	# Load the embeddings inferred using tf-idf.
	tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)
	tfidf_vectorizer.fit(kwargs["gene_id_to_preprocessed_description"].values())
	sentence_id_to_tfidf_embedding = {i:tfidf_vectorizer.transform([text]).toarray()[0] for i,text in kwargs["sentence_id_to_preprocessed_sentence"].items()}


	# Put all those approaches in a dictionary with the necessary fields.
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
			"vectorization_function":lambda x: kwargs["word_embedding_object"].get_mean_embedding(x),
			"preprocessing_function":lambda x: preprocess_string(x),
			"threshold":0.6,
		}
	}
	return(approaches)











