import sys
import os
import glob



sys.path.append("../oats")
from oats.utils.utils import save_to_pickle
from oats.annotation.ontology import Ontology



ONTOLOGY_DIR = "ontologies"
for path in glob.glob(os.path.join(ONTOLOGY_DIR,"*.obo")):
	ontology_name = os.path.splitext(os.path.basename(path))[0]
	new_path = os.path.join(ONTOLOGY_DIR,"{}.pickle".format(ontology_name))
	ontology_object = Ontology(path)
	save_to_pickle(ontology_object, new_path) 
print("finished loading and saving ontology objects to pickles")