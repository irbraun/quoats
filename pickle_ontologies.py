import sys


sys.path.append("../oats")
from oats.utils.utils import save_to_pickle
from oats.annotation.ontology import Ontology




# Define the paths.
go_obo_path = "ontologies/go.obo"                                                                
po_obo_path = "ontologies/po.obo"                                                             
pato_obo_path = "ontologies/pato.obo"
go_pickle_path = "ontologies/go.pickle"                                                                
po_pickle_path = "ontologies/po.pickle"                                                             
pato_pickle_path = "ontologies/pato.pickle"



# Build the ontology objects and save them to pickles so they can be quickly read from there instead of obo files.
pato = Ontology(pato_obo_path)
po = Ontology(po_obo_path)
go = Ontology(go_obo_path)
save_to_pickle(pato, pato_pickle_path)
save_to_pickle(po, po_pickle_path)
save_to_pickle(go, go_pickle_path)
print("done")

