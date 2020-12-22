import sys


sys.path.append("../oats")
from oats.utils.utils import save_to_pickle
from oats.annotation.ontology import Ontology




# Paths to obo ontology files are used to create the ontology objects used.
# If pickling these objects works, that might be bette because building the larger ontology objects takes a long time.
go_obo_path = "resources/go.obo"                                                                
po_obo_path = "resources/po.obo"                                                             
pato_obo_path = "resources/pato.obo"
go_pickle_path = "resources/go.pickle"                                                                
po_pickle_path = "resources/po.pickle"                                                             
pato_pickle_path = "resources/pato.pickle"






# Build the ontology objects and save them to pickles so they can be quickly read from there instead of obo files.
pato = Ontology(pato_obo_path)
po = Ontology(po_obo_path)
go = Ontology(go_obo_path)
save_to_pickle(pato, pato_pickle_path)
save_to_pickle(po, po_pickle_path)
save_to_pickle(go, go_pickle_path)
print("done")

