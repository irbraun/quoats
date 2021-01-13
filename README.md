## quoats

[quoats.dill-picl.org](https://quoats.dill-picl.org/)

### Description
This repository contains the code for a [streamlit](https://www.streamlit.io/) application for querying datasets of phenotype descriptions using annotations with ontology terms as well as natural language. An instance of the tool is available at [here](https://quoats.dill-picl.org/) for querying a dataset of plant genes and their associated annotations and free text phenotype descriptions. More information about that dataset including where the data comes from and how it is preprocessed and formatted is available [here](https://github.com/irbraun/plant-data). This application is associated with a publication about applying natural language processing approaches to plant phenotype data that is currently in preparation.


### Running as a Python Script
The script used to run the streamlit application can also be tested as a Python script by passing arguments to the script. The arguments that can be passed to the `main.py` script are as follows.
```
--type -t:     The type of query to perform, one of {freetext, keywords, terms, identifiers}.
--query -q:    The query string, with formatting depending on the type of query selected.
--limit -l:    The maximum number of genes to include in the output file.
--output -o:   The path where the output file should be saved.
--species -s:  Optional. Limits the output to only contain genes from this specific species.
```
Examples of valid queries using the script this way are listed here.
```
python main.py -t freetext -q 'small leaves. abnormal roots.' -l 50 -o myfolder/myresults.tsv
python main.py -t keywords -q 'leaves, roots, auxin' -l 75 -o myfolder/myresults.tsv
python main.py -t terms -q 'PO:0000004' -l 200 -o myfolder/myresults.tsv -s Arabidopsis
python main.py -t identifiers -q arabidopsis:atg7 -l 100 -o myfolder/myresults.tsv
```



### Publication
A paper describing this web application and related analysis is in preparation.


### Feedback
Send any feedback, comments, questions, or suggestions to irbraun at iastate dot edu.
