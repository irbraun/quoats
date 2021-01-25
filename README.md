## quoats

[quoats.dill-picl.org](https://quoats.dill-picl.org/)

### Description
This repository contains the code for a [streamlit](https://www.streamlit.io/) application for querying datasets of phenotype descriptions using annotations with ontology terms as well as natural language. An instance of the tool is available at [here](https://quoats.dill-picl.org/) for querying a dataset of plant genes and their associated annotations and free text phenotype descriptions. More information about that dataset including where the data comes from and how it is preprocessed and formatted is available [here](https://github.com/irbraun/plant-data). This application is associated with a publication about applying natural language processing approaches to plant phenotype data that is currently in preparation.


### Running as a Python Script
The script used to run the streamlit application can also be tested as a Python script by passing arguments to the script. The arguments that can be passed to the `main.py` script are as follows.
```
--type -t:      The type of query to perform, one of {freetext, keywords, terms, identifiers}.
--query -q:     The query string, with formatting depending on the type of query selected.
--limit -l:     The maximum number of genes to include in the output file.
--output -o:    The path where the output file should be saved.
--species -s:   Optional. Limits the output to only contain genes from this specific species.
--threshold -r: Optional. Cosine similarity threshold for matches to include in output.
--algorithm -a: Optional. Currently one of {tfdif, word2vec, doc2vec}.
```
Examples of valid queries using the script this way are listed here. Note that the formatting of the query depends on the type of search being performed. Free text queries are phenotype-related words, phrases, or sentences separated by periods. Keyword queries are words or phrases separated by commas. Ontology term queries are term identifiers separated by commas or whitespace. Gene identifier queries consist of a species string and a gene identifier separated by a colon.
```
python main.py -t freetext -q 'small leaves. abnormal roots.' -l 50 -o myfolder/myresults.tsv
python main.py -t keywords -q 'leaves, roots, auxin' -l 75 -o myfolder/myresults.tsv
python main.py -t terms -q PO:0000004 -l 200 -s arabidopsis -o myfolder/myresults.tsv 
python main.py -t identifiers -q arabidopsis:atg7 -l 100 -o myfolder/myresults.tsv
```


### Reusing with new or additional data
This application can be repurposed for other datasets and ontologies, or reused with different or additional data. The relevant code that would need to be changed for different underyling datasets is contained in `paths_and_models.py`. This file contains paths that point to the data and ontologies used, and specify their names in the context of the web application. This file also defines with methods and models are used to compare text descriptions, including the text preprocessing methods associated with each, and their names in the context of the web application. 



### Publication
A paper describing this web application and related analysis is in preparation.


### Feedback
Send any feedback, comments, questions, or suggestions to irbraun at iastate dot edu.
