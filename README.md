## quoats

[quoats.dill-picl.org](https://quoats.dill-picl.org/)

### Description
This repository contains the code for a [streamlit](https://www.streamlit.io/) application for querying datasets of phenotype descriptions using annotations with ontology terms as well as natural language. An instance of the tool is available at [quoats.dill-picl.org](https://quoats.dill-picl.org/) for querying a dataset of plant genes and their associated annotations and free text phenotype descriptions. More information about that dataset including where the data comes from and how it is preprocessed and formatted is available [here](https://github.com/irbraun/plant-data). This application is associated with a publication about applying natural language processing approaches to plant phenotype data that is currently in preparation.


### Running as a Python Script
The script used to run the streamlit application can also be tested as a Python script by passing arguments to the script. The arguments than can be passed to the `main.py` script are as follows.
```
--type -t:     The type of query that should be executed, one of {freetext, keywords, terms}.
--query -q:    The query string, with formatting depending on the type of query selected.
--limit -l:    The maximum number of genes to include in the output file.
--output -o:   The path where the output file should be saved.
--species -s:  Optional. Limits the output to only contain genes from this specific species.
```

### Publication
A paper describing this web application and related analysis is in preparation.


### Feedback
Send any feedback, comments, questions, or suggestions to irbraun at iastate dot edu.
