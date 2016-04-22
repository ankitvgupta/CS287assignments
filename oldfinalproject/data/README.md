## Data generation
To generate the dataset, there are several steps. These are adapted from the instructions given in http://f1000research.com/articles/2-231/v1

- Get the gene ontology data for humans from http://geneontology.org/page/download-annotations. Rename the downloaded file to human.gz
- Then, run ./get_gene_data.sh to get the rest of the data
- Ultimately, the training data will be in human.pr.filter (protein sequences) and human.go.filter (output annotations).

