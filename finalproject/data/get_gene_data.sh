#!/bin/sh
set -x

#step 1,download gene ontology data
#wget "http://cvsweb.geneontology.org/cgi-bin/cvsweb.cgi/go/gene-associations/gene_association.goa_human.gz?rev=HEAD" -O human.gz

#step 2,downlaod gene sequence data
wget -O sprot.gz "ftp://ftp.expasy.org/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"

#step 3,gunzip
gunzip *.gz

#step 4, convert gene ontology data to our format
#one line one data, format: gene_db_id \t gene_ontology_id1 gene_ontology_id2 ............
./get_go.py human> human.go

#step 5, convert gene sequence to our format
#one line one data, format: gene_db_id \t gene_sequence
./get_gene.py sprot > human.pr

#step 6, get same gene_id data as parallel corpus 
./get_corpus.py human.go human.pr
