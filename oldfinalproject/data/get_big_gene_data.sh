#!/bin/sh
set -x

#step 1,download gene ontology data
wget "ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/gene_association.goa_uniprot.gz" -O multispecies.gz

#step 2,downlaod gene sequence data
wget -O sprot_multispecies.gz "ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"

wget -O trembl_multispecies.gz "ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.fasta.gz"

#step 3,gunzip
gunzip *.gz

cat sprot_multispecies trembl_multispecies > all_multispecies

#step 4, convert gene ontology data to our format
#one line one data, format: gene_db_id \t gene_ontology_id1 gene_ontology_id2 ............
./get_go.py multispecies > multispecies.go

#step 5, convert gene sequence to our format
#one line one data, format: gene_db_id \t gene_sequence
./get_gene.py all_multispecies > multispecies.pr

#step 6, get same gene_id data as parallel corpus 
./get_corpus.py multispecies.go multispecies.pr
