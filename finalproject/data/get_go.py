#!/usr/bin/env python
#coding=utf-8
import sys
#convert original go annotation format to this:
#one line one data, format: gene_db_id + "\t" + "go term1" + "space" + "go term2"  ....

if __name__=="__main__":
    if len(sys.argv) != 2:
        print "please input filename"
        sys.exit()

    in_file = open(sys.argv[1], "r")

    gene_dict = {}
    for line in in_file:
        if line.find("!")==0:
            continue
        line = line.strip()
        item_list = line.split("\t")
        gene_name = item_list[1]   #gene name, here we use its database id
        gene_word = item_list[4].split(":")[1]  # go termn id
		
        if not gene_dict.has_key(gene_name): #set first id
            gene_dict.setdefault(gene_name,[])
            gene_dict[gene_name].append(gene_word)
        else:
            if gene_word not in gene_dict[gene_name]:#only leave one same gene db id
                gene_dict[gene_name].append(gene_word)

    #print 
    for key in gene_dict:
    	print key+ "\t" + " ".join(gene_dict[key])
