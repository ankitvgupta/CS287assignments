#!/usr/bin/env python
#coding=utf-8
import sys

if __name__=="__main__":
    if len(sys.argv) != 3:
        print "please input go and dna filename"
        sys.exit()

    go_file = open(sys.argv[1], "r")
    gene_file = open(sys.argv[2], "r")

    go_file_out = open(sys.argv[1] + ".filter","w")
    gene_file_out = open(sys.argv[2] + ".filter","w")

    gene_dict = {}
    for line in go_file:
        line = line.strip()
        item_list = line.split("\t")
        gene_name = item_list[0]
        gene_seq = item_list[1]
        gene_dict[gene_name] = gene_seq

    gene_list_dict = {}

    for line in gene_file:
        line = line.strip()
        item_list = line.split("\t")
        if len(item_list)==1:
            continue
        gene_name = item_list[0]
        gene_seq = item_list[1]

        if gene_dict.has_key(gene_name):
            if not gene_list_dict.has_key(gene_name):
                go_file_out.write(gene_dict[gene_name] + "\n")
                gene_file_out.write(gene_seq + "\n")
                gene_list_dict[gene_name]=1

    """
    i = 0
    for item in gene_list:
        if gene_dict.has_key(item):
            i = i + 1

    print "go number " + str(len(gene_dict))
    print "sequene number " + str(len(gene_list))
    print "go and sequence number " + str(i)
    """
