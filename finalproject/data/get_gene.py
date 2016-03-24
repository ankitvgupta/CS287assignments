#!/usr/bin/env python
#coding=utf-8
import os
import sys
import time
reload(sys)
sys.setdefaultencoding('utf-8')
#convert fasta format to this format:
#one line one data, format: gene_db_id + "\t" + seqeunce


#process one record of fasta
def process_content(content):
    item_list = content.split("\n")
    gene_name = item_list[0].split("|")[1]
    sequence = "".join( item_list[1:] )
    print gene_name + "\t" + sequence.strip()

if __name__ == "__main__":
    if len(sys.argv) < 2 :
        print "please input filename"
        sys.exit(0)

    filename = sys.argv[1]
    infile = open(filename,'r')
    outfilename = filename + ".div"

    content = ""
  
    while True:
        line = infile.readline()
        if line:
            if line.find(">")==0 and content!="":
                process_content(content)
                content = line
            else:
                content = content + line
        else:
            break
    
    #last
    process_content(content)
