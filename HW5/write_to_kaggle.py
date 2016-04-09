import numpy as np
import h5py
import argparse
import sys

SEP = '\t'

args = {}

# inverted from preprocessing
def load_tag_dict(file_path, front_tag="<t>", back_tag="</t>"):
    tag_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            tag, idx = line.strip().split(' ')
            tag_dict[int(idx)] = tag[2:]

    tag_dict[max(tag_dict.keys())+1] = front_tag
    tag_dict[max(tag_dict.keys())+1] = back_tag

    return tag_dict

def load_global_to_local(file_path, start_tag=2, stop_tag=3):
	g_to_l = {0: 0, 1: 1}
	with open(file_path, 'r') as f:
		g = 2
		last_l = None
		for line in f:
			if line[0].isdigit():
				_, l, _, _ = line.split(SEP)
				last_l = int(l)
				g_to_l[g] = int(l)
				g += 1
			else:
				g_to_l[g] = last_l + 1
				g_to_l[g+1] = last_l + 2
				g += 2
				last_l = None

	return g_to_l

# for debugging
def print_nth_sentence(file_path, n):
	with open(file_path, 'r') as f:
		c = 0
		for line in f:
			if c+1 == n:
				print line,
			if not line[0].isdigit():
				c += 1


def write_to_txt(test_outputs, tag_dict, gtl_dict, outfile, back_tag="</t>"):
	with open(outfile, 'w') as f:
		f.write("ID,Labels\n")
		for i in range(len(test_outputs)):
			this_output = str(i+1)+','
			for j in range(len(test_outputs[i])):
				c = test_outputs[i][j][0]
				if c == 0:
					break
				else:
					cname = tag_dict[c]
					if cname == back_tag:
						break
					else:
						this_output += cname
				for v in test_outputs[i][j][1:]:
					if v == 0:
						break
					else:
						this_output += '-'+str(gtl_dict[v])
				this_output += ' '
			this_output = this_output.rstrip() + '\n'
			f.write(this_output)
	print "Finished writing to", outfile


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('testfile', help="Test hdf5 file",
                        type=str)
    parser.add_argument('kagglefile', help="Kaggle txt file name",
    					type=str, default="test.txt", nargs='?')
    args = parser.parse_args(arguments)
    testfile = args.testfile
    outfile = args.kagglefile


    # Read in from HDF5
    print "Reading from hdf5"
    with h5py.File(testfile, "r") as f:

    	test_outputs = f['test_outputs']
    	tag_dict = load_tag_dict("data/tags.txt")
    	#gtl_dict = load_global_to_local("data/test.num.txt")
    	gtl_dict = load_global_to_local("data/dev.num.txt")
    	write_to_txt(test_outputs, tag_dict, gtl_dict, outfile)

if __name__ == '__main__':
    # sys.exit(main(sys.argv[1:]))
    print_nth_sentence("data/dev.num.txt", 111)