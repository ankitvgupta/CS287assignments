INDICES_PATH = "odyssey/outputs/out/filter_seq_1/all_indicies_to_keep_sorted_unique.txt"
DATA_PATH = "data/ss.txt"
OUT_FILE1 = "data/ss_filtered_input.txt"
OUT_FILE2 = "data/ss_filtered_output.txt"

def parse_human(data_file, sorted_idxs):

    X_strings = []
    Y_strings = []

    idx = 0
    prev_idx_to_keep = -1
    next_idx_to_keep = sorted_idxs.pop(0)

    next_line_is_seq = False
    next_line_is_output = False

    with open(data_file, 'r') as f:
        for line in f:
			if 'A:sequence' in line:
			    next_line_is_output = False
			    next_line_is_seq = True
			elif 'A:secstr' in line:
			    next_line_is_seq = False
			    next_line_is_output = True
			elif (':sequence' in line) or ('secstr' in line):
			    next_line_is_seq = False
			    next_line_is_output = False
			elif next_line_is_seq:
				if idx == next_idx_to_keep:
					prev_idx_to_keep = next_idx_to_keep
					try:
						next_idx_to_keep = sorted_idxs.pop(0)
					except IndexError:
						break
					assert prev_idx_to_keep < next_idx_to_keep
					X_strings.append(line[:-1])
				else:
					next_line_is_output = False
				idx += 1
			elif next_line_is_output:
			    Y_strings.append(line[:-1])

	if len(X_strings) != len(Y_strings):
		print len(X_strings), len(Y_strings)

    return X_strings, Y_strings

def load_sorted_indices(data_file=INDICES_PATH):
	sorted_indices = []
	prev_idx_to_keep = -1
	with open(data_file, 'r') as f:
		for line in f:
			this_idx = int(line)
			assert prev_idx_to_keep < this_idx
			prev_idx_to_keep = this_idx
			sorted_indices.append(this_idx)
	return sorted_indices

def save_strings(strings, outfile):
	with open(outfile, 'w') as f:
		for string in strings:
			f.write(string+'\n')

print "Loading indices..."
sorted_indices = load_sorted_indices()
print "Done."
print "Loading strings..."
X_strings, Y_strings = parse_human(DATA_PATH, sorted_indices)
print "Done."
print "Writing out..."
save_strings(X_strings, OUT_FILE1)
save_strings(Y_strings, OUT_FILE2)
print "Done."
print "Saved to", OUT_FILE1, "and", OUT_FILE2