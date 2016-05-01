INDICES_PATH = "odyssey/outputs/out/filter_seq_1/all_indicies_to_keep_sorted_unique.txt"
DATA_PATH = "data/ss.txt"
OUT_FILE1 = "data/ss_filtered_input.txt"
OUT_FILE2 = "data/ss_filtered_output.txt"

def parse_human(data_file, sorted_idxs):

    X_strings = ''
    Y_strings = ''

    input_idx = -1
    output_idx = -1
    input_idx_into_sorted_idx = 0
    output_idx_into_sorted_idx = 0
    num_idxs = len(sorted_idxs)
    next_input_idx_to_keep = sorted_idxs[input_idx_into_sorted_idx]
    next_output_idx_to_keep = sorted_idxs[output_idx_into_sorted_idx]

    next_line_is_seq = False
    next_line_is_output = False

    with open(data_file, 'r') as f:
		for line in f:
			if 'A:sequence' in line:
			    next_line_is_output = False
			    next_line_is_seq = True
			    X_strings = X_strings+'<'
			    input_idx += 1
			    input_idx_into_sorted_idx += 1
				if input_idx_into_sorted_idx < num_idxs:
					next_input_idx_to_keep = sorted_idxs[input_idx_into_sorted_idx]
			elif 'A:secstr' in line:
			    next_line_is_seq = False
			    next_line_is_output = True
			    Y_strings = Y_strings+'<'
			    output_idx += 1
			    output_idx_into_sorted_idx += 1
					if output_idx_into_sorted_idx < num_idxs:
						next_output_idx_to_keep = sorted_idxs[output_idx_into_sorted_idx]
			elif (':sequence' in line) or ('secstr' in line):
			    next_line_is_seq = False
			    next_line_is_output = False
			elif next_line_is_seq:
				if input_idx == next_input_idx_to_keep:
					
					X_strings = X_strings+line[:-1]

			elif next_line_is_output:
				if output_idx == next_output_idx_to_keep:
					
					Y_strings = Y_strings+line[:-1]

		splitX = X_strings.split('<')[1:]
		splitY = Y_strings.split('<')[1:]

		if len(splitX) != len(splitY):
			print len(splitX), len(splitY)
			assert False

    return splitX, splitY

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