# Usage: python test_increasing.py < train_tags.txt
import sys 
prev_num = -1
for line in sys.stdin:
	first = line.split('\t')[0]
	if first.isdigit():
		num = int(first)
		assert(num > prev_num)
		prev_num = num
