import sys
for line in sys.stdin:
	line = line.strip('\n')
	vals = line.split(" ")
	#print vals[-1]
	if vals[-1] != "50":
		continue
	print line