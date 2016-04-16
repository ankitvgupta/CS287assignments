beam_viterbi_accuracy = '''1	0.20408163265306	
2	0.22448979591837	
3	0.30612244897959	
4	0.44897959183673	
5	0.63265306122449	
6	0.63265306122449	
7	0.69387755102041	
8	0.79591836734694	
9	0.89795918367347	
10	0.89795918367347	
11	0.89795918367347	
12	1	
13	1	
14	1'''

beam_viterbi_efficiency = '''5	0.018401861190796	0.25655102729797	
10	0.060006856918335	0.28201103210449	
15	0.10544180870056	0.30379486083984	
20	0.16871094703674	0.35351705551147	
25	0.30562591552734	0.35228800773621	
30	0.42222881317139	0.41639184951782	
35	0.54674816131592	0.44654989242554	
40	0.72248911857605	0.51300692558289	
45	0.89531683921814	0.48929882049561	
50	1.0999820232391	0.56344389915466	
55	1.2913339138031	0.64389586448669	
60	1.6783909797668	0.84966778755188	
65	2.0830111503601	0.69084310531616	
70	2.1428220272064	0.69285583496094	
75	2.3829228878021	0.72679710388184	
80	2.7803220748901	0.76758694648743	
85	3.0081448554993	0.94790005683899	
90	3.5679311752319	0.91430282592773	
95	4.0664789676666	0.89973187446594	
100	5.4597499370575	1.2646780014038'''

from matplotlib import pyplot as plt
import seaborn as sns

sns.set_context("paper")

# accuracy plot
# X = []
# Y = []
# lines = beam_viterbi_accuracy.split('\n')
# for line in lines:
# 	x, y = line.rstrip().split('	')
# 	X.append(int(x))
# 	Y.append(float(y))

# plt.plot(X, Y)
# plt.xlabel("Number of Beams")
# plt.ylabel("Accuracy")
# plt.title("Beam Search Accuracy Increases with Number of Beams (n=100, C=30)")
# plt.savefig('beam_accuracy.png')

# efficiency plot
X = []
Y1 = []
Y2 = []
lines = beam_viterbi_efficiency.split('\n')
for line in lines:
	x, y1, y2 = line.rstrip().split('	')
	X.append(int(x))
	Y1.append(float(y1))
	Y2.append(float(y2))

plt.plot(X, Y1, label='Viterbi')
plt.plot(X, Y2, label='Beam Search')
plt.legend(loc=4)
plt.xlabel("Number of Classes")
plt.ylabel("Runtime (s)")
plt.title("Beam Search is More Efficient than Viterbi with Many Classes (n=100, k=10)")
plt.savefig('beam_viterbi_efficiency.png')

