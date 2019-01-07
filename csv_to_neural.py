import os
files = os.listdir()
for f in files:
	if (f[-7:] != "2nd.csv"):
		continue
	infileName = f.split('.')[0]
	print(infileName)
	right = []
	left = []
	with open("{}.csv".format(infileName), "r") as infile:
		for line in infile:
			listt = line.split(',')
			if (listt[2].lower() == " header"):
				if (listt[4] != " 2" and listt[4] != " 3"):
					print ("Too many tracks!! ("+listt[4]+")")
			if listt[2].lower() == " note_on_c" or listt[2].lower() == " note_off_c":
				if listt[0] == "2":
					#right hand
					right.append(listt)
				else:
					left.append(listt)

	current = []
	rightLen = len(right)
	leftLen = len(left)
	time = 0
	rightPos = 0
	leftPos = 0
	with open("{}_NN.csv".format(infileName), "w") as outfile:
		while (rightPos < rightLen or leftPos < leftLen):
			#while we have lines left
			while (rightPos < rightLen and right[rightPos][1] == " {}".format(time)):
				#while right hand time matches current time
				l = right[rightPos]
				char = chr(int(l[4]))
				if (l[2].lower() == " note_on_c" and l[5] == " 0\n") or l[2].lower() == " note_off_c":
					#note off
					current.remove(char)
				else:
					#note on
					current.append(chr(int(l[4])))

				rightPos += 1
			while (leftPos < leftLen and left[leftPos][1] == " {}".format(time)):
				l = left[leftPos]
				char = chr(int(l[4]))
				if (l[2].lower() == " note_on_c" and l[5] == " 0\n") or l[2].lower() == " note_off_c":
					#note off
					current.remove(char)
				else:
					#note on
					current.append(chr(int(l[4])))

				leftPos += 1
			if (len(current) == 0):
				outfile.write("~") #Empty click
			else:
				outfile.write("".join(sorted(list(set(current)))))
			outfile.write("\n")
			time += 1
		outfile.write ("~\n~\n~\n~\n~\n~\n~\n~\n~\n~\n~\n~\n~\n~\n~\n~\n")
		outfile.write ("~\n~\n~\n~\n~\n~\n~\n~\n~\n~\n~\n~\n~\n~\n~\n~\n")



