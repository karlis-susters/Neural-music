import os
files = os.listdir()
for i in range(-5, 6):
	for f in files:
		if (f[-6:] != "NN.csv"):
			continue;
		infileName = f.split('.')[0]
		print(infileName)
		with open("{}.csv".format(infileName), "r") as infile:
			with open("training_transposed.csv", "a") as outfile:
				for line in infile:
					for char in line:
						if (char == "\n" or char == "~"):
							outfile.write(char)
						else:
							outfile.write(chr(ord(char)+i))
					
			