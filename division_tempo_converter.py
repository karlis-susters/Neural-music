import os
import sys
files = os.listdir()

def get_coef(currTime, coefs):
	for t in range(len(coefs)):
		if coefs[t][0] > currTime:
			return coefs[t-1]
	if len(tempos) == 0:
		return [0,-1]
	return coefs[-1]

for f in files:
	if (f.split('.')[-1][:3].lower() != "mid"):
		continue
	infile = f.split('.')[0]
	print(infile)
	os.system("midicsv \"{}.mid\" \"{}.csv\"".format(infile, infile))
	outfile = open("{}_1st.csv".format(infile), "w")
	outfile2 = open("{}_2nd.csv".format(infile), "w")
	divide = 48
	outfileName = infile
	tempos = [] #Stores [raw_time, coefficient]
	tempoStarts = [0] #Stores [accounted_time]
	#CONVERT CLICKS PER QUARTER NOTE
	with open("{}.csv".format(infile), "r") as file:
		for line in file:
			list = line.split(',')
			if (list[2] == " Header"):
				divide = int(list[5]) / 8
				if int(list[5]) % 8 != 0:
					sys.exit()
				print("Divide:" + str(divide))
				list[5] = " 8\n"
			currTime = int(list[1]) / divide
			list[1] = " " + str(round(currTime))
			outfile.write(",".join(list))
	outfile.close()
		#GET ALL TEMPOS
	with open("{}_1st.csv".format(infile), "r") as inoutfile:
		for line in inoutfile:
			list = line.split(',')

			if (list[2] == " Tempo"):
				tempo = int(list[3])
				currTime = int(list[1])
				c = get_coef(currTime, tempos)
				currTime = (currTime - get_coef(currTime, tempos)[0]) * get_coef(currTime, tempos)[1] + tempoStarts[-1]

				if (tempo < 310000):
					#coefficient = 0.5
					tempos.append([int(list[1]), 0.5])
					tempoStarts.append(currTime)
				elif (tempo < 625000):
					#coefficient = 1
					tempos.append([int(list[1]), 1])
					tempoStarts.append(currTime)
				elif (tempo < 875000):
					#coefficient = 1.5
					tempos.append([int(list[1]), 1.5])
					tempoStarts.append(currTime)
				else:
					#coefficient = 2
					tempos.append([int(list[1]), 2])
					tempoStarts.append(currTime)
				list[3] = " 500000\n"
	
	
	tempoStarts.pop(0)
	with open("{}_1st.csv".format(infile), "r") as inoutfile:
		for line in inoutfile:
			list = line.split(',')
			currTime = int(list[1])
			coef = get_coef(currTime, tempos)
			currTime = (currTime - coef[0])*coef[1] + tempoStarts[tempos.index(coef)]
			list[1] = " " + str(round(currTime))
			if (list[2] == " Tempo"):
				list[3] = " 500000\n"
			outfile2.write(",".join(list))
	outfile2.close()
	os.system("csvmidi \"{}_2nd.csv\" \"conv_{}_2nd.midi\"".format(infile, outfileName))

