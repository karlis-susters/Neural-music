import os

header = """0, 0, Header, 1, 2, 8
1, 0, Start_track
1, 0, SMPTE_offset, 96, 0, 3, 0, 0
1, 0, Time_signature, 4, 2, 24, 8
1, 0, Key_signature, 0, "major"
1, 0, Tempo, 500000
1, 0, End_track
2, 0, Start_track
2, 0, MIDI_port, 0
2, 0, Title_t, "Neural Suite"
2, 0, Program_c, 0, 0
2, 0, Control_c, 0, 7, 100
2, 0, Control_c, 0, 10, 64
"""


files = os.listdir()
for f in files:
	if (f[-6:] != "NN.csv"):
		continue
	infileName = f.split('.')[0]
	print (infileName)
	with open("{}_back_from_NN.csv".format(infileName), "w") as outfile:
		with open(f, "r") as infile:
			outfile.write(header)
			time = 0
			current = []
			newcurrent = []
			for line in infile:
				#outfile.write(str(current) + ": {}\n".format(time))
				for note in current:
					if not (note in line):
						#old note, has to be removed
						#current.remove(note)
						outfile.write("2, {}, Note_on_c, 0, {}, 0\n".format(time, ord(note)))
					else:
						newcurrent.append(note)
				current = newcurrent
				newcurrent = []
				for note in line:
					if note == "\n" or note == "~":
						continue
					if not (note in current):
						#New note
						current.append(note)
						outfile.write("2, {}, Note_on_c, 0, {}, 107\n".format(time, ord(note)))
				time += 1
			outfile.write(
"""2, {}, End_track
0, 0, End_of_file""".format(time+10))					



