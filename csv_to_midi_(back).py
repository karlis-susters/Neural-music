import os
files = os.listdir()
for f in files:
    if (f[-8:] != "m_NN.csv"):
        continue
    infile = f.split('.')[0]
    print(infile)
    os.system("csvmidi \"{}.csv\" \"{}.midi\"".format(infile, infile))
