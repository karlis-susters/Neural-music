import time
from os import system
neuron_c = [64, 128, 256, 512]
time_steps = [64, 128, 256]
for n in neuron_c:
	for t in time_steps:
		if ((n != 512 or t != 256) and (n != 256 or t != 256)):
			continue
		with open("params.txt", "w") as f:
			f.write(str(n))
			f.write("\n")
			f.write(str(t))
		system("python BachReplicator_ZPD_.py")
		time.sleep(450)
