from PIL import Image
import random
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd

def read_training_data(classes, train_data_size):
	x_data = []
	y_data = []
	first = True
	with open("ALL_DATA_TRANSPOSED.csv", "r") as input:
		for line in input:
			for char in line:
				if (char == "~"):
					charnum = 94
				elif (char == "\n"):
					charnum = 95
				else:
					charnum = ord(char)
				x_data.append(charnum-27)
				if (not first):
					y_data.append(charnum-27)
				if (first):
					first = False
	x_data.pop(-1)
	test_x_data = x_data[-train_data_size:]
	x_data = x_data[:-train_data_size]
	test_y_data = y_data[-train_data_size:]
	y_data = y_data[:-train_data_size]
	#return (x_data, y_data, test_x_data, test_y_data)
	with open ("TRAINING_&_TEST_DATA.pickle", "wb") as picklef:
		pickle.dump([x_data, y_data, test_x_data, test_y_data], picklef)


def read_pickle():
	with open ("TRAINING_&_TEST_DATA.pickle", "rb") as picklef:
		data = pickle.load(picklef)
	return data

