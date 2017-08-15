from mvnc import mvncapi as mvnc
import sys
import numpy
import cv2
import time
import csv
import os
import sys


def execute_graph(blob,img):
	mvnc.SetGlobalOption(mvnc.GlobalOption.LOGLEVEL, 2)
	devices = mvnc.EnumerateDevices()
	if len(devices) == 0:
		print('No devices found')
		quit()
	device = mvnc.Device(devices[0])
	device.OpenDevice()
	opt = device.GetDeviceOption(mvnc.DeviceOption.OPTIMISATIONLIST)
	with open(blob, mode='rb') as f:
		blob = f.read()
	graph = device.AllocateGraph(blob)
	graph.SetGraphOption(mvnc.GraphOption.ITERATIONS, 1)
	iterations = graph.GetGraphOption(mvnc.GraphOption.ITERATIONS)
	graph.LoadTensor(img.astype(numpy.float16), 'user object')
	output, userobj = graph.GetResult()
	graph.DeallocateGraph()
	device.CloseDevice()
	return output,userobj


# to do
params = sys.argv[1:]


# open the network blob file
# implies we must compile net first to graph with sdk

blob = "../network/graph"

input_data = None #todo

output,userobj=execute_graph(blob,input_data))
print('\n------- predictions --------')

# display output
