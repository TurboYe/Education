#!/user/bin/python
# -* - coding:UTF-8 -*-
from abaqus import *
from abaqusConstants import *
import job
import os

os.chdir(r"E:\Microneedle\pythonProject1\w0.001Input_Files")
count = 7001
while count < 7021:
	inputfilename = "Beam_No%d.inp" %(count)
	jobName = "Beam_No%d" %(count)
	mdb.JobFromInputFile(name=jobName, inputFileName=inputfilename)
	mdb.jobs[jobName].setValues(numCpus=2, numDomains=2)
	mdb.jobs[jobName].submit()

	# mdb.jobs[jobName].waitForCompletion()
	count = count + 1







