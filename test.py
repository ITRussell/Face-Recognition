# Author: Ian Russell


################################################
################ Dependencies ##################

import os
import scipy.io
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import time

# Custom imports, make sure in working directory
from PCA import PCA
from LDA import LDA


################################################
################## Settings ####################

# Data, make sure in working directory
file = "PIE.mat"

# Dimensions
d1,d2 = 100, 100

# KNN Params.
n = 3 # -> Value is parameter for KNN
trainNums = [5,10,15]

# Eigen_Faces
Eigen_Faces = True # Change to True to see plot output


#########################################################
################## Utility Functions ####################


def load_data(file):

	data = scipy.io.loadmat(file)
	data, labels = data["Data"], data["Label"]
	return data, labels

def score(finaldf):
	Guesses = list(finaldf["Predicted"])
	Truths = list(finaldf["Actual"])
	
	i = 0
	Answers = []
	for element in Guesses:
		if element == Truths[i]:
			Answers.append("Correct")
		i += 1

	hit_rate = len(Answers)/len(Truths)

	return hit_rate

def pca(data, labels, d1, n, trainNum):
	
	pca = PCA(data, d1)
	reduced = pca.reduce()

	# Visualize with first five eigen_vectors
	#model.eigface(1) 

	# Get transformed data
	results = reduced[0]
	values = reduced[0]
	results = pd.DataFrame(results)
	results["Label"] = labels


	X_train, X_test, y_train, y_test = train_test_split(values, results["Label"], test_size=trainNum, random_state=42)

	neigbors = KNeighborsClassifier(n_neighbors=n)
	neigbors.fit(X_train, y_train)

	predictions = neigbors.predict(X_test)

	final = pd.DataFrame()
	final["Predicted"] = predictions
	actual = pd.Series(y_test).reset_index().drop(columns="index")
	final["Actual"] = actual

	return final




def lda(data, labels, d1, d2, n, trainNum):
	
	# Intialize PCA
	model = PCA(data, d1)
	reduced = model.reduce()	 

	# Get transformed data
	transformed = reduced[0]
	transformed = pd.DataFrame(transformed)
	transformed["Label"] = labels

	# Perform LDA
	lda = LDA(transformed, d2)

	# Create DataFrame
	values = pd.DataFrame(lda.disc().T)
	results = values
	results["Label"] = labels

	# Classifier

	X_train, X_test, y_train, y_test = train_test_split(values, results["Label"], test_size=trainNum, random_state=42)

	neigbors = KNeighborsClassifier(n_neighbors=n)
	neigbors.fit(X_train, y_train)

	predictions = neigbors.predict(X_test)

	final = pd.DataFrame()
	final["Predicted"] = predictions
	actual = pd.Series(y_test).reset_index().drop(columns="index")
	final["Actual"] = actual

	return final



def eigen_faces(data, d1, numFaces):

	model = PCA(data, d1)
	model.eigface(numFaces)

##############################################
################## OUTPUT ####################


data, labels = load_data("PIE.mat")


index = [trainNum for trainNum in trainNums]

pcaScores = []
pcaTimes = []

ldaScores =[]
ldaTimes =[]

print()
print("Calculating PCA...")
print()



for trainNum in trainNums:
	start_time = time.time()

	# Run classifier
	results = pca(data,labels, d1, n, 1-trainNum/21)
	pcaScores.append(str(round(score(results)*100, 1)) + "%")
	pcaTimes.append("%ss" % (round(time.time() - start_time, 2)))

print("Calculating LDA...")

for trainNum in trainNums:
	start_time = time.time()

	# Run classifier
	results = lda(data, labels, d1, d1, n, 1-trainNum/21)
	ldaScores.append(str(round(score(results)*100, 1)) + "%")
	ldaTimes.append("%ss" % (round(time.time() - start_time, 2)))
	


tabular = pd.DataFrame(columns=["trainNum", "PCA_Score","LDA_Score", "PCA_Time", "LDA_Time"])
tabular["trainNum"] = index
tabular["PCA_Score"] = pcaScores
tabular["LDA_Score"] = ldaScores
tabular["PCA_Time"] = pcaTimes
tabular["LDA_Time"] = ldaTimes
print()
print(tabular)
print()

if Eigen_Faces:
	eigen_faces(data, d1, 5)