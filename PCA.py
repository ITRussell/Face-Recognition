# Author: Ian Russell

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import scipy.ndimage

class PCA:

	def __init__(self, A, d_reduction):
		"""
		Pass in numpy array to instantiate PCA parameters
		"""
		self.A = A
		self.d_reduction = d_reduction


	def decompose(self):
		"""
		Function to perform singular value decompostion.
		Returns eigen values and vectors.
		"""
		
		# Center Data
		mean = np.mean(self.A, axis=0)
		centered = self.A-mean 
		
		# Compute covariance and corresponding eigen decomp.
		
		U, S, V = np.linalg.svd(centered)

		return U, S, V

	def reduce(self):
		"""
		Function to perform dimensionality reduction.
		Returns projection array for corresponding
		dimensionality reduction.
		"""
		res = self.decompose()
		U = res[0]
		S = res[1]
		V = res[2]
		# Decompose data matrix
	    
		components = V[:self.d_reduction]
		projected = U[:,:self.d_reduction]*S[:self.d_reduction]

		return projected, components

		

	def eigface(self, faces):

		v = self.reduce()[1]

		
		for i in range(faces):
		    plt.imshow(scipy.ndimage.rotate((v[i].reshape(30,30)), -90), interpolation='nearest', cmap='gray')
		    plt.show()