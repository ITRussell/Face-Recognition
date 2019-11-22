# Author: Ian Russell

import numpy as np
import pandas as pd


class LDA:
	
	def __init__(self,data, d_disc):
		"""
		Pass in pandas dataframe to instantiate 
		parameters.
		"""
		self.data = pd.DataFrame(data)
		self.mean_vectors = self.data.groupby("Label").mean().T
		self.d_disc = d_disc
		self.size = self.data.to_numpy().shape


	def w_scatter(self):
		"""
		Computes within class scatter matrix.
		"""

		# Intitialize scatter matrix
		within_scatter = np.zeros((self.size[1]-1, self.size[1]-1))
		
		# Outer loop iterates through classes Sw
		for cl, rows in self.data.groupby('Label'):
			
			# Remove label
			rows = rows.drop(['Label'], axis=1)

			# Intialize inner matrix Si, scatter for each class
			s = np.zeros((self.size[1]-1, self.size[1]-1))
			
			# Compute for each feature the scatter matrix
			for index, row in rows.iterrows():
				
				x, mc = row.values.reshape(self.size[1]-1,1), self.mean_vectors[cl].values.reshape(self.size[1]-1,1)
				s += (x - mc).dot((x-mc).T)

				# Sum all inner matrices (Si)
				within_scatter += s

		return within_scatter


	def b_scatter(self):
		"""
		Computes between class scatter matrix.
		"""

		# Mean of each frature
		feature_means = self.data.drop(['Label'],axis=1).mean()

		# Intialize scatter
		between_class_scatter = np.zeros((self.size[1]-1, self.size[1]-1))

		# Compute between class matrix
		for c in self.mean_vectors:

			n = len(self.data.loc[self.data['Label'] == c].index)
			mc, m = self.mean_vectors[c].values.reshape(self.size[1]-1, 1), feature_means.values.reshape(self.size[1]-1,1)
			between_class_scatter += n * (mc-m).dot((mc-m).T)

		return between_class_scatter



	def disc(self):
		"""
		Function to select linear discriminates (n eigen vectors).
		"""

		Sw, Sb = self.w_scatter(), self.b_scatter()
		
		# Eigen's for product of Sw and Sb
		eig_vals, eig_vects = (np.linalg.eig(np.linalg.inv(Sw).dot(Sb)))
		eig_vects = eig_vects.real
	
		# Sort
		og = list(eig_vals)
		sorted_og = sorted(og, reverse=True)

		# Top rank eigen values
		top_d = [val for val in sorted_og[:self.d_disc]]

		# Get index locations of eigen vectors
		top_vects_locations = [og.index(value) for value in top_d]
		

		# Create a list of top ranked eigen vectors (descending) and shape appropriately for reduction  
		W = [np.array(eig_vects[top_vects_locations[i]]).reshape(self.size[1]-1,1) for i in range(len(top_vects_locations))]

		# Stack eigen vectors to form projection matrix
		matrix_w = np.hstack(W)

		B = np.transpose(matrix_w).dot(np.transpose(self.data.drop(["Label"],axis=1).to_numpy()))

		return B