import math
import numpy as np

class KDE(object):
	def __init__(self, num_bins=21, sigma=0.05):
		super(KDE, self).__init__()
		
		self.num_bins = num_bins
		self.sigma = sigma
		self.alfa = 1/math.sqrt(2*math.pi*(sigma**2))
		self.beta = -1/(2*(sigma**2))

		self.sample_points = np.linspace(0,1,num=num_bins)

	def calculate(self, data):
		batch_size, num_instances, num_features = data.shape

		sample_points = np.tile(self.sample_points,(batch_size,num_instances,num_features,1))
		data = np.reshape(data,(batch_size,num_instances,num_features,1))
		diff = sample_points - np.tile(data,(1,1,1,self.num_bins))
		diff_2 = diff**2
		result = self.alfa * np.exp(self.beta*diff_2)
		out_unnormalized = np.sum(result,axis=1)
		norm_coeff = np.sum(out_unnormalized, axis=2, keepdims=True)
		out = out_unnormalized / norm_coeff
  
		return np.asarray(out, dtype=np.float32)