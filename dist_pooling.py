#In the paper they have used Distributed pooling to estimate feature distribution,
#I will use the same

import math

import torch
import torch.nn as nn

class DistributionPoolingFilter(nn.Module):
	__constants__ = ['num_bins', 'sigma']

	def __init__(self, num_bins=16, sigma=0.0167):
		super(DistributionPoolingFilter, self).__init__()

		self.num_bins = num_bins
		self.sigma = sigma
		self.alfa = 1/math.sqrt(2*math.pi*(sigma**2))
		self.beta = -1/(2*(sigma**2))

		sample_points = torch.linspace(0,1,steps=num_bins, dtype=torch.float32, requires_grad=False)
		self.register_buffer('sample_points', sample_points)

	def extra_repr(self):
		return 'num_bins={}, sigma={}'.format(
			self.num_bins, self.sigma
		)

	def forward(self, data):
		batch_size, num_instances, num_features = data.size()
		sample_points = self.sample_points.repeat(batch_size,num_instances,num_features,1)
		data = torch.reshape(data,(batch_size,num_instances,num_features,1))
		diff = sample_points - data.repeat(1,1,1,self.num_bins)
		diff_2 = diff**2
		result = self.alfa * torch.exp(self.beta*diff_2)
		out_unnormalized = torch.sum(result,dim=1)
		norm_coeff = torch.sum(out_unnormalized, dim=2, keepdim=True)
		out = out_unnormalized / norm_coeff
		
		return out