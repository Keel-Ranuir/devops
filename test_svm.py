import numpy as np
import torch
from svm import SVM, rbf

def test_linear():
	x_1 = torch.Tensor(np.ones(5)).type(torch.float32)
	x_2 = torch.Tensor(np.ones(5)).type(torch.float32)
	assert SVM.linear(x_1, x_2) == 5
	
