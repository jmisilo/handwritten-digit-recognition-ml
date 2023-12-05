import torch

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
