import torch

# x = torch.empty(5, 3)
# print(x)

# x = torch.rand(5, 3)
# print(x)

# x = x.new_ones(5, 3, dtype=torch.double)
# print(x)
# y = torch.rand_like(x, dtype=torch.float)
# print(y)
# print(y.size())
#
# result = torch.empty(5,3)
# torch.add(x, y, out=result)
# print(result)

# print(x[:,:1])
#
# x = torch.rand(4, 4)
# y = x.view(16)
# z = x.view(-1,8)
# print(x.size(),y.size(),z.size())

# x = torch.rand(1)
# print(x)
# print(x.item())

# a = torch.ones(5)
# b = a.numpy()
# print(b)
# a.add_(1)
# print(a)
# print(b)

# import numpy as np
# a = np.ones(5)
# b = torch.from_numpy(a)
# print(b)
# np.add(a, 1, out=a)
# print(a)
# print(b)

print(torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.empty(5, 3)
    device = torch.device('cuda')
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to('cpu', torch.double))
