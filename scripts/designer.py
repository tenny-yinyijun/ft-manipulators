import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import FTWSNetwork
from manipulator import LinkManipulator

# Define the function g
def g(y, other_matrix, area):
    # upsample y
    matrix_40_unsqueezed = y.unsqueeze(0).unsqueeze(0)
    matrix_400 = F.interpolate(matrix_40_unsqueezed, size=(400, 400), mode='bilinear', align_corners=False)
    # Remove batch dimension and channel dimension
    matrix_400 = matrix_400.squeeze(0).squeeze(0)

    # Example function to maximize: element-wise dot product with another matrix
    return torch.sum(torch.mul(matrix_400, other_matrix)) / area  # Element-wise dot product

# Normal gradient descent optimization using PyTorch
def gradient_descent_optimize(neural_net, g, other_matrix, area, learning_rate=0.01, num_iterations=100, num_initializations=100):
    # Optimization loop
    max_g_value = float('-inf')
    best_x = None
    for n in range(num_initializations):
      it = 0
      # x = (torch.rand(4) * 0.7) + 0.1
      
      # Generate the first 4 elements between 0.1 and 0.8
      x = 0.1 + 0.4 * torch.rand(4)
      
      x.requires_grad = True
      # print("initilization #", n, ", init x value = ", x)
      optimizer = optim.SGD([x], lr=learning_rate)
      
      while it < num_iterations:
          # Forward pass through the neural network
          y = neural_net(x)

          yoverlap  = y.view(40, 40)[10:30, 10:30]
          # Compute the value of g
          value_g = g(yoverlap, other_matrix, area)
          # print(torch.sum(other_matrix))
          # print(other_matrix.shape)
          # print(torch.sum(y))
          # print(y.shape)
          # print("iter=", it, "coverage=", (value_g.item())/14)
          max_g_value = max(max_g_value, value_g.item())
          if max_g_value == value_g.item():
            best_x = x

          # Backpropagate gradients of g with respect to x
          optimizer.zero_grad()
          value_g.backward()
          
          # Update x using gradient descent
          with torch.no_grad():
              x += learning_rate * x.grad
              if (x < 0.1).any():
                it = num_iterations+1
              elif (x > 0.5).any():
                it = num_iterations+1

          # Manually zero the gradients after updating x
          x.grad.zero_()

          it += 1
    # print("finished all iterations")
    print("max predicted coverage=", max_g_value)
    print("corresponding best x=", best_x)
    return best_x.detach().numpy()
    # # Return the optimized value of x and the corresponding value of g
    # return x.detach().numpy(), g(neural_net(x), other_matrix).item()

# Generate another random matrix for the element-wise dot product
trajectory_np = np.load("task3.npy")
# print(trajectory_np.shape)
other_matrix = torch.from_numpy(trajectory_np).float()
area = torch.sum(other_matrix)

# Create an instance of the neural network
neural_net = FTWSNetwork()
FTWSNetwork.load_state_dict(neural_net, torch.load("weights0606-t1000-e4000-ftwsnet.pth"))
# FTWSNetwork.load_state_dict(neural_net, torch.load("weights0522-t1000-e4000-ftwsnet.pth"))
# FTWSNetwork.load_state_dict(neural_net, torch.load("weights0417-t9000-e10000-ftwsnet.pth"))

# Verification Loop, needs user0set threshold
design_threshold = 0.9
actual_coverage = 0.0
dh_params = np.zeros((4, 4))

while actual_coverage < design_threshold:
  # Perform optimization
  x = gradient_descent_optimize(neural_net, g, other_matrix, area)
  # Calculate the actual coverage
  # dh_params[:,2] = x[0:4]
  # dh_params[:3,3] = x[4:7]
  m = LinkManipulator(x)
  rmap = m.get_reachability_map(max_value=None)
  
  rmap = torch.from_numpy(rmap).float()
  
  rmap_overlap = rmap.view(40, 40)[10:30, 10:30]
  
  actual_coverage = g(rmap_overlap, other_matrix, area)
  
  print("Actual Coverage: ", actual_coverage)
  
  if actual_coverage < 1.0:
    # produce report
    pass
    

# for i in range(0, 3):
#   print("")
  # Perform optimization
  # gradient_descent_optimize(neural_net, g, other_matrix, area)

# motion planning code in:
# mms-project/combined/2d-sys