import numpy as np
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

from ftwsnet import FTWSNetwork3D, FTWSNetwork2D
from manipulators.link_manipulator_2d import LinkManipulator2D
from manipulators.link_manipulator_3d import LinkManipulator3D


def gradient_descent_optimize_2d(res, neural_net, g, other_matrix, area, learning_rate=0.005, num_iterations=20, num_initializations=30):
    # Optimization loop
    res_bound = res // 4
    max_g_value = float('-inf')
    best_x = None
    plots = []
    for _ in range(num_initializations):
        it = 0

        # Generate the first 4 elements between 0.1 and 0.8
        x = 0.1 + 0.4 * torch.rand(4)
        x.requires_grad = True
        
        optimizer = optim.SGD([x], lr=learning_rate)
        
        losses = []
        while it < num_iterations:
            # Forward pass through the neural network
            y = neural_net(x)
      
            yoverlap  = y.view(res, res)[res_bound:(res-res_bound), res_bound:(res-res_bound)]
            # Compute the value of g
            value_g = g(yoverlap, other_matrix, area)

            losses.append(value_g.item())

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

            x.grad.zero_()

            it += 1
        plots.append(losses)

    print("max predicted coverage=", max_g_value)
    print("corresponding best x=", best_x)
    return best_x.detach().numpy()

def gradient_descent_optimize_3d(res, neural_net, g, other_matrix, area, learning_rate=0.005, num_iterations=30, num_initializations=30):
    # Optimization loop
    max_g_value = float('-inf')
    best_x = None
    for _ in range(num_initializations):
        it = 0
        
        # Generate the first 4 elements between 0.1 and 0.8
        first_four = 0.1 + 0.28 * torch.rand(4)

        # Generate the last 4 elements between -pi and pi
        last_four = np.pi * (2 * torch.rand(3) - 1)

        # Concatenate the two tensors along the first dimension to create a tensor of length 8
        x = torch.cat((first_four, last_four))
        
        x.requires_grad = True
        optimizer = optim.SGD([x], lr=learning_rate)
        
        while it < num_iterations:
            # Forward pass through the neural network
            y = neural_net(x)

            # Compute the value of g
            value_g = g(y.view(res, res, res), other_matrix, area)

            max_g_value = max(max_g_value, value_g.item())
            if max_g_value == value_g.item():
                best_x = x

            # Backpropagate gradients of g with respect to x
            optimizer.zero_grad()
            value_g.backward()
            
            # Update x using gradient descent
            with torch.no_grad():
                x += learning_rate * x.grad
                if (x[0:4] < 0.1).any():
                    it = num_iterations+1
                elif (x[0:4] > 0.38).any():
                    it = num_iterations+1
                elif (x[4:8] < -np.pi).any():
                    it = num_iterations+1
                elif (x[4:8] > np.pi).any():
                    it = num_iterations+1

            # Manually zero the gradients after updating x
            x.grad.zero_()

            it += 1
    print("max coverage=", max_g_value)
    print("corresponding best x=", best_x)
    
    return best_x.detach().numpy()

def load_model(dim, res, ckpt):
    if dim == 2:
        model = FTWSNetwork2D(res)
    else:
        model = FTWSNetwork3D(res)
    
    # Load the model checkpoint
    model.load_state_dict(torch.load(ckpt))
    return model


def coverage_2d(y, other_matrix, area):
    # upsample y
    matrix_40_unsqueezed = y.unsqueeze(0).unsqueeze(0)
    matrix_400 = F.interpolate(matrix_40_unsqueezed, size=(400, 400), mode='bilinear', align_corners=False)
    # Remove batch dimension and channel dimension
    matrix_400 = matrix_400.squeeze(0).squeeze(0)

    return torch.sum(torch.mul(matrix_400, other_matrix)) / area

def coverage_3d(y, other_matrix, area):
    # Example function to maximize: element-wise dot product with another matrix
    return torch.sum(torch.mul(y, other_matrix)) / area  # Element-wise dot product

if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, required=True, help="Dimension of the task space")
    parser.add_argument("--res", type=int, required=True, help="Resolution of the task space")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--task", type=str, required=True, help="Task name")

    args = parser.parse_args()

    res = args.res
    res_bound = res // 4

    model = load_model(args.dim, args.res, args.ckpt)

    # task
    trajectory_np = np.load(f"example_tasks/{args.dim}d/{args.task}.npy")
    trajectory_matrix = torch.from_numpy(trajectory_np).float()
    area = torch.sum(trajectory_matrix)

    design_threshold = 0.9
    actual_coverage = 0.0

    iteration = 0

    while actual_coverage < design_threshold and iteration < 10:
        if args.dim == 2:
            x = gradient_descent_optimize_2d(args.res, model, coverage_2d, trajectory_matrix, area)
            m = LinkManipulator2D(x, args.res)
            rmap = m.get_reachability_map_new()
            rmap = torch.from_numpy(rmap).float()
            
            rmap_overlap = rmap.view(res, res)[res_bound:(res-res_bound), res_bound:(res-res_bound)]
            actual_coverage = coverage_2d(rmap_overlap, trajectory_matrix, area)

        else:
            x = gradient_descent_optimize_3d(args.res, model, coverage_3d, trajectory_matrix, area)
            dh_params = np.zeros((4, 4))
            dh_params[0,1] = 0.089
            dh_params[1:,1] = 0.10475
            dh_params[:,2] = x[0:4]
            dh_params[:3,3] = x[4:7]
            m = LinkManipulator3D(dh_params)

            rmap = m.get_reachability_map()
            rmap = torch.from_numpy(rmap).float()
            
            rmap_overlap = rmap.view(res, res, res)[res_bound:(res-res_bound), res_bound:(res-res_bound)]
            actual_coverage = coverage_3d(rmap_overlap, trajectory_matrix, area)
        
        print("Actual Coverage: ", actual_coverage)
        
        if actual_coverage >= design_threshold:
            print("finished at iteration ", iteration)
            exit(0)
        
        iteration += 1
        
    print("design failed after 10 iterations")