# This training script try plotting and saving train-val-losses and model weights.

import numpy as np
import time
import os
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import argparse
from torch import nn
from torch.utils.data import DataLoader
from dataset import FTWSDataset
from datetime import datetime
from ftwsnet import FTWSNetwork3D, FTWSNetwork2D


def train(train_loader, val_loader, dim, res, output_dir, save_freq=10):
    if dim == 2:
        model = FTWSNetwork2D(res)
    else:
        model = FTWSNetwork3D(res)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    def train_one_epoch():
        running_loss = 0.
        last_loss = 0.

        for i, data in enumerate(train_loader):
            x1, target = data
            optimizer.zero_grad()
            x1 = x1.float()
            outputs = model(x1)
            
            target = target.view(target.shape[0], -1)
            target = target.to(torch.float32)


            loss = loss_fn(outputs, target)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            last_loss = running_loss
            running_loss = 0.

        return last_loss
    
    epoch_number = 0

    train_losses = []
    val_losses = []

    EPOCHS = 500

    best_vloss = 1_000_000.

    start = time.time()
    for _ in tqdm(range(EPOCHS)):
        model.train(True)
        avg_loss = train_one_epoch()

        running_vloss = 0.0
        
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vx1, vy = vdata
                vx1 = vx1.float()
                voutputs = model(vx1)
                vy = vy.view(vy.shape[0], -1)
                vloss = loss_fn(voutputs, vy)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        # print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        train_losses.append(avg_loss)
        val_losses.append(avg_vloss)
        
        if epoch_number % save_freq == 0:
            model_path = f"{output_dir}/ep{epoch_number}.pth"
            torch.save(model.state_dict(), model_path)

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            # save model
            model_path = f"{output_dir}/best.pth"
            torch.save(model.state_dict(), model_path)

        epoch_number += 1
    total_time = time.time() - start
    print('Training took {} seconds'.format(total_time))

    print("length of train losses: ", len(train_losses))
    print("length of val losses: ", len(val_losses))

    # plot losses
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.legend()
    plt.savefig(f"{output_dir}/losses.png")

    np.save(f"{output_dir}/train_loss.npy", train_losses)
    np.save(f"{output_dir}/val_loss.npy", val_losses)
        

def load_data(dataset_name, num_examples):
    manipulator_config_file = f"data/{dataset_name}/manipulator_configs.npy"
    reachability_map_file = f"data/{dataset_name}/reachability_maps.npy"

    manipulator_configs = np.load(manipulator_config_file)
    reachability_maps = np.load(reachability_map_file)
    if num_examples is not None:
        manipulator_configs = manipulator_configs[:num_examples]
        reachability_maps = reachability_maps[:num_examples]
    
    # split
    train_size = int(num_examples * 0.8)
    train_configs = manipulator_configs[:train_size]
    train_maps = reachability_maps[:train_size]
    val_configs = manipulator_configs[train_size:]
    val_maps = reachability_maps[train_size:]

    training_data = FTWSDataset(train_configs, train_maps)
    validation_data = FTWSDataset(val_configs, val_maps)

    res = reachability_maps.shape[1]

    print('Training set has {} instances'.format(len(training_data)))
    print('Validation set has {} instances'.format(len(validation_data)))

    train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(validation_data, batch_size=64, shuffle=True)

    return train_loader, val_loader, res


if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, required=True, help="Dimension of the task space")
    parser.add_argument("--dataset", type=str, required=True, help="Name of dataset to train on")
    parser.add_argument("--size", type=int, required=True, help="Size of dataset to train on")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model_name = f"ftwsnet_{args.size}_{timestamp}"
    checkpoint_path = f"checkpoints/{args.dataset}/{model_name}"

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # data loaders
    train_loader, val_loader, res = load_data(args.dataset, args.size)

    # train
    train(train_loader, val_loader, args.dim, res, checkpoint_path, save_freq=10)