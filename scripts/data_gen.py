import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

import argparse
from utils import *
from manipulators.link_manipulator_2d import LinkManipulator2D
from manipulators.link_manipulator_3d import LinkManipulator3D

link_min = 0.1
link_max = 0.38

angle_min = -np.pi
angle_max = np.pi

def main_2d(dof, num_examples, datasetname):
    all_reachability_maps = np.zeros((num_examples, 40, 40)) # TODO fixed resolution

    # generate random configuration
    cfgs = np.random.uniform(link_min, link_max, (num_examples, dof))

    with open(datasetname+"/manipulator_configs.npy", "wb") as f:
        np.save(f, cfgs)

    for i in tqdm(range(num_examples)):
        cfg = cfgs[i]
        m = LinkManipulator2D(cfg)

        for jid in np.arange(dof):
            anum = 0
            for a in np.arange(-np.pi, np.pi, 0.2):
                anum += 1
                m.fix_joint(jid, a)
                all_cfg = m.generate_joint_configurations()
                for cfg in all_cfg:
                    p, has_collision = m.get_eff_pose(cfg)
                    if not has_collision:
                        m.update_map(p, a) 
        all_reachability_maps[i] = m.get_reachability_map(anum)

    with open(datasetname+"/reachability_maps.npy", "wb") as f:
        np.save(f, all_reachability_maps)


def main_3d(dof, num_examples, datasetname):
    dh_params = np.zeros((4, 4))
    dh_params[0,1] = 0.089
    dh_params[1:,1] = 0.10475

    all_reachability_maps = np.zeros((num_examples, 10, 10, 10))
    linkcfgs = np.random.uniform(link_min, link_max, (num_examples, 4))
    anglecfgs = np.random.uniform(angle_min, angle_max, (num_examples, 3))
    cfgs = np.hstack((linkcfgs, anglecfgs))
    with open(datasetname+"/manipulator_configs.npy", "wb") as f:
        np.save(f, cfgs)

    for i in tqdm(range(num_examples)):
        cfg = cfgs[i]
        dh_params[:,2] = cfg[0:4]
        dh_params[:3,3] = cfg[4:7]
        m = LinkManipulator3D(dh_params)

        all_reachability_maps[i] = m.get_reachability_map()
        
    with open(datasetname+"/reachability_maps.npy", "wb") as f:
        np.save(f, all_reachability_maps)





if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, required=True, help="Dimension of the configuration space")
    parser.add_argument("--dof", type=int, required=True, help="Degree of freedom (only supports 4 or 5)")
    parser.add_argument("--num_examples", type=int, required=True, help="Number of examples to generate")

    args = parser.parse_args()


    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    dataset_name = f"data/{args.dim}d_{args.dof}dof_{args.num_examples}_{timestamp}"

    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)

    if args.dim == 2:
        main_2d(args.dof, args.num_examples, dataset_name)
    else:
        main_3d(args.dof, args.num_examples, dataset_name)