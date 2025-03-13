# Robust Task-Based Design of Modular Manipulators with Single Joint Failure

## Getting Started

1. [Installation](#installation)
2. [Data Generation](#data-generation)
3. [Training the Reachability Predictor](#training-the-reachability-predictor)
4. [Running the Designer](#running-the-designer)
5. [Running the Motion Planner](#running-the-motion-planner)

## Installation

Packages required:
- numpy
- matplotlib
- pytorch

## Data Generation

To generate training data for reachability prediction:

```bash
python scripts/data_gen.py --dim 2 --dof 4 --res 40 --num_examples 2000 # currently only suports dof = 4 or 5
```

## Training the Rechability Predictor
```bash
python scripts/train.py --dim 2 --dataset <dataset-name>  --size <num-training-instance>
```

## Running the Designer
To design a failure-tolerant manipulator for any of the example tasks under `example_tasks/`, run:
```bash
python scripts/designer.py --predictor --task  
```

## Running the Motion Planner

TODO