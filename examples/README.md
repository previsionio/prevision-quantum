Examples
========

This directory contains multiple examples that can be run with
prevision-quantum-nn.

# Classification

Binary classification datasets are the most common tasks performes in Machine
Learning applications. Here, we propose two

## Moon

Number of features: 2 (x and y)
Number of categories: 2

## Spirals

Number of features: 2 (x and y)
Number of categories: 2

# Multiclassification

## Iris

Number of features: 8 Number of categories: 3

# Regression

## Sinusoid

This is a simple example that shows how to use prevision-quantum-nn for
regression tasks. Number of features: 1

# Reinforcement Learning examples

## LunarLander-v2

Number of features: 8 Number of possible actions: 4

## MountainCar-v0

This problem requires very little qubits to encode a proper value function.
However, this problem is complicated as the only reward given to the
Reinforcement Learning agent is at the end of the episode - only if the agent
has successfully brought the car to the target... Otherwise, the reward is -1
per each step that did not bring the car to the target.

Number of features: 2 Number of possible actions: 3

# Going further with Open-ML datasets

Open-ML provides with a great repository of ML datasets on which people can
benchmark their algorithms. In these examples, you will find a way to simply
import a dataset from the OpenML repository, and try prevision-quantum-nn!
In order to guide you a bit before you do so, we have selected 5 datasets of
reasonable complexity.

We consider that each dataset with less than 10 features can be easily encoded
with angle/displacement/squeezing embeddings. For datasets with more than 10
features, it is recommended to perform dimension reduction, or to use
amplitude/mottonen embeddings.

## Dataset 1462

Number of features: 4
