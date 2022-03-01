#!/bin/bash

python3 get_learning_metrics.py [0.8,0.9,1.0,1.1,1.2,1.3,1.5] 4 all 2
python3 get_learning_metrics.py [0.8,0.9,1.0,1.1,1.2,1.3,1.5] 5 all 2
python3 get_learning_metrics.py [0.8,0.9,1.0,1.1,1.2,1.3,1.5] 6 all 2
python3 get_learning_metrics.py [0.8,0.9,1.0,1.1,1.2,1.3,1.5] 7 all 2
python3 get_learning_metrics.py [0.8,0.9,1.0,1.1,1.2,1.3,1.5] 13 all 2
python3 get_learning_metrics.py [0.8,0.9,1.0,1.1,1.2,1.3,1.5] 14 all 2
python3 get_learning_metrics.py [0.8,0.9,1.0,1.1,1.2,1.3,1.5] 15 all 2

python3 get_learning_metrics.py [0.8,0.9,1.0,1.1,1.2,1.3,1.5] 4 all 5
python3 get_learning_metrics.py [0.8,0.9,1.0,1.1,1.2,1.3,1.5] 5 all 5
python3 get_learning_metrics.py [0.8,0.9,1.0,1.1,1.2,1.3,1.5] 6 all 5
python3 get_learning_metrics.py [0.8,0.9,1.0,1.1,1.2,1.3,1.5] 7 all 5
python3 get_learning_metrics.py [0.8,0.9,1.0,1.1,1.2,1.3,1.5] 13 all 5
python3 get_learning_metrics.py [0.8,0.9,1.0,1.1,1.2,1.3,1.5] 14 all 5
python3 get_learning_metrics.py [0.8,0.9,1.0,1.1,1.2,1.3,1.5] 15 all 5

python3 get_learning_metrics.py 0 4 all 4 breast_cancer
python3 get_learning_metrics.py 0 5 all 4 breast_cancer
python3 get_learning_metrics.py 0 6 all 4 breast_cancer
python3 get_learning_metrics.py 0 7 all 4 breast_cancer
python3 get_learning_metrics.py 0 13 all 4 breast_cancer
python3 get_learning_metrics.py 0 14 all 4 breast_cancer
python3 get_learning_metrics.py 0 15 all 4 breast_cancer

