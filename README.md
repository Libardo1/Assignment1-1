# Assigment 1

This repository contains all my solutions for the [Assigment 1](http://cs224d.stanford.edu/assignment1/index.html) of the course CS224d: Deep Learning for Natural Language Processing. All the code can be found in the source folder. A report for the written assignment is in the folder report.

### Requirements
* Numpy
* Matplotlib

## Usage

```
$ bash source/cs224d/datasets/get_datasets.sh 
cd source/
python q3_run.py
bash test.sh
python q4_sentiment.py

python q4_sentiment.py --help
usage: q4_sentiment.py [-h] [-p PASSWORD] [-s STEPS] [-e EVERY] [-l LEARNING]

optional arguments:
  -h, --help            show this help message and exit
  -p PASSWORD, --password PASSWORD
                        Password for the robotanara mail.(default=None)
  -s STEPS, --steps STEPS
                        number of training steps (default=10000)
  -e EVERY, --every EVERY
                        Show result for every x steps (default=100)
  -l LEARNING, --learning LEARNING
                        learning rate (default=3.0)
```


## Example

```
$ cd source/
$ python q4_sentiment.py -s 2 -e 1


>> All the regularization params are = [  6.78575513e-06   9.68394763e-06   2.29498087e-05   7.80778107e-05
   2.31349338e-04   2.57305890e-04   1.55843947e-03   2.74343961e-03
   5.52506312e-03   2.00527059e-02]
>> Training for reg=0.000007
iter 1: 1.939244
iter 2: 1.926059
Train accuracy (%): 25.959738
Dev accuracy (%): 26.248865
>> Training for reg=0.000010
iter 1: 1.939312
iter 2: 1.926126
Train accuracy (%): 25.959738
Dev accuracy (%): 26.248865
>> Training for reg=0.000023
iter 1: 1.939619
iter 2: 1.926433
Train accuracy (%): 25.959738
Dev accuracy (%): 26.248865
>> Training for reg=0.000078
iter 1: 1.940898
iter 2: 1.927707
Train accuracy (%): 25.959738
Dev accuracy (%): 26.248865
>> Training for reg=0.000231
iter 1: 1.944455
iter 2: 1.931248
Train accuracy (%): 25.959738
Dev accuracy (%): 26.248865
>> Training for reg=0.000257
iter 1: 1.945057
iter 2: 1.931848
Train accuracy (%): 25.959738
Dev accuracy (%): 26.248865
>> Training for reg=0.001558
iter 1: 1.975247
iter 2: 1.961897
Train accuracy (%): 25.959738
Dev accuracy (%): 26.248865
>> Training for reg=0.002743
iter 1: 2.002742
iter 2: 1.989245
Train accuracy (%): 25.959738
Dev accuracy (%): 26.248865
>> Training for reg=0.005525
iter 1: 2.067284
iter 2: 2.053368
Train accuracy (%): 25.948034
Dev accuracy (%): 26.248865
>> Training for reg=0.020053
iter 1: 2.404367
iter 2: 2.386675
Train accuracy (%): 25.948034
Dev accuracy (%): 26.339691

>> === Recap ===
Reg   Train   Dev
6.785755E-06  25.959738 26.248865
9.683948E-06  25.959738 26.248865
2.294981E-05  25.959738 26.248865
7.807781E-05  25.959738 26.248865
2.313493E-04  25.959738 26.248865
2.573059E-04  25.959738 26.248865
1.558439E-03  25.959738 26.248865
2.743440E-03  25.959738 26.248865
5.525063E-03  25.948034 26.248865
2.005271E-02  25.948034 26.339691

Best regularization value: 2.005271E-02
Test accuracy (%): 28.959276
The duration of the whole training with 20 steps is 19.91 seconds,
which is equal to:  0:0:0:19  (DAYS:HOURS:MIN:SEC)

```
