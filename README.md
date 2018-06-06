# IRL Methods

High-quality reference implementations of various algorithms for Inverse
Reinforcement Learning.

This code is designed as a learning and research aid. It is not recommended
(or tested) for prodcution usage.

(c) 2018 Aaron Snoswell

## Requirements

Tested and developed with Python 3.5.5 on Windows. Requirements are handled
by [setup.py](/setup.py), but currently include

 * [numpy](http://www.numpy.org/)
 * [cvxopt](http://cvxopt.org/)
 * [gym](https://github.com/openai/gym)

## Installation

Clone this repository, then let `setup.py` do it's thing. I suggest you use a
python environment manager like [Conda](https://conda.io/).

```
git clone https://github.com/aaronsnoswell/irl_methods
cd irl_methods
python setup.py install
```

A PIP package will be forthcoming when the package is more mature.

## Usage

Documentation coming. For now check the docstrings in the individual modules.

## Status

### Linear Programming IRL by [Ng and Russell, 2000](http://ai.stanford.edu/~ang/papers/icml00-irl.pdf)

 * [x] Planned
 * [x] [Implemented](/irl_methods/linear_programming.py)
 * [x] Tested
 * [x] [Example written](/irl_methods/examples/ex_lp_gridworld.py)

### Linear Programming IRL for large state spaces by [Ng and Russell, 2000](http://ai.stanford.edu/~ang/papers/icml00-irl.pdf)

 * [x] Planned
 * [x] [Implemented](/irl_methods/large_linear_programming.py)
 * [ ] Tested
 * [ ] Example written

### Trajectory based Linear Programming IRL by [Ng and Russell, 2000](http://ai.stanford.edu/~ang/papers/icml00-irl.pdf)

 * [x] Planned
 * [x] [Implemented](/irl_methods/trajectory_linear_programming.py)
 * [ ] Tested
 * [ ] Example written

### Maximum-margin IRL by [Ratliff, Bagnell & Zinkevich, 2006](https://www.ri.cmu.edu/pub_files/pub4/ratliff_nathan_2006_1/ratliff_nathan_2006_1.pdf)

 * [x] [Planned](/irl_methods/maximum_margin.py)
 * [ ] Implemented
 * [ ] Tested
 * [ ] Example written

### Projection based IRL by [Abbeel and Ng, 2004](https://people.eecs.berkeley.edu/~russell/classes/cs294/s11/readings/Abbeel+Ng:2004.pdf)

 * [x] Planned
 * [X] [Implemented](/irl_methods/projection.py)
 * [ ] Tested
 * [ ] Examples written

### Maximum margin IRL by [Ratliff, Bagnell & Zinkevich, 2006](https://www.ri.cmu.edu/pub_files/pub4/ratliff_nathan_2006_1/ratliff_nathan_2006_1.pdf)

 * [x] [Planned](/irl_methods/maximum_margin.py)
 * [ ] Implemented
 * [ ] Tested
 * [ ] Examples written

### Maximum entropy IRL by [Ziebart, Bagnell and Dey, 2008](https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf)

 * [x] [Planned](/irl_methods/maximum_entropy.py)
 * [ ] Implemented
 * [ ] Tested
 * [ ] Examples written

### Deep Maximum entropy IRL by [Wulfmeier, Ondrùška and Posner, 2016](https://arxiv.org/pdf/1507.04888.pdf)

 * [x] [Planned](/irl_methods/deep_maximum_entropy.py)
 * [ ] Implemented
 * [ ] Tested
 * [ ] Examples written

