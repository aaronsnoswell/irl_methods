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
 * [openAI gym](https://github.com/openai/gym)

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

### Linear Programming IRL by [Ng and Russell, 2000](http://ai.stanford.edu/~ang/papers/icml00-irl.pdf).

 * [x] Planned
 * [x] [Implemented](linear_programming.py)
 * [x] Tested
 * [ ] Examples written

