# -*- coding: utf-8 -*-
"""Multidimensional discrete <-> continuous conversions

Copyright 2018 Aaron Snoswell
"""

import math
import collections
import numpy as np


def adc(vec, low_bounds, high_bounds, size):
    """Recursive multidimensional analog to digital conversion

    Converts a vector of continuous coordinates to a single index into a
    discretised version of the same space.

    Args:
        vec (numpy array): Vector of coordinates to convert
        low (numpy array): Vector of minimum values for each dimension
        high (numpy array): Vector of maximum values for each dimension
        size (list): List of discretisation levels for each dimension

    Returns:
        (int): Discrete index into the discretised space
    """

    assert len(vec) == len(low_bounds) == len(high_bounds) == len(size), \
        "Vectors are not correct shape"
    

    def _adc(val, min_val, max_val, steps):
        """Analog to digital level conversion

        Args:
            val (float): Analog value to convert
            min_val (float): Minimum possible analog value
            max_val (float): Maximum possible analog value
            steps (int): Number of discrete levels

        Returns:
            (int): Discrete index of the resulting digital level
        """
        return int((val - min_val) / (max_val - min_val) * (steps-1))


    if len(vec) == 1:
        # Reached single dimensional case - apply analog to digital formula
        return _adc(vec[0], low_bounds[0], high_bounds[0], size[0])

    else:
        # Pop off leading elements
        val, vec = vec[0], vec[1:]
        min_val, low_bounds = low_bounds[0], low_bounds[1:]
        max_val, high_bounds = high_bounds[0], high_bounds[1:]
        steps, size = size[0], size[1:]
        size_of_lower_dimensions = np.prod(size)

        # Compute the current dimension's index
        index = _adc(val, min_val, max_val, steps) * size_of_lower_dimensions

        # Recurse
        index += adc(vec, low_bounds, high_bounds, size)

        # Convert to int
        return int(index)


def dac(index, low_bounds, high_bounds, size):
    """Recursive multidimensional digital to analog conversion

    Converts a single index into a discreteised multidimensional space to a
    vector of approximate continuous coordinates into that space.

    Args:
        x (integer): Discrete index to convert
        low_bounds (numpy array): Vector of minimum values for each dimension
        high_bounds (numpy array): Vector of maximum values for each dimension
        size (list): List of discretisation levels for each dimension

    Returns:
        (numpy array): Vector of approximate continuous coordinates
    """

    assert len(low_bounds) == len(high_bounds) == len(size), \
        "Vectors are not correct shape"


    def _dac(x, min_val, max_val, steps):
        """Digital to analog level conversion

        Args:
            x (int): Digital value to convert
            min_val (float): Minimum possible analog value
            max_val (float): Maximum possible analog value
            steps (int): Number of discrete levels

        Returns:
            (float): Analog value of the given index
        """
        return ((max_val - min_val) / steps) * (x + 0.5) + min_val

    
    def flatten(l):
        """Flatten an irregular recursive list of lists

        Args:
            l (list): A list of arbitrarily nested lists

        Returns:
            (list): Flattened list containing ordered elements of all
                sub-lists
        """
        for el in l:
            if isinstance(el, collections.Iterable) \
                and not isinstance(el, (str, bytes)):
                yield from flatten(el)
            else:
                yield el
        

    if len(low_bounds) == 1:
        # Reached single dimensional case - apply digital to analog formula
        return _dac(index, low_bounds[0], high_bounds[0], size[0])

    else:
        # Pop off leading elements
        min_val, low_bounds = low_bounds[0], low_bounds[1:]
        max_val, high_bounds = high_bounds[0], high_bounds[1:]
        steps, size = size[0], size[1:]
        size_of_lower_dimensions = np.prod(size)

        # Compute the current dimension's index
        current_index = math.floor(index / size_of_lower_dimensions)

        # Recurse
        ret = [
            _dac(current_index, min_val, max_val, steps),
            dac(
                index - current_index * size_of_lower_dimensions,
                low_bounds,
                high_bounds,
                size
            )
        ]

        # Flatten the list of lists
        return np.array(list(flatten(ret)))


if __name__ == "__main__":
    # Run simple test case / demo

    num_steps = 20
    x = np.linspace(0, 2*math.pi, 20)
    y = np.sin(x)
    y_disc = [adc([yi], [-1], [1], [num_steps]) for yi in y]
    y_cts = [dac(yi, [-1], [1], [num_steps]) for yi in y_disc]

    # Plot results
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ln1 = ax2.plot(x, y_disc, 'r.-', label="Discretised signal")
    ax2.set_ylabel('Discrete indices', color='r')
    ax2.set_yticks(range(0, num_steps))
    ax2.set_ylim([-1, num_steps])
    ax2.tick_params('y', colors='r')

    ln2 = ax1.plot(x, y, 'b.-', label="Original data")
    ln3 = ax1.plot(x, y_cts, '.-', color="deepskyblue", label="Approximate continuous reconstruction")
    ax1.set_ylabel('Continuous values', color='b')
    ax1.tick_params('y', colors='b')
    ax1.grid(axis='x')
    ax2.set_xticks(x)

    # Add legend
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc=0)

    plt.title("Comparison of discrete and continuous conversions")
    fig.tight_layout()
    plt.grid()
    plt.show()
