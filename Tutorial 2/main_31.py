#!/usr/bin/env python
"""
Main file for assignment 3.1
RBF Network on a sin function

Authors: Kostis SZ, Romina Ariazza and Clara Tump
"""
import numpy as np
import generate_data

verbose = True

def part_3_1():

    sin, square = generate_data.sin_square(verbose=verbose)
    

def part_3_2():

    sin, square = generate_data.sin_square(verbose=verbose)
    sin.train_sin_x = generate_data.add_noise(sin.train_sin_x)
    sin.train_sin_y = generate_data.add_noise(sin.train_sin_y)
    sin.test_sin_x = generate_data.add_noise(sin.test_sin_x)
    sin.test_sin_y = generate_data.add_noise(sin.test_sin_y)



if __name__ == "__main__":
    part_3_1()
    # part_3_2()
