#!/usr/bin/env python3

import argparse
import os


def hill_function(x, a, b, c):
    return (a * x ** c) / (x ** c + b ** c)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_deck', help='input deck', type=argparse.FileType('r'))
    args = parser.parse_args()

    print(args.input_deck)

    lines = args.input_deck.readlines()

    args.input_deck.close()

    X = {line.split()[0]: float(line.split()[2]) for line in lines}

    Y = hill_function(X['x'], X['a'], X['b'], X['c'])

    with open('output_deck', 'w') as f:
        f.writelines(lines)
        f.write(('Y = {}' + os.linesep).format(Y))
