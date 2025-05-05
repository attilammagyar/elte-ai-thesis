import os.path
import sys

import numpy as np
import matplotlib.pyplot as plt


def main(argv):
    out_file = os.path.join(os.path.dirname(argv[0]), "..", "img", "softmax-temperature.png")

    x = np.linspace(0.0, 1.0, 1000)
    y = f(x)

    plt.figure(figsize=(12, 6))
    plt.plot(x, softmax(y, 0.5), "-", label="T=0.5", linewidth=3)
    plt.plot(x, softmax(y, 1.0), "--", label="T=1.0", linewidth=3)
    plt.plot(x, softmax(y, 1.7), ":", label="T=1.7", linewidth=3)
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.savefig(out_file, bbox_inches="tight")

    return 0


def f(x):
    return (N(x, 0.15, 0.07) + N(x, 0.5, 0.12) + N(x, 0.8, 0.09)) / 3.0


def N(x, u, s):
    s = 2.0 * (s ** 2.0)

    return np.exp(- (x - u) ** 2.0 / s) / (np.pi * s) ** 0.5


def softmax(x, t):
    return np.exp(x / t) / np.exp(x / t).sum()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
