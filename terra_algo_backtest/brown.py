import math

import matplotlib.pyplot as plt
import numpy as np
from colour import Color

from .utils import pltShowWaitKey


def brownian_motion(N, T, h):
    """
    :param int N : the number of discrete steps
    :param int T: the number of continuous time steps
    :param float h: the variance of the increments
    """
    dt = 1.0 * T / N  # the normalizing constant
    random_increments = np.random.normal(0.0, 1.0 * h, N) * np.sqrt(
        dt
    )  # the epsilon values
    brownian_motion = np.cumsum(random_increments)  # calculate the brownian motion
    brownian_motion = np.insert(brownian_motion, 0, 0.0)  # insert the initial condition

    return brownian_motion.copy(), random_increments.copy()


def drifted_brownian_motion(mu, sigma, N, T, h, niceConnect=False):
    """
    :param float mu: drift coefficient
    :param float sigma: volatility coefficient
    :param int N : number of discrete steps
    :param int T: number of continuous time steps
    :param float h: variance of increments for brownian motion
    :param bool niceConnect: fix drift cuts
    :returns list: drifted Brownian motion
    """
    # standard brownian motion
    W, _ = brownian_motion(N, T, h)
    # the normalizing constant
    dt = 1.0 * T / N
    # generate the time steps
    time_steps = np.linspace(0.0, N * dt, N + 1)
    step_size = math.floor(N / len(mu))
    X = np.linspace(0.0, N * dt, N + 1)
    for muIndex in range(0, len(mu)):
        # get indexes inside time series
        start = step_size * muIndex
        end = step_size * (muIndex + 1) if muIndex < len(mu) else None
        # update original brownian motion with part drift
        X[start:end] = mu[muIndex] * time_steps[start:end] + sigma * W[start:end]
        # smooth the drift point
        if start > 0 and niceConnect:
            X[start:end] -= X[start] - X[start - 1]

    return X[0: len(X) - 1]


def runSamples(
    N,
    T,
    h,
    sigma,
    seed,
    drifts,
    varianceIncrements=None,
    continuousSteps=None,
    sigmaSteps=None,
    niceDriftConnect=False,
):
    """
    run brownian motions with different parameters to try the outcome = playground
    example runSamples(500, 14, 1, 50, 420, [100, -50, 0, 35, 10], [1, 2, 4, 8],
    [1, 2, 4, 8], [1, 2, 8, 32])
    :param int N : number of discrete steps
    :param int T: number of continuous time steps
    :param float h: variance of increments for brownian motion
    :param float sigma: volatility coefficient
    :param [float] mu: drift coefficient
    :param [float] varianceIncrements: variance increments
    :param [float] continuousSteps: continuous steps
    :param [float] sigmaSteps: sigma steps
    :param bool niceDriftConnect: fix drift cuts
    :returns list: drifted Brownian motion
    """
    np.random.seed(seed)
    # generate a brownian motion
    red = Color("red")
    colorId = 0

    # brownian continuous steps
    if continuousSteps:
        colors = list(red.range_to(Color("green"), len(continuousSteps)))
        for T in continuousSteps:
            X, epsilon = brownian_motion(N, T, h)
            plt.plot(np.linspace(0, X.size, X.size), X, colors[colorId].hex)
            colorId += 1
        plt.legend(continuousSteps)
        plt.title("Brownian motion, increasing continuous steps (T)")
        pltShowWaitKey()

    # brownian increasing increment variance
    if varianceIncrements:
        colorId = 0
        colors = list(red.range_to(Color("green"), len(varianceIncrements)))
        for T in varianceIncrements:
            X, epsilon = brownian_motion(N, T, h)
            plt.plot(np.linspace(0, X.size, X.size), X, colors[colorId].hex)
            colorId += 1
        plt.legend(varianceIncrements)
        plt.title("Brownian motion, increasing increment variance (h)")
        pltShowWaitKey()

    # drift increasing continuous steps
    if continuousSteps:
        colorId = 0
        colors = list(red.range_to(Color("green"), len(continuousSteps)))
        for T in continuousSteps:
            X = drifted_brownian_motion(drifts, sigma, N, T, h, niceDriftConnect)
            plt.plot(np.linspace(0, X.size, X.size), X, colors[colorId].hex)
            colorId += 1
        plt.legend(continuousSteps)
        plt.title("Drifted motion, increasing continuous steps (T)")
        pltShowWaitKey()

    # drift increasing increment variance
    if varianceIncrements:
        colorId = 0
        colors = list(red.range_to(Color("green"), len(varianceIncrements)))
        for T in varianceIncrements:
            X = drifted_brownian_motion(drifts, sigma, N, T, h, niceDriftConnect)
            plt.plot(np.linspace(0, X.size, X.size), X, colors[colorId].hex)
            colorId += 1
        plt.legend(varianceIncrements)
        plt.title("Drifted motion, increasing increment variance (h)")
        pltShowWaitKey()

    # drift increasing sigma
    if sigmaSteps:
        colorId = 0
        colors = list(red.range_to(Color("green"), len(sigmaSteps)))
        for T in sigmaSteps:
            X = drifted_brownian_motion(drifts, sigma, N, T, h, niceDriftConnect)
            plt.plot(np.linspace(0, X.size, X.size), X, colors[colorId].hex)
            colorId += 1
        plt.legend(sigmaSteps)
        plt.title("Drifted motion, increasing sigma (sigma)")
        pltShowWaitKey()


def simulationSamples(
    seed=1420, doPlot=False, steps=365 * 24, timeframe=60 * 60 * 4
) -> dict:
    """
    list of market model simulations to run against peg
    TODO: add and describe the test cases
    TODO: add luna crash data replay
    :returns dict: dictionary of test function prices arrays
    """
    prices = {}
    np.random.seed(seed)

    X = drifted_brownian_motion([-100, 10, -10, 10, -10], 1, steps, 1, 4, True)
    X = X - np.min(X) + 0.1
    if doPlot:
        plt.plot(
            np.linspace(0, X.size, X.size), X, "g", label="drifted big dump, steady"
        )
    prices["drifted big dump, steady"] = {"data": X, "timeframe": timeframe}

    X = drifted_brownian_motion([-10, 10, -10, 10, 100], 1, steps, 1, 4, True)
    X = X - np.min(X) + 0.1
    if doPlot:
        plt.plot(
            np.linspace(0, X.size, X.size), X, "r", label="drifted steady, big pump"
        )
    prices["drifted steady, big pump"] = {"data": X, "timeframe": timeframe}

    X, epsilon = brownian_motion(steps, 1, 2)
    X = X - np.min(X) + 0.1
    if doPlot:
        plt.plot(np.linspace(0, X.size, X.size), X, "y", label="steady brownian")
    prices["steady brownian"] = {"data": X, "timeframe": timeframe}

    X, epsilon = brownian_motion(steps, 1, 2)
    X = X - np.min(X) + 0.1
    if doPlot:
        plt.plot(np.linspace(0, X.size, X.size), X, "b", label="wild brownian")
    prices["wild brownian"] = {"data": X, "timeframe": timeframe}
    if doPlot:
        plt.legend()
        plt.title("Repeg test scenarios")
        plt.show()

    return prices
