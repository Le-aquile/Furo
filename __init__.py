from .scrapers import scrape__page, get_pages_in_category
from .preprocessors import FastBPE
from .machine_learning import (
    relu, sigmoid, tanh, leaky_relu, softmax, linear, swift,
    NeuralNetwork, mse_loss, mse_loss_derivative, RLAgent,
    MAML, LinearRegression2D, DecisionTree, Perceptron
)
from .furo_gym import FallingGame