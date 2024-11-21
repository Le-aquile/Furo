import numpy as np

def adam_optimizer(grad_func, theta, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, steps=100):
    m = np.zeros_like(theta)  # Initialize first moment
    v = np.zeros_like(theta)  # Initialize second moment
    for t in range(1, steps + 1):
        grad = grad_func(theta)  # Compute gradient
        m = beta1 * m + (1 - beta1) * grad  # Update first moment
        v = beta2 * v + (1 - beta2) * (grad ** 2)  # Update second moment
        m_hat = m / (1 - beta1 ** t)  # Bias correction for first moment
        v_hat = v / (1 - beta2 ** t)  # Bias correction for second moment
        theta -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)  # Update parameters
    return theta


def rmsprop(theta, grad_func, learning_rate=0.01, beta=0.9, epsilon=1e-8, steps=100):
    grad_squared_avg = np.zeros_like(theta)  # Initialize average of squared gradient
    for _ in range(steps):
        grad = grad_func(theta)
        grad_squared_avg = beta * grad_squared_avg + (1 - beta) * grad**2
        theta -= learning_rate * grad / (np.sqrt(grad_squared_avg) + epsilon)
    return theta


def sgd(theta, grad_func, learning_rate=0.01, steps=100):
    for _ in range(steps):
        grad = grad_func(theta)
        theta -= learning_rate * grad
    return theta