import numpy as np

class LinearRegression2D:
    """
    Implements a simple 2D linear regression using NumPy.
    
    This class is useful for applying a linear regression model to a dataset
    where you want to predict an output variable `y` (dependent) based on an 
    input variable `x` (independent) according to the relationship:
    
        y = a * x + b
    
    where `a` represents the slope (coefficient) and `b` is the intercept.
    
    Usage:
        - Train the model on input and output data.
        - Use the model to make predictions on new data.
    
    Example:
        model = LinearRegression2D()
        model.fit(x, y)
        predictions = model.predict(new_x)
    """

    def __init__(self):
        self.a = None  # Slope coefficient
        self.b = None  # Intercept

    def fit(self, x, y):
        """
        Calculates the coefficients `a` and `b` of the linear regression model 
        using the least squares method.

        Parameters:
            x (np.array): 1D array of independent variable values.
            y (np.array): 1D array of dependent variable values.
        """
        # Add a column of ones to account for the intercept term
        X = np.vstack([x, np.ones(len(x))]).T
        
        # Calculate regression coefficients (a, b) using the least squares method
        coefficients = np.linalg.inv(X.T @ X) @ X.T @ y
        
        # Assign coefficients
        self.a, self.b = coefficients

    def predict(self, x):
        """
        Predicts `y` values for an array of `x` values using the trained model.

        Parameters:
            x (np.array): 1D array of independent variable values.

        Returns:
            np.array: Array of predicted `y` values.
        """
        # Check if the model has been trained
        if self.a is None or self.b is None:
            raise ValueError("The model has not been trained. Call the `fit` method before making predictions.")
        
        # Calculate predicted `y` values
        return self.a * x + self.b

    def coefficients(self):
        """
        Returns the coefficients `a` and `b` of the model.

        Returns:
            tuple: A tuple (a, b) where `a` is the slope and `b` is the intercept.
        """
        return self.a, self.b







if __name__ == "__main__":
    print("Example usage:")

    # Sample data
    x2 = np.array([1, 2, 3, 4, 5])
    y2 = np.array([2, 4, 5, 4, 5])

    # Create an instance of the linear regression model
    model = LinearRegression2D()

    # Train the model
    model.fit(x2, y2)

    # Get the coefficients
    a, b = model.coefficients()
    print("Slope (a):", a)
    print("Intercept (b):", b)

    # Make predictions on new data
    y_pred = model.predict(x2)
    print("Predictions y:", y_pred)