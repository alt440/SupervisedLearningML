# The gradient and the derivative knowledge (Gradient descent)
# So you need to take the derivative of your function to know what is the slope of your function.
# Then, the gradient is found by replacing the variables in your derivative by a coordinate point.
# Say x^2 is your function. Derivative is 2x. If x = -30, then gradient is 2x which is 2* (-30) = -60.

# Where does this theory apply? Say you have that function, x^2, where the point it is at 0 is the point you want
# to reach to minimize the error rate. What you normally do calculate the gradient of a certain coordinate, and then
# do the current value (x = -30) - the gradient (-60), which gives us: -30 - (-60) = 30. Then we restart the process
# from the new x coordinate 30.
# Also, notice that in the function x^2, by starting at -30, you can never reach 0 by subtracting the gradient. So,
# there is something being added called the learning rate which will subtract part of the gradient every time instead
# of the value of the gradient. This value is called the learning rate. So, in the scenario mentioned above, I
# subtracted the gradient from -30 which is: -30 - (-60) = 30. With the learning rate, we do this instead:
# -30 - (learning_rate * (-60)), where learning_rate is a value between 0 and 1. For a learning_rate of 0.1, this
# would mean -30 - (0.1 * -60) = -24.

# Minimizing the function
# 3x^2 + xy + 5y^2
# Partial derivative : deriving for one variable and treating other variables as constants.
# Remember the thing of dy/dx? means that you derive for x and the rest is considered constants, which is a partial derivative.
# so 3x^2 + xy + 5y^2 's partial derivative is dy/dx = 6x + y + C, where C is constant. You keep a y here because it
# is from xy, and the derivative of x is 1, which leaves us with y. However, 5y^2 does not hold any x value, so it
# just ends up being a constant C.
# the partial derivative for y dx/dy = 10y + x + C, where C is 3x^2 because it holds no link to y, x was originally
# xy, but derivative of y leaves us with x, and 5y^2 derivative is 10y.
# So for functions with multiple variables, you must do partial derivatives, and from there calculate the gradient.

if __name__ == '__main__':
    # Function to minimize
    fc = lambda x, y: (3*x**2) + (x*y) + (5*y**2)
    # Set partial derivates
    partial_derivatives_x = lambda x, y: (6*x) + y
    partial_derivatives_y = lambda x, y: (10*y) + x
    # Set variables
    x = 10
    y = -13
    # Learning rate
    learning_rate = 0.1
    # Means "What is the value of this function given x=10 and y=-13?" Result: 1015
    print("Fc = %s" % (fc(x, y)))

    # One epoch is one period of minimization
    for epoch in range(0, 20):
        # Compute gradient (functions call return result)
        x_gradient = partial_derivatives_x(x, y)
        y_gradient = partial_derivatives_y(x, y)
        # Apply gradient descent
        x = x - learning_rate * x_gradient
        y = y - learning_rate * y_gradient
        # Keep track of the function value.
        # Notice that you want to end with an Fc value that is the nearest to 0. For that, you need to adjust your
        # learning rate accordingly.
        print("Fc = %s" % (fc(x,y)))

    # Print final variables values
    print("")
    print("x = %s" % x)
    print("y = %s" % y)