import numpy as np
import matplotlib.pyplot as plt

# https://www.youtube.com/watch?v=J_twHo5k1K8&list=PLpEPgC7cUJ4b1ARx8PyIQa_sdZRL2GXw5&index=1

# this approach of supervised learning functions using this type of equation:
# score = w1 * x1 + w2 * x2 + b
# Where b is the bias, w1 and w2 are weights set by the machine learning algorithm, and x1 and x2 are the inputs received

# then we normalize the score by giving it a value between 0 and 1 so that we can determine whether the patient is
# sick or healthy.

# Also note that the function above for the score describes a linear relationship. In a more complex relationship,
# this formula will change.

def create_dataset():

    """
    Generates the dataset that will be used for data.
    Note that there is two categories of data: sick and healthy people
    :return: Some data containing concentration of red and white blood cell for nb_subjects_per_category people, as well
    as the normalized value target that we want to have.
    """
    nb_subjects_per_category = 100

    # Generate random data using numpy
    # Two values are: Concentration of red blood cell and concentration of white blood cell
    # Generates two values and add the corresponding value with -2. Sick people get score lower than 0
    sick = np.random.randn( nb_subjects_per_category, 2) + np.array([-2,-2])
    # Generates two values and add the corresponding value with 2. Healthy people get score higher than 0
    healthy = np.random.randn( nb_subjects_per_category, 2) + np.array([2, 2])

    # combines the two arrays
    full_data = np.vstack([sick, healthy])

    # means that those sick people get a value of zero, and those healthy get a value of 1.
    # this gives an array of 10 composed of 5 0s followed by 5 1s.
    targets = np.concatenate((np.zeros(nb_subjects_per_category), np.zeros(nb_subjects_per_category) + 1))

    # Plot points. This is the data set being shown in a graph.
    # features[:, 0] means that we are slicing our 2D features of shape 100,2 and taking only the first column of all data
    # features[:, 1] means that we are slicing our array by taking only the second column of our data points
    # s: is marker size (draws bigger points)
    # c: describes the possible colors. Because our targets are 0s and 1s, then there is only two colors. Also, targets
    # array shows how to color the different elements in full_data depending on the index of targets. So I know the 50
    # last elements in full_data will have their own color because the last 50 items in targets all hold same value.
    plt.scatter(full_data[:, 0], full_data[:, 1], s=40, c=targets, cmap=plt.cm.Spectral)
    # save picture of data points drawn.
    plt.savefig("DataPoints.png")

    # can return multiple parameters at once
    return full_data, targets

def init_variables():
    """
    Initialize the weights and the bias
    :return: w1, w2, and b values from the formula w1 * x1 + w2 * x2 + b = score
    """
    weights = np.random.normal(size=2)
    bias = 0
    return weights, bias

def pre_activation(features, weights, bias):
    """
    Compute the score given the parameters. features stand for the data, weights stand for the w1 and w2 in the
    linear relationship formula, and the bias stands for b in the linear relationship formula.
    :param features: your dataset as np.array
    :param weights: the array of weights (there is 2 here) as np.array
    :param bias: an integer value
    :return: score before being normalized
    """
    # this is a dot product between features and weights, added to bias after.
    return np.dot(features, weights) + bias

def activation(z):
    """
    This method normalizes the score between 0 and 1. For some reason, with np you can pass an array and it gives
    you a normalized score for each of the inputs...
    :param z: score not normalized between 0 and 1
    :return: normalized score between 0 and 1
    """
    # formula for sigmoid
    return 1 / (1 + np.exp(-z))

def derivative_activation(z):
    """
    This method is applying the derivative of the sigmoid formula applied in the activation method
    :param z: the unnormalized score
    :return: the derivative value based on that score
    """
    return activation(z) * (1 - activation(z))

def train(features, targets, weights, bias):
    """
    This algorithm trains our data set to better classify the information.
    :param features: Our created data set.
    :param targets: What we want our algorithm to reach (list of 0s and 1s describing what we want our normalized scores
    to reach)
    :param weights: The weights (np.array of size 2) from the linear formula.
    :param bias: The bias (integer) from the linear formula
    :return: Best weights/ bias to separate the healthy from the sick
    """
    # see gradient_descent for explanation
    epochs = 100
    learning_rate = 0.1

    picture_nb = 2

    # Print current accuracy. How many people have been classified as sick/healthy correctly?
    predictions = predict(features, weights, bias)
    print("Accuracy: ", np.mean(predictions == targets))

    for epoch in range(epochs):
        if epoch % 10 == 0:
            # get normalized scores
            predictions = activation(pre_activation(features, weights, bias))
            # compare with targets to see how bad our algorithm is
            print("Cost = %s" % cost(predictions, targets))
            # Replot graph. Check in create_dataset for explanation of parameters
            if picture_nb == 2:
                plt.plot(features[:, 0], (weights[0] * features[:, 0] + bias) / -weights[1], color='red')
            elif picture_nb == 11:
                plt.plot(features[:, 0], (weights[0] * features[:, 0] + bias) / -weights[1], color='green')
            else:
                plt.plot(features[:, 0], (weights[0] * features[:, 0] + bias) / -weights[1], color='orange')
            picture_nb+=1

        # Initialize gradients
        # weights_gradients is 2D array with 2 values
        weights_gradients = np.zeros(weights.shape)
        bias_gradient = 0
        # Go through each row
        for feature, target in zip(features, targets):
            # Compute prediction
            z = pre_activation(feature, weights, bias)
            # Get normalized score
            y = activation(z)
            # Update gradients based on formulas established before. Look at gradient_descent to understand what we
            # are doing. Also, the formulas are below, just before the call of the function train.
            weights_gradients += (y - target) * derivative_activation(z) * feature
            # no multiplication of feature because it does not depend on some coordinates.
            bias_gradient += (y - target) * derivative_activation(z)

        # Update variables. These are the lines that result the cost to get reduced.
        weights = weights - learning_rate * weights_gradients
        bias = bias - learning_rate * bias_gradient

    # Print final accuracy. How many people have been classified as sick/healthy correctly?
    predictions = predict(features, weights, bias)
    print("Accuracy: ", np.mean(predictions == targets))

    plt.scatter(features[:, 0], features[:, 1], s=40, c=targets, cmap=plt.cm.Spectral)
    plt.savefig("DataPointsLineEvolution.png")
    # legend for understanding
    plt.legend(['Original division', 'New division', 'New division', 'New division', 'New division', 'New division',
                'New division', 'New division', 'New division', 'Final division'], loc='upper left')
    # save picture of data points drawn.
    plt.savefig("DataPointsLineEvolutionLegend.png")

def predict(features, weights, bias):
    """
    Gives the value that our ML algorithm returns based on the weights and bias we have set.
    :param features: Our data set
    :param weights: The weights we have established (np.array of size 2) from the linear formula
    :param bias: The bias we have established (integer) from the linear formula
    :return: The prediction (whether the patient is sick or healthy) as a 1 or 0.
    """
    z = pre_activation(features, weights, bias)
    # Get normalized scores
    y = activation(z)
    # Get 0 or 1 value
    return np.round(y)

def cost(predictions, targets):
    """
    This function describes the error rate we have with our data set at classifying the information correctly.
    In other words, how bad is our algorithm? How bad did it do at classifying whether people are healthy or sick?
    :param features: Our data set
    :param weights: The weights from the linear formula (np.array of size 2)
    :param bias: The bias from the linear formula (integer)
    :return: The error rate of our algorithm
    """
    # averages the error across all data points, taking the values that have not been rounded to 0 and 1.
    return np.mean( (predictions - targets)**2)


# this stands as the main method of this class. If this class is not the one being run, then this code will not execute.
if __name__ == '__main__':
    # assigns both returned variables to their corresponding variables
    data, targets = create_dataset()
    # the values that are being set by the ML algorithm
    weights, bias = init_variables()
    # Compute the score of each of our data set inputs (known as pre activation from the tutorial)
    z = pre_activation(data, weights, bias)
    # Compute the normalized score of each of our data set inputs (known as activation from the tutorial)
    a = activation(z)
    # Now here I print the normalized values of the array. Notice that our target values (the ones we must reach)
    # are very different from the values we currently have
    print("Values we must reach:\n")
    print(targets)
    print("Values we currently have:\n")
    print(a)

    # This is where comes the machine learning algorithm that will find the proper weights and bias that can associate
    # correctly all of the data we have. Notice that from the targets and a matrices, there are some values that are
    # not associated to the right category. If the value is equal or over 0.5, then our score means our patient is
    # healthy. Under 0.5 means our patient is unhealthy.

    # Now to understand what is going to be done below, make sure you look into the Gradient_descent.py file and read
    # the notes there.
    # t = target (the value we want)
    # y = value received
    # So our error rate function is E = 1/2 * (y - t)^2. Remember the formula established at the beginning of this file:
    # score = w1 * x1 + w2 * x2 + b
    # we will do partial derivatives for each of the values set by our machine.
    # a'(z) = derivative of the normalized score
    # t = target (the value we want)
    # y = value received
    # This results in:
    # dE/ dw1 = (y - t) * a'(z) * x1
    # dE/ dw2 = (y - t) * a'(z) * x2
    # dE/ db = (y - t) * a'(z)
    train(data, targets, weights, bias)