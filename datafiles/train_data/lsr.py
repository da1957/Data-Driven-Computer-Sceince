import re
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from enum import Enum


class Model(Enum):
    LINEAR = 0
    POLYNOMIAL = 1
    UNKNOWN = 2


class Function(Enum):
    EXP = 0
    TAN = 1
    SIN = 2


class Method(Enum):
    K_FOLD = 0
    LEAVE_P_OUT = 1


FILES = ["adv_1.csv", "adv_2.csv", "adv_3.csv",
         "basic_1.csv", "basic_2.csv", "basic_3.csv", "basic_4.csv", "basic_5.csv",
         "noise_1.csv", "noise_2.csv", "noise_3.csv"]


def load_points_from_file(filename):
    """Loads 2d points from a csv called filename"""
    try:
        points = pd.read_csv(filename, header=None)
    except RuntimeError as e:
        points = pd.read_csv("train_data/"+filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys, all_x, all_y, n, title):
    """Visualises the input file with each segment plotted in a different colour."""
    colour = np.concatenate([[i] * 20 for i in range(n)])
    plt.set_cmap('Dark2')
    plt.scatter(all_x, all_y, c=colour)
    for i in range(n):
        plt.plot(xs[i], ys[i], 'r-', lw=1)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def display_best_functions(functions, error_data):
    """Prints the performance data of the top three performing functions."""
    performance = [(x, y, i) for i, [x, y] in enumerate(error_data)]
    performance = sorted(performance, key=lambda x: x[0])
    for i in range(3):
        index = performance[i][2]
        if functions[index].isnumeric():
            if functions[index] == "1":
                print("Linear", end="\t")
            else:
                print("Poly " + functions[index], end="\t")
        else:
            print(functions[index], end="\t")
        print("mean=" + str(performance[i][0]) + "\tstd=" + str(performance[i][1]))


def analyse_performance(function, error_data, filename, segment):
    """Plots a graph of the performance of all functions on the given segment."""
    means = np.log(error_data[:, 0])
    std = np.log(error_data[:, 1])
    display_best_functions(function, error_data)
    fig, ax = plt.subplots()
    ax.scatter(function, std, label="standard deviation", c='g')
    ax.scatter(function, means, label="mean", c='b')
    ax.set_xlabel("function")
    ax.set_ylabel("log error")
    ax.set_title(filename + " Segment " + str(segment))
    ax.legend()
    plt.show()


def least_squares(x, y):
    """Returns the parameters with maximum likelihood."""
    w_hat = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)
    return w_hat


def extract_data(filename):
    """Splits the data into its line segments."""
    all_x, all_y = load_points_from_file(filename)
    n = len(all_x) // 20
    xs = np.array([[all_x[j] for j in range(i*20, i*20+20)] for i in range(n)])
    ys = np.array([[all_y[j] for j in range(i*20, i*20+20)] for i in range(n)])
    return xs, ys, all_x, all_y, n


def construct_linear_feature(x):
    """Returns the linear feature vector for the given data."""
    ones = np.ones(x.shape)
    big_x = np.column_stack((ones, x))
    return big_x


def construct_polynomial_feature(x, order):
    """Returns the polynomial feature vector of the specified order for the given data."""
    terms = tuple(x**i for i in range(order+1))
    big_x = np.column_stack(terms)
    return big_x


def construct_unknown_feature(x, function):
    """Returns the unknown feature vector of the specified function for the given data."""
    ones = np.ones(x.shape)
    if function == Function.EXP:
        big_x = np.column_stack((ones, np.exp(x)))
    elif function == Function.TAN:
        big_x = np.column_stack((ones, np.tan(x)))
    elif function == Function.SIN:
        big_x = np.column_stack((ones, np.sin(x)))
    else:
        big_x = np.column_stack((ones, x))
    return big_x


def construct_feature(xs, model, order=2, function=Function.EXP):
    """Returns the feature vector of the given data as specified."""
    if model == Model.POLYNOMIAL:
        big_x = construct_polynomial_feature(xs, order)
    elif model == Model.UNKNOWN:
        big_x = construct_unknown_feature(xs, function)
    else:
        big_x = construct_linear_feature(xs)
    return big_x


def sum_squared_error(ys, feature, params):
    """Returns the sum-squared error of the given output points from the fitted line."""
    fitted_ys = np.matmul(feature, params)
    error = np.sum((fitted_ys - ys) ** 2)
    return error


def k_fold(n, k):
    """Randomly splits the data indices into k groups."""
    indices = np.array([i for i in range(n)])
    np.random.shuffle(indices)
    partitions = np.array([indices[i::k] for i in range(k)])
    return partitions


def add_to_perm(perm, n, p):
    """Recursively adds values to a permutation."""
    if p == 0:
        return [perm]
    perms = []
    # print(perm)
    for i in range(perm[-1] + 1, n):
        new_perm = perm.copy()
        new_perm.append(i)
        # print(new_perm)
        new_perms = add_to_perm(new_perm, n, p-1)
        perms += new_perms
    return perms


def leave_p_out(n, p):
    """Returns all permutations of size p of indices from 0 to n-1."""
    permutations = []
    for i in range(n-p+1):
        perm = [i]
        perms = add_to_perm(perm, n, p-1)
        permutations += perms
    return np.array(permutations)


def partition_data(xs, ys, partition):
    """Splits the data into a test set (as specified by the partition) and a training set."""
    n = len(xs)
    x_test = np.array([xs[i] for i in range(n) if i in partition])
    x_train = np.array([xs[i] for i in range(n) if i not in partition])
    y_test = np.array([ys[i] for i in range(n) if i in partition])
    y_train = np.array([ys[i] for i in range(n) if i not in partition])
    return x_test, x_train, y_test, y_train


def cross_validate(xs, ys, partitions, model, order=2, function=Function.EXP):
    """Trains the model and cross-validates, returning the mean and standard deviation of errors."""
    errors = []
    for partition in partitions:
        x_test, x_train, y_test, y_train = partition_data(xs, ys, partition)
        big_x = construct_feature(x_train, model, order, function)
        w_hat = least_squares(big_x, y_train)
        big_x_test = construct_feature(x_test, model, order, function)
        error = sum_squared_error(y_test, big_x_test, w_hat)
        errors.append(error)
    mean = np.mean(errors)
    std = np.nanstd(errors)
    return mean, std


def evaluate_functions(xs, ys, polynomials, functions, method=Method.K_FOLD):
    """Uses k-fold cross-validation on different functions.
    Returns the performance data of each function."""
    if method == Method.K_FOLD:
        partitions = k_fold(len(xs), 10)
    else:
        partitions = leave_p_out(20, 2)
    names = []
    error_data = []
    for i in polynomials:
        m, s = cross_validate(xs, ys, partitions, Model.POLYNOMIAL, order=i)
        names.append(str(i))
        error_data.append([m, s])
    for function in functions:
        m, s = cross_validate(xs, ys, partitions, Model.UNKNOWN, function=function)
        names.append(function.name)
        error_data.append([m, s])
    return np.array(names), np.array(error_data)


def contains_half_pi():
    """Returns true if a file contains an x value that is a multiple of π/2 or -π/2 not including 0."""
    for filename in FILES:
        _, _, all_x, _, _ = extract_data(filename)
        multiples_of_pi = [x for x in all_x if x != 0 and (x % (np.sign(x) * np.pi / 2)) == 0]
        if len(multiples_of_pi) > 0:
            return True
    return False


def display_function_data(means, stds):
    """Plots the mean and std of the errors of each model across all files."""
    mean_logs = np.log(means)
    std_logs = np.log(stds)
    function = [str(i) for i in range(1, 21)] + ["EXP", "TAN", "SIN"]
    fig, ax = plt.subplots()
    ax.scatter(function, std_logs, label="standard deviation", c='g')
    ax.scatter(function, mean_logs, label="mean", c='b')
    ax.set_xlabel("function")
    ax.set_ylabel("total log error")
    ax.legend()
    plt.show()


def display_raw_data(xs, ys, file, segment):
    """Plots the original data points."""
    fig, ax = plt.subplots()
    ax.scatter(xs, ys)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(file + " Segment " + segment)
    plt.show()


def analyse_functions():
    """Performs analysis on data collected on the performance of all
    models on all files. It then makes a call to display the means and
    standard deviations of each model's overall performance."""
    means = np.array([.0 for _ in range(23)])
    stds = np.array([.0 for _ in range(23)])
    for file in FILES:
        x_segments, y_segments, all_x, all_y, n = extract_data(file)
        for i in range(n):
            polys = [i for i in range(1, 21)]
            funcs = [f for f in Function]
            functions, error_data = evaluate_functions(x_segments[i], y_segments[i], polys, funcs)
            for j in range(len(functions)):
                means[j] += error_data[j][0]
                stds[j] += error_data[j][1]
    display_function_data(means, stds)


def choose_best(functions, error_data):
    """Chooses the model with lowest mean error."""
    index = min([0, 1, 2], key=lambda x: error_data[x][0])
    if functions[index] == "1":
        return Model.LINEAR
    elif functions[index].isnumeric():
        return Model.POLYNOMIAL
    else:
        return Model.UNKNOWN


def main(argv):
    """
    --pi-hunt searches all files for multiples of π/2 in the x data, returning true if found
    --data=k plots the raw input data for segment k (with k ≥ 0)
    --plot chooses between the 3 functions, fits and plots the curve for all segments
    --plot=k does the same but only for the kth segment (with k ≥ 0)
    --func=k forces the specified function to be used for fitting (with 0 ≤ k ≤ 20 or k in {sin, tan, exp}).
             Use with --plot(=k)
    --eval=k evaluates the performance of all models on the kth segment (with k ≥ 0)
    --analyse evaluates the performance of all functions across all files and displays a graph summarising the results
    """
    # extract command line arguments and options
    opts = [arg for arg in argv if arg.startswith("--")]
    args = [arg for arg in argv if not arg.startswith("--")]
    evals = [opt for opt in opts if re.search("^--eval=", opt)]
    data = [opt for opt in opts if re.search("^--data=", opt)]
    plots = [opt for opt in opts if re.search("^--plot=", opt)]
    funcs = [opt for opt in opts if re.search("^--func=", opt)]
    segment = -1
    forced_model = -1
    if len(evals) > 0:
        _, seg = evals[0].split("=")
        segment = int(seg)
    if len(data) > 0:
        _, seg = data[0].split("=")
        segment = int(seg)
    if len(plots) > 0:
        _, seg = plots[0].split("=")
        segment = int(seg)
    if len(funcs) > 0:
        _, func = funcs[0].split("=")
        forced_model = func
    if "--analyse" in opts:
        analyse_functions()
        sys.exit(0)
    if "--pi-hunt" in opts:
        print(contains_half_pi())
        sys.exit(0)
    if len(args) > 0:
        # extract data from file, deal with any opts as necessary and initialise variables
        filename = args[0]
        x_segments, y_segments, all_x, all_y, n = extract_data(filename)
        if (segment != -1) and not (0 <= segment < n):
            sys.exit("segment value out of range")
        params = []
        total_error = 0
        gen_y_segments = []
        xs_plots = []
        ys_plots = []
        order = 3
        function = Function.SIN
        for i in range(n):
            # evaluate the performance of different functions on the chosen segment
            if i == segment:
                if len(evals) > 0:
                    polys = [i for i in range(1, 11)]
                    funcs = [f for f in Function]
                    functions, error_data = evaluate_functions(x_segments[i], y_segments[i], polys, funcs,
                                                               method=Method.K_FOLD)
                    analyse_performance(functions, error_data, filename, i)
                    sys.exit(0)
                elif len(data) > 0:
                    display_raw_data(x_segments[i], y_segments[i], argv[0], str(i))
                    sys.exit(0)
                elif len(plots) == 0:
                    print("Command line options error")
                    sys.exit(0)
                else:
                    pass
            # determine which of the three models to fit to the data
            functions, error_data = evaluate_functions(x_segments[i], y_segments[i], [1, 3], [Function.SIN],
                                                       method=Method.LEAVE_P_OUT)
            if forced_model == -1:
                model = choose_best(functions, error_data)
            else:
                if forced_model.isnumeric():
                    if forced_model == "1":
                        model = Model.LINEAR
                    else:
                        model = Model.POLYNOMIAL
                        order = int(forced_model)
                else:
                    model = Model.UNKNOWN
                    function = Function[forced_model]
            # fit the model to the data
            big_x = construct_feature(x_segments[i], model, order=order, function=function)
            w_hat = least_squares(big_x, y_segments[i])
            gen_ys = np.matmul(big_x, w_hat)
            gen_y_segments.append(gen_ys)
            # generate data for plotting
            start = min(x_segments[i])
            stop = max(x_segments[i])
            xs_plot = np.array([start + (stop - start) * i/100 for i in range(100)])
            big_x_plot = construct_feature(xs_plot, model, order=order, function=function)
            ys_plot = np.matmul(big_x_plot, w_hat)
            xs_plots.append(xs_plot)
            ys_plots.append(ys_plot)
            # find the sum squared error
            params.append(w_hat)
            error = sum_squared_error(y_segments[i], big_x, params[i])
            if len(plots) > 0 and i == segment:
                print(error)
                break
            total_error += error
        if len(plots) == 0:
            print(total_error)
        if len(plots) > 0:
            start = segment * 20
            end = segment * 20 + 20
            title = argv[0] + " Segment " + str(segment)
            view_data_segments([xs_plots[segment]], [ys_plots[segment]], all_x[start:end], all_y[start:end], 1, title)
        if "--plot" in opts:
            view_data_segments(xs_plots, ys_plots, all_x, all_y, n, argv[0])


if __name__ == "__main__":
    main(sys.argv[1:])

