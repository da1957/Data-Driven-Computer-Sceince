import os
import argparse
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()


filename = sys.argv[1]
xs, ys = load_points_from_file(filename)


def least_squares(x, k):
    # extend the first column with 1s


    ones = np.ones(x.shape)
    x_e = np.column_stack((ones, x))

    for i in range(k - 1):
        power = i + 2
        n = x ** power

        x_e = np.insert(x_e, power, n, axis=-1)

    return x_e



def coef_matrix(x, y):
  if np.linalg.det(x.T.dot(x)) <= 0:
     v = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)
  else:
     v = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)


  return v

def sumColumn(matrix):
    return np.sum(matrix, axis=1)

def least_squares_unknown(x):
    ones = np.ones(x.shape)
    x_u = np.column_stack((ones, np.sin(x)))


    return x_u
#print(least_squares_unknown(xs))

def sum_squared_error_unknown(x,y):
    len_data = len(xs)
    num_segments = len_data // 20

    b = 0
    e = 20
    s = []




    for i in range(num_segments):
        x_s = xs[b:e]
        y_s = ys[b:e]

        x_u = least_squares_unknown(x_s)
        w_u = coef_matrix(x_u,y_s)
        y_h_u = x_u * w_u
        y_hat_u = sumColumn(y_h_u)


        sqeu = np.sum((y_s - y_hat_u) ** 2)
        s.append(sqeu)






        b += 20
        e += 20


    return s



def sum_squared_error_polynomial(x,y,k):

   len_data = len(xs)
   num_segments = len_data // 20

   b = 0
   e = 20
   s = []


   for i in range (num_segments):

        x_s = xs[b:e]
        y_s = ys[b:e]


        x_e = least_squares(x_s,k)
        w = coef_matrix(x_e,y_s)
        yh = x_e * w
        y_hat = sumColumn(yh)

        sqe = np.sum((y_s - y_hat) ** 2)
        s.append(sqe)

        b +=20
        e +=20
   return s





def sum_square_error(y,y_hat):
    return np.sum((y - y_hat)**2)





def total_sum(k1,k2):


    sum = 0


    len_data = len(xs)
    num_segments = len_data // 20

    d = 0
    e = 20

    a = sum_squared_error_polynomial(xs, ys, k1)

    #print("k1", a)
    b = sum_squared_error_polynomial(xs, ys, k2)
    #print("k2",b)

    c = sum_squared_error_unknown(xs, ys)
    #print("u",c)


    for i in range(num_segments):
        x_s = xs[d:e]
        y_s = ys[d:e]




        if a[i] < b[i] and a[i]< c[i] :

            x_e_s = least_squares(x_s,k1)



        elif b[i] < c[i]:
            x_e_s = least_squares(x_s,k2)
        else:
            x_e_s = least_squares_unknown(x_s)
        #print(x_e_s)




        wh = coef_matrix(x_e_s, y_s)
        y_c_h = x_e_s * wh
        y_c = sumColumn(y_c_h)
        sum += sum_square_error(y_s,y_c)




        d+=20
        e+=20
        plt.plot(x_s, y_c, 'r', lw=2)


    arg = len(sys.argv) - 1
    if sys.argv[arg] == "--plot":
        print("total reconstruction error:", sum)
        view_data_segments(xs, ys)

        plt.show()
    else:
        print("total reconstruction error:", sum)
total_sum(k1=1,k2=3)













def cross_validation_u():
    len_data = len(xs)
    num_segments = len_data // 20



    b = 0
    e = 20
    d = []
    fig, ax = plt.subplots()

    ax.scatter(xs, ys, s=200)

    for i in range(num_segments):
        x_s = xs[b:e]
        y_s = ys[b:e]

        x_test = x_s[10:]
        x_train = x_s[:10]
        y_test = y_s[10:]
        y_train = y_s[:10]



        x_u_test = least_squares_unknown(x_test)
        x_u_train = least_squares_unknown(x_train)
        w_u = coef_matrix(x_u_train, y_train)
        y_u_test = x_u_test * w_u
        y_h_u_test = sumColumn(y_u_test)

        cross_error_u = ((y_test - y_h_u_test) ** 2).mean()
        d.append(cross_error_u)
        #print("unknown cross val:", cross_error_u)

        b += 20
        e += 20
    return d





def cross_validation(k):

    len_data = len(xs)
    num_segments = len_data // 20
    c = []


    b = 0
    e = 20


    for i in range(num_segments):
        x_s = xs[b:e]
        y_s = ys[b:e]




        x_test = x_s[10:]
        x_train = x_s[:10]
        y_test = y_s[10:]
        y_train = y_s[:10]


        x_e_test = least_squares(x_test, k)
        x_e_train = least_squares(x_train, k)


        x_s_e = least_squares(x_s, k)
        w1 = coef_matrix(x_s_e, y_s)
        w2 = coef_matrix(x_e_train, y_train)

        y_e_test = x_e_test * w2
        y_a = sumColumn(y_e_test)

        cross_error = ((y_test - y_a) ** 2).mean()
        c.append(cross_error)
        #print("cross validation error:",cross_error)


        y_e = x_s_e * w1
        y_p = sumColumn(y_e)




        #plt.plot(x_s, y_p, 'r', lw=4)
        b +=20
        e +=20



    return c
#print(cross_validation(k=3))
#plt.show()

def find_function(k1,k2):
    f = []
    a = cross_validation(k1)
    #print("k1",a)
    b = cross_validation(k2)
    #print("k3",b)
    c = cross_validation_u()
    #print("u",c)

    for i in range (len(c)):

        if a[i] < b[i] and a[i]< c[i]:
            f.append(a[i])
        elif b[i]<c[i]:
            f.append(b[i])

        else:
            f.append(c[i])
    return f
#print(find_function(k1=1,k2=3))




















