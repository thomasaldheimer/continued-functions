import numpy as np
import cmath as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

color_codes = {1: 'red', 2: 'yellow', 3: 'blue', 4: 'green', 5: 'magenta', 6: 'crimson', 7: 'violet', 8: 'gold',
               9: 'palegreen', 10: 'orange', 11: 'skyblue', 12: 'purple', 13: 'aqua', 14: 'pink', 15: 'lime',
               16: 'mistyrose', 0: 'white'}

patches = {1: mpatches.Patch(color=color_codes[1], label='1 lp'),
           2: mpatches.Patch(color=color_codes[2], label='2 lp'),
           3: mpatches.Patch(color=color_codes[3], label='3 lp'),
           4: mpatches.Patch(color=color_codes[4], label='4 lp'),
           5: mpatches.Patch(color=color_codes[5], label='5 lp'),
           6: mpatches.Patch(color=color_codes[6], label='6 lp'),
           7: mpatches.Patch(color=color_codes[7], label='7 lp'),
           8: mpatches.Patch(color=color_codes[8], label='8 lp'),
           9: mpatches.Patch(color=color_codes[9], label='9 lp'),
           10: mpatches.Patch(color=color_codes[10], label='10 lp'),
           11: mpatches.Patch(color=color_codes[11], label='11 lp'),
           12: mpatches.Patch(color=color_codes[12], label='12 lp'),
           13: mpatches.Patch(color=color_codes[13], label='13 lp'),
           14: mpatches.Patch(color=color_codes[14], label='14 lp'),
           15: mpatches.Patch(color=color_codes[15], label='15 lp'),
           16: mpatches.Patch(color=color_codes[16], label='16 lp'),
           0: mpatches.Patch(color=color_codes[0], label='undef')}

#functions = ['exp', 'log', 'sqrt', 'cube_root', 'sine', 'cosine', 'tan', 'arcsine', 'arccosine', 'arctan', 'sineh',
#             'cosineh', 'tanh', 'arcsineh', 'arccosineh', 'arctanh', 'exp_sin', 'sin_exp', 'log_sin', 'sin_log',
#             'cos_sin', 'fraction_1', 'fraction_2']
functions = ['exp_sin', 'sin_exp', 'log_sin', 'sin_log', 'cos_sin', 'fraction_1', 'fraction_2']


def continued_function(z, n, function):
    """
    Return a sequence of n iterations of a continued function for a value z
    """
    if function == 'exp':
        def f(x):
            return cm.exp(x*z)
    elif function == 'log':
        def f(x):
            return cm.log(x*z)
    elif function == 'sqrt':
        def f(x):
            return cm.sqrt(x*z)
    elif function == 'cube_root':
        def f(x):
            return (x*z)**(1.0/3.0)
    elif function == 'sine':
        def f(x):
            return cm.sin(x*z)
    elif function == 'cosine':
        def f(x):
            return cm.cos(x*z)
    elif function == 'tan':
        def f(x):
            return cm.tan(x*z)
    elif function == 'arcsine':
        def f(x):
            return cm.asin(pade[k]*z)
    elif function == 'arccosine':
        def f(x):
            return cm.acos(pade[k]*z)
    elif function == 'arctan':
        def f(x):
            return cm.atan(pade[k]*z)
    elif function == 'sineh':
        def f(x):
            return cm.sinh(x*z)
    elif function == 'cosineh':
        def f(x):
            return cm.cosh(x*z)
    elif function == 'tanh':
        def f(x):
            return cm.tanh(x*z)
    elif function == 'arcsineh':
        def f(x):
            return cm.asinh(x*z)
    elif function == 'arccosineh':
        def f(x):
            return cm.acosh(x*z)
    elif function == 'arctanh':
        def f(x):
            return cm.atanh(x*z)
    elif function == 'exp_sin':
        def f(x):
            return cm.exp(cm.sin(x*z))
    elif function == 'sin_exp':
        def f(x):
            return cm.sin(cm.exp(x*z))
    elif function == 'log_sin':
        def f(x):
            return cm.log(cm.sin(x*z))
    elif function == 'sin_log':
        def f(x):
            return cm.sin(cm.log(x*z))
    elif function == 'cos_sin':
        def f(x):
            return cm.cos(cm.sin(x*z))
    elif function == 'fraction_1':
        def f(x):
            return 1/(1+x*z)
    elif function == 'fraction_2':
        def f(x):
            return z/(1+x*z)
    else:
        raise ValueError("Incorrect function: {}". format(function))
    pade = [1]
    try:
        for k in range(n):
            z_next = f(pade[k])
            pade.append(z_next)
        return pade
    except (OverflowError, ValueError):
        return [0]


def test_limit_points(sequence, max_number_of_lp):
    """
    Return the number (color coded) of limit points in a sequence.
    Points within a distance epsilon are indistinguishable.
    """
    if len(sequence) == 1:
        return 0  # undefined
    epsilon = 1e-10
    try:
        for i in range(2, max_number_of_lp+2):
            if abs(sequence[-1]-sequence[-i]) < epsilon:
                return i-1
        return 0  # undefined
    except (OverflowError, ValueError):
        return 0  # undefined


def scatter(function, list_lp, max_number_of_lp, a_re, b_re, a_im, b_im, legend):
    """
    Generate a scatter plot for each number of limit points (each color code).
    """
    fig = plt.figure(figsize=(16, 16), dpi=1200)
    ax = fig.add_subplot(111, facecolor=color_codes[0])

    for lp in range(1, max_number_of_lp+1):
        x = []
        y = []
        color = color_codes[lp]  # Use predefined dictionary with color codes
        for k in range(len(list_lp)):
            if list_lp[k][2] == lp:
                x.append(list_lp[k][0])
                y.append(list_lp[k][1])
        ax.scatter(x, y, s=1, color=color, marker=',', lw=0, alpha=1)
        ax.set_aspect('equal', 'box')
    plt.xlim(a_re, b_re)
    plt.ylim(a_im, b_im)
    plt.xlabel('Re')
    plt.ylabel('Im')
    if legend:
        list_of_handles = [patches[i] for i in range(max_number_of_lp+1)]
        plt.legend(bbox_to_anchor=(1.0, 1.0), handles=list_of_handles)
    fig.tight_layout()
    if not os.path.exists('output'):
        os.makedirs('output')
    plt.savefig("output/{}_{}_lp.png".format(function, max_number_of_lp))

    return None


def generate_picture(function, max_number_of_lp):
    assert isinstance(max_number_of_lp, int), "max_number_of_lp must be of type int"
    assert 17 > max_number_of_lp >= 1, "max_number_of_lp must be 17 > max_number_of_lp >= 1"

    N = 1500  # num in linspace ("resolution" in picture)
    n = 600  # number of terms in pade sequence

    a_Re = -5.0  # start value Real axis
    b_Re = 5.0  # end value Real axis
    Re_x = np.linspace(a_Re, b_Re, N)

    a_Im = -5.0  # start value Imaginary axis
    b_Im = 5.0  # end value Imaginary axis
    Im_y = np.linspace(a_Im, b_Im, N)

    list_of_limit_points = []
    for k in range(N):
        for i in range(N):
            z = complex(Re_x[i], Im_y[k])
            no_of_limit_points = test_limit_points(sequence=continued_function(z=z, n=n, function=function),
                                                   max_number_of_lp=max_number_of_lp)
            list_of_limit_points.append([Re_x[i], Im_y[k], no_of_limit_points])

    scatter(function=function,
            list_lp=list_of_limit_points,
            max_number_of_lp=max_number_of_lp,
            a_re=a_Re, b_re=b_Re, a_im=a_Im, b_im=b_Im,
            legend=False)


for func in functions:
    generate_picture(func, 16)
    print('COMPLETE: {}'.format(func))
