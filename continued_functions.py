
###################################

import numpy as np
import cmath as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

###################################

color_codes = {1:'red',2:'yellow',3:'blue',4:'green',5:'magenta',6:'crimson',7:'violet',8:'gold',9:'palegreen',10:'orange'
               ,11:'skyblue',12:'purple',13:'aqua',14:'pink',15:'lime',16:'mistyrose',0:'black'}

red = mpatches.Patch(color='red',label='1 lp')
yellow = mpatches.Patch(color='yellow',label='2 lp')
blue = mpatches.Patch(color='blue',label='3 lp')
green = mpatches.Patch(color='green',label='4 lp')
magenta = mpatches.Patch(color='magenta',label='5 lp')
black = mpatches.Patch(color='black',label='undef')

###################################

def f(z,n):
    """
    Return a sequence of n iterations of a continued function for a value z
    """
    pade = [1]
    try:
        for k in range(n):
            
            # Functions
            z_next = cm.exp(pade[k]*z) # exp
            #z_next = cm.log(pade[k]*z) # log
            #z_next = cm.sqrt(pade[k]*z) # sqrt
            #z_next = (pade[k]*z)**(1/3.0) # cube root
            #z_next = z/(1+pade[k]*z) # Control form of this
            #z_next = 1/(1+pade[k]*z) # Control form of this

            # Trigonometric
            #z_next = cm.sin(pade[k]*z) #sine
            #z_next = cm.cos(pade[k]*z) # cosine
            #z_next = cm.tan(pade[k]*z) # tan
            #z_next = cm.asin(pade[k]*z) # arcsine
            #z_next = cm.acos(pade[k]*z) # arccosine
            #z_next = cm.atan(pade[k]*z) # arctan

            # Hyperbolics
            #z_next = cm.sinh(pade[k]*z) #sine 
            #z_next = cm.cosh(pade[k]*z) # cosine
            #z_next = cm.tanh(pade[k]*z) # tan
            #z_next = cm.asinh(pade[k]*z) # arcsine SAME AS ARCSIN
            #z_next = cm.acosh(pade[k]*z) # arccosine
            #z_next = cm.atanh(pade[k]*z) # arctan            

            #Composite
            #z_next = cm.exp(cm.sin(pade[k]*z))
            #z_next = cm.sin(cm.exp(pade[k]*z))
            #z_next = cm.log(cm.sin(pade[k]*z))
            #z_next = cm.sin(cm.log(pade[k]*z))
            #z_next = cm.cos(cm.sin(pade[k]*z))
            
            pade.append(z_next)
        return pade
    except (OverflowError,ValueError):
        return [0]

def test(pade):
    """
    Return the number (color coded) of limit points in a sequence.
    Points within a distance epsilon are indistinguishable.
    """
    epsilon = 1e-10
    try:
        if len(pade) == 1:
            return 'black' #undefined
        if abs(pade[-1]-pade[-2])<epsilon:
            return 'red' #1 limit point
        elif abs(pade[-1]-pade[-3])<epsilon:
            return 'yellow' #2 limit points
        elif abs(pade[-1]-pade[-4])<epsilon:
            return 'blue' #3 limit points
        elif abs(pade[-1]-pade[-5])<epsilon:
            return 'green' #4 limit points
        elif abs(pade[-1]-pade[-6])<epsilon:
            return 'magenta' #5 limit points

        #elif abs(pade[n]-pade[n-6])<epsilon:
        #    return 'crimson' #6
        #elif abs(pade[n]-pade[n-7])<epsilon:
        #    return 'violet' #7
        #elif abs(pade[n]-pade[n-8])<epsilon:
        #    return 'gold' #8
        #elif abs(pade[n]-pade[n-9])<epsilon:
        #    return 'palegreen' #9
        #elif abs(pade[n]-pade[n-10])<epsilon:
        #    return 'orange' #10
        #elif abs(pade[n]-pade[n-11])<epsilon:
        #    return 'skyblue' #11
        #elif abs(pade[n]-pade[n-12])<epsilon:
        #    return 'purple' #12
        #elif abs(pade[n]-pade[n-13])<epsilon:
        #    return 'aqua' #13
        #elif abs(pade[n]-pade[n-14])<epsilon:
        #    return 'pink' #14
        #elif abs(pade[n]-pade[n-15])<epsilon:
        #    return 'lime' #15
        #elif abs(pade[n]-pade[n-16])<epsilon:
        #    return 'mistyrose' #16

        else:
            return 'black' #undefined
    except (OverflowError,ValueError):
        return 'black' #undefined

def scatter(list_lp,no_of_lp):
    """
    Generate a scatter plot for each number of limit points (each color code).
    """
    x = []
    y = []
    assert isinstance(no_of_lp,int),'number of limit points must be an integer number'
    Color = color_codes[no_of_lp] #Use predefined dictionary with color codes
    for k in range(len(list_lp)):
        if list_lp[k][2] == Color:
            x.append(list_lp[k][0])
            y.append(list_lp[k][1])
    fig = plt.figure(1)
    ax = fig.add_subplot(111,axisbg='black')
    return ax.scatter(x,y,s=10,color=Color,marker='.',lw=0,alpha=1)

###################################

# N=1500 and n=600 takes approx 10-20min to generate a pic (this is what's used in the compendium).
# If you use lower N, (N < 1000) make sure to use larger dots (s=10 or larger) in scatter(). Change this manually.
# I.e. for N>1000 I use around s=2 to s=4, for N<1000 I use around s=10 in scatter().
# One can also choose to use a smaller grid (a_Re,b_Re)x(a_Im,b_Im). 

N = 300 #num in linspace ("resolution" in picture) 
n = 300 #number of terms in pade sequence

a_Re = -4.0 #start value Real axis
b_Re = 4.0 #end value Real axis
Re_x = np.linspace(a_Re,b_Re,N)

a_Im = -4.0 #start value Imaginary axis
b_Im = 4.0 #end value Imaginary axis
Im_y = np.linspace(a_Im,b_Im,N)

list_of_limit_points = []
for k in range(N):
    for i in range(N):
        z = complex(Re_x[i],Im_y[k])
        no_of_limit_points = test(f(z,n))
        list_of_limit_points.append([Re_x[i],Im_y[k],no_of_limit_points])

for i in range(6): #no. of limit points included in picture (max 17) - make sure this number match the one in test()
    scatter(list_of_limit_points,i)
plt.xlim(a_Re,b_Re)
plt.ylim(a_Im,b_Im)
plt.xlabel('Re')
plt.ylabel('Im')
#plt.legend(bbox_to_anchor=(1.0,1.0),handles=[red,yellow,blue,green,magenta,black])
plt.show()
