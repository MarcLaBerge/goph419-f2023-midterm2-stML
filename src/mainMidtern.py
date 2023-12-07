import numpy as np
import matplotlib.pyplot as plt


from linalg_interp import spline_function
from linalg_interp import build
from linalg_interp import norm_sys
from linalg_interp import jacobi
from linalg_interp import check_dom
from linalg_interp import solve_spline_coefs
from linalg_interp import gen_spline_func



def main ():


    carbon = np.loadtxt(".\\data\\question1data.txt")
    print("Opening data")
    yearFocus = 2015.25
    year = carbon[:,0] #the first column
    co2Numbers = carbon[:,1] #second column

    #Build the systems
    print("Building systems")
    A, B = build(year, co2Numbers)

    #Checking if we have satisfied the conditions
    dia_dom = check_dom(A)
    if dia_dom == True:
        print("system passes diagonal dominance, will converge")
    else:
        print("Does not pass, convergence not guaranteed")
    
    #Normalize the system
    a,b = norm_sys(A,B)

    #Solve system
    x = jacobi(a,b)

    #spline coefficients
    a,b,d = solve_spline_coefs(year, co2Numbers, x)

    #create spline function
    spl = gen_spline_func(a,b,x,d,yearFocus,year)

    #Calculating
    co2Mar = spl(yearFocus)

    splCubic = spline_function(year, co2Numbers)
    x_vector = np.linspace(year[0],year[-1],100)
    cubic_plot = np.array([splCubic(yr) for yr in x_vector])
    

    #plotting
    plt.subplot(1,1,1)
    plt.plot(x_vector,cubic_plot,'--b',label="Interpolated splines")
    plt.plot(year,co2Numbers,'xr',label="Raw Data")
    plt.plot(yearFocus,co2Mar,'xg',label="CO2 in March 2015")
    plt.ylabel("CO2 Concentration [micromol/mol]")
    plt.xlabel("Year")
    plt.legend()
    plt.savefig(".\\figures\\co2_splines.png")
    plt.show()

if __name__ == "__main__":
    main()

