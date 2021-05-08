import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



def main():#my main function
#variable for regression
 x = np.array([31,33,31,49,53,69,101,99,143,132,109])
 y = np.array([705,540,650,840,890,850,1200,1150,1700,900,1550])
 x=x[:,np.newaxis]
 y=y[:,np.newaxis]   



 linear_regression= LinearRegression()#define a variable for linear regression function

 polynomial_regression=PolynomialFeatures(degree=5)#try different degree polynomial regression
 polynomial=polynomial_regression.fit_transform(x)

 linear_regression.fit(polynomial,y) #and fits it

 y_pol=linear_regression.predict(polynomial)
 #and for plot the graph
 plt.plot(x,y_pol)#regression line
 plt.scatter(x,y,marker='*', color='black')#points for graph
 plt.title("Polynomial Regression")
 plt.xlabel("house size (sqm)")
 plt.ylabel("rent")
 plt.show()
 
if __name__ == "__main__": #run the code
    main()