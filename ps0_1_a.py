import matplotlib.pyplot as plt
import numpy as np

def main(): #my main method
#variable for regression
    x = np.array([31,33,31,49,53,69,101,99,143,132,109])
    y = np.array([705,540,650,840,890,850,1200,1150,1700,900,1550])
    x=x[:,np.newaxis]
    y=y[:,np.newaxis]   
    
    # size of the dataset 
    num = np.size(x)
 
    meanof_X = np.mean(x)
    meanof_Y = np.mean(y)
    
    # calculate the cross-deviation 
    cross_y=np.sum(y*x)-(num*meanof_Y*meanof_X)
    cross_x=np.sum(x*x)-(num*meanof_X*meanof_X)

    # calculating regression coefficients
    b1 = cross_y / cross_x
    b0 = meanof_Y - b1*meanof_X

    print("Estimated coefficient values:\n1)  {} \n2)  {}".format(b0, b1))
    
    
    w = np.matmul(np.linalg.pinv(x),y) #calculate of the w value
    print("w  is =" ,w)


    y_prediction = b1*x + b0 # predicted response

    # plotting the points
    plt.scatter(x, y, marker='*' , color='black')
    plt.plot(x, y_prediction) # plot the regression line
    plt.xlabel('house size (sqm)')
    plt.title("Linear Regression")
    plt.ylabel(' rent')
    plt.show()


 
if __name__ == "__main__":
  main()