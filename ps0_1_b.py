# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 12:54:17 2020

@author: Esin Ayyıldız
"""
#libraries
import matplotlib.pyplot as plt
import numpy as np

def main(): #my main method
 e = [] #error for my array
 x = np.array([31,33,31,49,53,69,101,99,143,132,109])
 y = np.array([705,540,650,840,890,850,1200,1150,1700,900,1550])
 #initial values are zero for m and b current
 m_current =0
 b_current = 0
 lr = 0.000009
 num = len(x)
# plt.scatter(x,y,color='black',marker='*')
 for i in range(100):
     y_predicted = m_current * x + b_current 
     cost = (1/num) * sum([val**2 for val in (y-y_predicted)])#formula of empirical risk

     #plt.plot(x,y_predicted)
     #plt.xlabel('house size (sqm)')
     #plt.title("Linear Regression")
     #plt.ylabel(' rent')
     mderivative = -(2/num)*sum(x*(y-y_predicted))#calculate mderivative
     bderivative = -(2/num)*sum(y-y_predicted)#calculate yderivative
        
        
     m_current = m_current - lr * mderivative #calculate current m value
     b_current = b_current - lr * bderivative #calculate current b value
     print ("iteration = {}, m = {}, b = {}, cost = {} ".format(i,m_current,b_current,cost))
     e.append(cost)
     #plt.scatter(m_current,i,marker='*')
     #plt.xlabel("m current values")
     #plt.ylabel("iterations")
     #plt.scatter(b_current,i,marker='*')
     #plt.xlabel("b current values")
     #plt.ylabel("iterations")

 #plotting the empirical risk as a function of iteration
 plt.plot(np.arange(1, len(e)+1), e)#plot the empirical error
 plt.title("Empirical Risk")
 plt.xlabel("iterations")
 plt.ylabel("empirical error")
 plt.show()
if __name__ == "__main__":
    main()
