#Please note research on how to fit a polynomial line was conducted at: https://www.w3schools.com/python/python_ml_polynomial_regression.asp
#Data was taken from https://finance.yahoo.com/quote/AAPL/

# Importing the various libraries.
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import datetime

#Importing an excel csv file into the polynomial regression model about the AAPL stock price data. 
AAPL_data = pd.read_csv('AAPL_data.csv')

#Converting the closing price to float and stripping the $ sign from the closing price.
close_all = [float(d.strip('$')) for d in AAPL_data['Close/Last']]

#Converting the volume and date to arrays.
y = np.asarray(AAPL_data['Volume'])
x = np.asarray(close_all)

#Polynomial model with factor of 5 and fitting the data to the polynomial model. 
mymodel = np.poly1d(np.polyfit(x, y, 5))
myline = np.linspace(1, 175, 100)

#Plotting the volume and the closing price.
plt.plot(x,y)

#An x-axis label for the date.
plt.xlabel("Closing Price ($)")

#A y-axis for the Closing Price.
plt.ylabel("Volume (miliions)")

#A title for the data.
plt.title("Stock Volume Prediction: AAPL Closing Price")

#Plotting the polynomial regression model of factor 5.
plt.plot(myline,mymodel(myline))

#Plotting the legend. 
plt.legend(["Actual Data","Polynomial Regression Line"])

#Displaying the data. 
plt.show()

