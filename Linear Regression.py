import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
 
data =\
 [
 [1, 76],
 [2, 62],
 [3, 77],
 [4, 84.5],
 [5, 77.5],
 [6, 78],
 [7, 64],
 [8, 63],
 [9, 64],
 [10, 62.5],
 [11, 59],
 ]

xlabel = np.array(data)[:,0]
xlabel=np.append(xlabel, [[12],[13],[14]])
x = np.array(data)[:,0].reshape(-1,1)   
y = np.array(data)[:,1].reshape(-1,1)
to_predict_x= [12,13,14]
to_predict_x= np.array(to_predict_x).reshape(-1,1)
regsr=LinearRegression()
regsr.fit(x,y)
predicted_y= regsr.predict(to_predict_x)
m= regsr.coef_
c= regsr.intercept_
print("Predicted :\n",predicted_y)
plt.title('Rate of depression during the last 11 months in Iran')  
plt.xlabel('Month')  
plt.ylabel('Depresion rate') 
labels = np.array(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov','dec','jan','feb'])
plt.scatter(x,y, color="blue")
plt.xticks(xlabel, labels)
new_y=[ m*i+c for i in np.append(x,to_predict_x)]
new_y=np.array(new_y).reshape(-1,1)
plt.plot([12,13,14],predicted_y, marker="o", color="red")
plt.plot(np.append(x,to_predict_x),new_y,color="red")
plt.show()

