# Housing-Price-Prediction


## Imported all the required library
```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
```
### Loading and Viewing the data

~~~
data=pd.read_csv('train.csv')
data.head()
~~~
# Data Visualisation

### Ploting the Heatmap

![alt Survived](https://github.com/rahuljadli/Housing-Price-Prediction/blob/master/screen_shots/heatmap.png)

### Ploting Histogram of Saleprice

![alt histogram](https://github.com/rahuljadli/House-Price-Prediction-Advanced-Regression/blob/master/screen_shots/Saleprice%20histogram.png)

### Ploting effect of 2nd floor on house price

![alt 2nd floor ](https://github.com/rahuljadli/House-Price-Prediction-Advanced-Regression/blob/master/screen_shots/Effect%20of%202nd%20floor%20on%20saleprice.png)

### Ploting effect of Lot Area on Sale Price

![alt lot area ](https://github.com/rahuljadli/House-Price-Prediction-Advanced-Regression/blob/master/screen_shots/Effect%20of%20lot%20on%20saleprice.png)

### Ploting Basement area on house price

![alt basement area ](https://github.com/rahuljadli/House-Price-Prediction-Advanced-Regression/blob/master/screen_shots/Effect%20of%20basement%20area%20on%20saleprice.png)

### Ploting effect of MasVnr Area on Sale Price

![alt Masvnr Area ](https://github.com/rahuljadli/House-Price-Prediction-Advanced-Regression/blob/master/screen_shots/Effect%20of%20lot%20on%20saleprice.png)

### Ploting effect of Garage Year on Sale Price

![alt Garage year ](https://github.com/rahuljadli/House-Price-Prediction-Advanced-Regression/blob/master/screen_shots/Effect%20of%20garage%20year.png)

### Ploting effect of Year Renowed on Sale Price

![alt Garage year ](https://github.com/rahuljadli/House-Price-Prediction-Advanced-Regression/blob/master/screen_shots/Effect%20of%20year.png)

### Ploting effect of 1st floor on Sale Price

![alt 1st floor ](https://github.com/rahuljadli/House-Price-Prediction-Advanced-Regression/blob/master/screen_shots/effect%20of%201st%20floor.png)

### Ploting effect of  Year Build on Sale Price

![alt  Year Build ](https://github.com/rahuljadli/House-Price-Prediction-Advanced-Regression/blob/master/screen_shots/effect%20of%20year%20build.png)

### Ploting effect of  Basement area on Sale Price

![alt Basement area ](https://github.com/rahuljadli/House-Price-Prediction-Advanced-Regression/blob/master/screen_shots/efffec%20of%20bsmt%20area.png)

### Ploting effect of  Grive area on Sale Price

![alt Grive area ](https://github.com/rahuljadli/House-Price-Prediction-Advanced-Regression/blob/master/screen_shots/effect%20of%20grive.png)

### Ploting effect of  Garage area on Sale Price

![alt Garage area ](https://github.com/rahuljadli/House-Price-Prediction-Advanced-Regression/blob/master/screen_shots/effect%20of%20garage%20area.png)

### Ploting effect of  Cars on Sale Price

![alt Cars ](https://github.com/rahuljadli/House-Price-Prediction-Advanced-Regression/blob/master/screen_shots/garagecar.png)

### Ploting effect of  Total rooms on Sale Price

![alt Total rooms](https://github.com/rahuljadli/House-Price-Prediction-Advanced-Regression/blob/master/screen_shots/Total%20rooms.png)

### Ploting effect of  Overall quality on Sale Price

![alt Overall quality ](https://github.com/rahuljadli/House-Price-Prediction-Advanced-Regression/blob/master/screen_shots/overall%20quality.png)

### Ploting effect of  Fireplaces on Sale Price

![alt Cars ](https://github.com/rahuljadli/House-Price-Prediction-Advanced-Regression/blob/master/screen_shots/Fireplaces.png)

## Converting Categorical data into Continous using Label Encoder

~~~
df.GarageCond=le.fit_transform(df.GarageCond)
df.GarageYrBlt=le.fit_transform(df.GarageYrBlt)
df.GarageFinish=le.fit_transform(df.GarageFinish)
~~~

# Using Different Model's 

## Creating Training and Testing Data set

~~~
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

~~~
# Training the model

~~~
model=LogisticRegression()
model.fit(x_train,y_train)
~~~
# Making the prediction

~~~
new_prediction=model.predict(testing_data)
~~~

## Getting the accuracy score

~~~
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(prediction, y_test))
rmse
~~~
## Got RMSE value of 69140.009
# Here only shown one algorithm in the notebook all other algorithm are used there. 
