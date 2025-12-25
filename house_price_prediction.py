# House Price Prediction using Machine Learning

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample dataset
data = {
    'Area': [800, 1000, 1200, 1500, 1800, 2000],
    'Bedrooms': [1, 2, 2, 3, 3, 4],
    'Price': [40000, 50000, 60000, 75000, 90000, 110000]
}

df = pd.DataFrame(data)

X = df[['Area', 'Bedrooms']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

new_house = [[1600, 3]]
predicted_price = model.predict(new_house)
print("Predicted Price:", predicted_price[0])
