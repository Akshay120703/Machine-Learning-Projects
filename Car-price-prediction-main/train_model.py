import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load the data
car_dataset = pd.read_csv('car data.csv')

# Encode categorical variables
car_dataset.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2},
                     'Seller_Type': {'Dealer': 0, 'Individual': 1},
                     'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)

# Split data into features and target variable
x = car_dataset.drop(['Car_Name', 'Selling_Price'], axis=1)
y = car_dataset['Selling_Price']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, random_state=2)

# Train the linear regression model
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)

# Save the model to a pickle file
with open('lin_reg_model.pkl', 'wb') as file:
    pickle.dump(lin_reg_model, file)
