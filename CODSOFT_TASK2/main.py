import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r'C:\Users\sidda\Downloads\archive (1)\IMDb Movies India.csv', encoding='cp1252')

# Inspect the data
print(df.head())
print(df.info())
print(df.describe())

# Handle missing values
df = df.dropna()  # or use imputation methods if necessary

# Feature extraction
# Convert genres to dummy variables
genres = df['Genre'].str.get_dummies(sep=',')  # Update column name to 'Genre'
df = pd.concat([df, genres], axis=1)

# Encode categorical features
le_director = LabelEncoder()
df['director_encoded'] = le_director.fit_transform(df['Director'])

# Encode actors (considering there are 3 separate actor columns)
le_actor1 = LabelEncoder()
df['actor1_encoded'] = le_actor1.fit_transform(df['Actor 1'])

le_actor2 = LabelEncoder()
df['actor2_encoded'] = le_actor2.fit_transform(df['Actor 2'])

le_actor3 = LabelEncoder()
df['actor3_encoded'] = le_actor3.fit_transform(df['Actor 3'])

# Select features and target variable
features = ['director_encoded', 'actor1_encoded', 'actor2_encoded', 'actor3_encoded'] + list(genres.columns)
X = df[features]
y = df['Rating']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate a Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression - Mean Squared Error: {mse_lr}")
print(f"Linear Regression - R^2 Score: {r2_lr}")

# Train and evaluate a Random Forest Regressor model
rf_model = RandomForestRegressor()
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest - Mean Squared Error: {mse_rf}")
print(f"Random Forest - R^2 Score: {r2_rf}")

# Save the Random Forest model
joblib.dump(rf_model, 'movie_rating_model.pkl')

# Load and use the model
model = joblib.load('movie_rating_model.pkl')

# Feature importance
importances = rf_model.feature_importances_
feature_names = X.columns

for name, importance in zip(feature_names, importances):
    print(f"Feature: {name}, Importance: {importance}")

# Visualize predictions
plt.scatter(y_test, y_pred_rf)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Actual vs Predicted Ratings')
plt.show()
 
