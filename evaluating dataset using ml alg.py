import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

dataset_path = 'C:/Users/Lenovo/OneDrive - vitap.ac.in/Desktop/Starbucks Dataset.csv'
df = pd.read_csv(dataset_path, delimiter=',') 

print(df.head())
print("Columns in the dataset:", df.columns)

print(df.isnull().sum())

df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

df = df.drop('Date', axis=1)

target_column = 'Close'  

if target_column in df.columns:
  
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = DecisionTreeRegressor(random_state=42)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R^2 Score:", r2_score(y_test, y_pred))

    plt.figure(figsize=(10, 7))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    plt.show()
else:
    print(f"The target column '{target_column}' was not found in the dataset.")
