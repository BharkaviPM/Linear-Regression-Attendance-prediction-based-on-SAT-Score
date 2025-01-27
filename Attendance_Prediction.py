import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

file_path = r"C:\Users\USER\Downloads\Problem 5.csv"
df = pd.read_csv(file_path)
df['Attendance'] = df['Attendance'].map({'No': 0, 'Yes': 1})
correlation_matrix = df.corr()
print("Correlation Matrix:\n", correlation_matrix)
plt.figure(figsize=(12, 5))


# Split data for prediction - using both GPA and SAT
X = df[['GPA', 'SAT']]
y = df['Attendance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

slope_gpa = model.coef_[0]
slope_sat = model.coef_[1]
intercept = model.intercept_
print(f"Slope (GPA): {slope_gpa}")
print(f"Slope (SAT): {slope_sat}")
print(f"Intercept: {intercept}")

gpa_input = float(input("Enter gpa: "))  
sat_input = int(input("Enter sat score: "))  
predicted_attendance = model.predict([[gpa_input, sat_input]])[0]
print("Predicted Attendance is: ", predicted_attendance)


# GPA vs Attendance
plt.subplot(1, 2, 1)
plt.scatter(df['GPA'], df['Attendance'], color='blue', label='Data Points')
gpa_model = LinearRegression()
gpa_model.fit(df[['GPA']], df['Attendance'])
gpa_pred = gpa_model.predict(df[['GPA']])
plt.plot(df['GPA'], gpa_pred, color='red', label='Fitted Line')
plt.title('GPA vs Attendance')
plt.xlabel('GPA')
plt.ylabel('Attendance')
plt.legend()

# SAT vs Attendance
plt.subplot(1, 2, 2)
plt.scatter(df['SAT'], df['Attendance'], color='blue', label='Data Points')
sat_model = LinearRegression()
sat_model.fit(df[['SAT']], df['Attendance'])
sat_pred = sat_model.predict(df[['SAT']])
plt.plot(df['SAT'], sat_pred, color='red', label='Fitted Line')
plt.title('SAT vs Attendance')
plt.xlabel('SAT')
plt.ylabel('Attendance')
plt.legend()

plt.tight_layout()
plt.show()

# Plotting the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
