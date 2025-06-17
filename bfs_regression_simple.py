import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('least.csv')

X = df[['ID', 'ECHOvelocity']]
y = df['BFS']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(X_test)
print(y_pred)
print(f'R² Score: {r2_score(y_test, y_pred):.3f}')
print(f'MAE: {mean_absolute_error(y_test, y_pred):.3f}')
print(f'MSE: {mean_squared_error(y_test, y_pred):.3f}')

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_,
    'Absolute_Effect': np.abs(model.coef_)
})
feature_importance = feature_importance.sort_values('Absolute_Effect', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='Absolute_Effect', y='Feature', data=feature_importance, palette='viridis')
plt.title('Standardized feature importance analysis (sorted by absolute value of coefficients)')
plt.xlabel('importance')
plt.ylabel('feature')
plt.xlim(0, feature_importance['Absolute_Effect'].max()*1.2)
plt.show()

equation = f"BFS = {model.intercept_:.13f}"
for i, col in enumerate(X.columns):
    equation += f" + {model.coef_[i]:.13f}*({col}_normalization)"
print(equation)

scaler1 = StandardScaler()
scaler1.fit(X)
original_coef = model.coef_ / scaler1.scale_
original_intercept = model.intercept_ - np.sum(model.coef_ * scaler.mean_ / scaler.scale_)

equation = f"BFS = {original_intercept:.13f}"
for i, col in enumerate(X.columns):
    equation += f" + {original_coef[i]:.13f}*{col}"
print(equation)
