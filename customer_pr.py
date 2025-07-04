import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Load your dataset
df = pd.read_csv('customer_support_tickets.csv')

# Display first few rows
print(df.head())

# Dataset structure
print(df.info())

# Summary statistics
print(df.describe(include='all'))
# Check for missing values
print(df.isnull().sum())

# Drop columns that are irrelevant or identifiers
df.drop(['Ticket ID', 'Customer Name', 'Customer Email', 'Ticket Subject', 'Ticket Description', 'Resolution'], axis=1, inplace=True)

# Drop rows where Customer Satisfaction Rating is NaN (only 2769 rows have ratings)
df = df.dropna(subset=['Customer Satisfaction Rating'])

# Fill missing 'Time to Resolution' with a placeholder (if needed)
df['Time to Resolution'] = df['Time to Resolution'].fillna('Unknown')

# Convert Date columns to datetime
df['Date of Purchase'] = pd.to_datetime(df['Date of Purchase'], errors='coerce')
df['First Response Time'] = pd.to_datetime(df['First Response Time'], errors='coerce')
df['Time to Resolution'] = pd.to_datetime(df['Time to Resolution'], errors='coerce')
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
X = df.drop('Customer Satisfaction Rating', axis=1)
y = df['Customer Satisfaction Rating']  # Regression or Classification task depending on problem framing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Drop datetime columns from features
# Drop datetime columns
X = X.drop(['Date of Purchase', 'First Response Time', 'Time to Resolution'], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)

# Predictions
y_pred = rfc.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature Importance Plot
feature_importances = pd.Series(rfc.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Feature Importances')
plt.show()

df['YearMonth'] = df['Date of Purchase'].dt.to_period('M')
ticket_trends = df.groupby('YearMonth').size()

plt.figure(figsize=(10, 6))
ticket_trends.plot(marker='o')
plt.title('Customer Support Ticket Trends Over Time')
plt.xlabel('Year-Month')
plt.ylabel('Number of Tickets')
plt.grid()
plt.show()
# Customer Satisfaction Distribution
sns.histplot(df['Customer Satisfaction Rating'], bins=5, kde=True)
plt.title('Customer Satisfaction Distribution')
plt.show()

# Ticket Type Distribution
df['Ticket Type'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Ticket Type Distribution')
plt.show()
