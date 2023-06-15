import pandas as pd

# Read the Titanic dataset into a pandas DataFrame
df = pd.read_csv('datasets/train.csv')

# Exploratory Data Analysis
# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Get summary statistics of the dataset
print("Summary statistics:")
print(df.describe())

# Check for missing values in the dataset
print("Missing values:")
print(df.isnull().sum())

# Data Cleaning
# Remove columns with excessive missing values
df = df.drop(['Cabin', 'Ticket'], axis=1)

# Impute missing values for 'Age' column with the median age
median_age = df['Age'].median()
df['Age'].fillna(median_age, inplace=True)

# Perform Analysis
# Calculate the survival rate by gender
survival_by_gender = df.groupby('Sex')['Survived'].mean()
print("Survival Rate by Gender:")
print(survival_by_gender)

# Calculate the survival rate by passenger class
survival_by_class = df.groupby('Pclass')['Survived'].mean()
print("Survival Rate by Passenger Class:")
print(survival_by_class)

# Visualize the data
import matplotlib.pyplot as plt

# Bar chart for survival rate by gender
plt.bar(survival_by_gender.index, survival_by_gender.values)
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Gender')
plt.show()

# Pie chart for survival rate by passenger class
plt.pie(survival_by_class.values, labels=survival_by_class.index, autopct='%1.1f%%')
plt.title('Survival Rate by Passenger Class')
plt.show()

# Additional Functionality
# Age Distribution of Survivors and Non-Survivors
survivors = df[df['Survived'] == 1]
non_survivors = df[df['Survived'] == 0]

plt.hist(survivors['Age'], bins=20, alpha=0.5, label='Survivors')
plt.hist(non_survivors['Age'], bins=20, alpha=0.5, label='Non-Survivors')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution of Survivors and Non-Survivors')
plt.legend()
plt.show()

# Fare vs. Age Scatter Plot
plt.scatter(df['Age'], df['Fare'], c=df['Survived'], cmap='coolwarm', alpha=0.6)
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Fare vs. Age (Colored by Survival)')
plt.colorbar(label='Survived')
plt.show()

# Model Building (Logistic Regression)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Select relevant features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

# Prepare the data
X = df[features]
y = df['Survived']
X.loc[:, 'Sex'] = X['Sex'].map({'male': 0, 'female': 1})  # Convert 'Sex' to numeric

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Survival Analysis by Age and Fare
# Survival analysis by age and fare
import seaborn as sns

# Scatter plot of age vs. fare colored by survival status
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='Fare', hue='Survived')
plt.title('Survival Analysis by Age and Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()

