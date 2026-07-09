# ============================================
# Part A: Import Libraries
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================
# Part B: Create Employee Dataset
# ============================================

data = {
    "Employee": ["A", "B", "C", "D", "E", "F"],
    "Age": np.random.randint(22, 51, 6),
    "Salary": np.random.randint(30000, 100001, 6),
    "Years_of_Experience": np.random.randint(1, 31, 6)
}

df = pd.DataFrame(data)

print("Employee Dataset:")
print(df)


# ============================================
# Part C: Basic Analysis
# ============================================

# Display First 3 Rows

print("\nFirst 3 Rows:")
print(df.head(3))

# Total Number of Employees

print("\nTotal Number of Employees:")
print(len(df))

# Mean Salary

print("\nMean Salary:")
print(df["Salary"].mean())

# Median Salary

print("\nMedian Salary:")
print(df["Salary"].median())

# Mode Salary

print("\nMode Salary:")
print(df["Salary"].mode()[0])

# Variance of Years of Experience

print("\nVariance of Years of Experience:")
print(df["Years_of_Experience"].var())

# Standard Deviation of Years of Experience

print("\nStandard Deviation of Years of Experience:")
print(df["Years_of_Experience"].std())


# ============================================
# Part D: Filtering & Sorting
# ============================================

average_salary = df["Salary"].mean()

# Employees Above Average Salary

print("\nEmployees Above Average Salary:")
print(df[df["Salary"] > average_salary])

# Employees Having Experience >=5 and Salary <50000

print("\nEmployees with at least 5 Years Experience and Salary below 50000:")
print(df[(df["Years_of_Experience"] >= 5) &
         (df["Salary"] < 50000)])

# Sort by Age (Descending)

print("\nEmployees Sorted by Age (Descending):")
print(df.sort_values(by="Age", ascending=False))


# ============================================
# Part E: Statistical Measures
# ============================================

Q1 = df["Salary"].quantile(0.25)
Q2 = df["Salary"].quantile(0.50)
Q3 = df["Salary"].quantile(0.75)

IQR = Q3 - Q1

print("\nQ1:")
print(Q1)

print("\nQ2:")
print(Q2)

print("\nQ3:")
print(Q3)

print("\nIQR:")
print(IQR)


# ============================================
# Part F: Detecting Outliers (IQR Method)
# ============================================

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df[
    (df["Salary"] < lower_limit) |
    (df["Salary"] > upper_limit)
]

print("\nOutliers in Salary Column:")
print(outliers)


# ============================================
# Part G: Data Visualization
# ============================================

# Histogram

plt.figure(figsize=(6,4))
plt.hist(df["Salary"], bins=5)
plt.title("Histogram of Salary")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.show()

# Boxplot

plt.figure(figsize=(6,4))
sns.boxplot(y=df["Years_of_Experience"])
plt.title("Boxplot of Years of Experience")
plt.show()

# Scatter Plot

plt.figure(figsize=(6,4))
plt.scatter(df["Salary"], df["Years_of_Experience"])
plt.xlabel("Salary")
plt.ylabel("Years of Experience")
plt.title("Salary vs Years of Experience")
plt.show()

# Correlation Matrix

corr = df[["Age", "Salary"]].corr()

print("\nCorrelation Matrix:")
print(corr)

plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Pairplot

sns.pairplot(df[["Age", "Salary", "Years_of_Experience"]])
plt.show()


# ============================================
# Part H: Advanced Queries
# ============================================

# Employee(s) with Highest Salary

highest_salary = df["Salary"].max()

print("\nEmployee(s) with Highest Salary:")
print(df[df["Salary"] == highest_salary])

# Youngest Employee with Highest Experience

highest_experience = df["Years_of_Experience"].max()

youngest_employee = df[df["Years_of_Experience"] == highest_experience]

youngest_employee = youngest_employee.sort_values(by="Age")

print("\nYoungest Employee with Highest Experience:")
print(youngest_employee.head(1))