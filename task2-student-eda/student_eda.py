# Lab 5 - Task 2: Full EDA pass on a student performance dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Create the dataset (12 rows: 2 missing Marks, 1 exact duplicate) ----------
data = {
    "StudentID":     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 5],
    "Gender":        ["Male", "Female", "Male", "Female", "Male", "Female",
                      "Male", "Female", "Male", "Female", "Male", "Male"],
    "StudyHours":    [2, 5, 3, 8, 6, 4, 7, 1, 9, 5, 3, 6],
    "Attendance(%)": [78, 92, 85, 90, 88, 72, 80, 75, 94, 86, 82, 88],
    "Marks":         [55, 72, np.nan, 88, 78, 68, 82, 50, 92, np.nan, 60, 78],
}
# last row is an exact duplicate of StudentID 5
pd.DataFrame(data).to_csv("student_performance.csv", index=False)

# ---------- (a) Load and inspect ----------
df = pd.read_csv("student_performance.csv")
print("Shape:", df.shape)
print()
df.info()

# ---------- (b) Missing values & duplicates, then clean ----------
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nMissing values in Marks:", df["Marks"].isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()
median_marks = df["Marks"].median()
df["Marks"] = df["Marks"].fillna(median_marks)
print(f"Filled missing Marks with median = {median_marks}")
print("After cleaning:", df.shape)

# ---------- (c) Univariate analysis: histogram & boxplot of Marks ----------
plt.figure(figsize=(6, 4))
plt.hist(df["Marks"], bins=6, color="steelblue", edgecolor="black")
plt.xlabel("Marks")
plt.ylabel("Number of Students")
plt.title("Marks Distribution")
plt.tight_layout()
plt.savefig("marks_histogram.png")
plt.close()

plt.figure(figsize=(4, 5))
plt.boxplot(df["Marks"])
plt.ylabel("Marks")
plt.title("Marks Boxplot")
plt.tight_layout()
plt.savefig("marks_boxplot.png")
plt.close()

# ---------- (d) Bivariate analysis: scatter + correlation ----------
plt.figure(figsize=(6, 4))
plt.scatter(df["StudyHours"], df["Marks"], color="darkorange")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs. Marks")
plt.tight_layout()
plt.savefig("studyhours_vs_marks.png")
plt.close()

corr_matrix = df[["StudyHours", "Attendance(%)", "Marks"]].corr()
print("\nCorrelation matrix:")
print(corr_matrix)

# ---------- (e) Categorical breakdown by Gender ----------
print("\nStudents per Gender:")
print(df["Gender"].value_counts())
gender_summary = df.groupby("Gender")[["Marks", "Attendance(%)"]].mean().round(1)
print("\nAverage Marks and Attendance by Gender:")
print(gender_summary)

print("\nSaved marks_histogram.png, marks_boxplot.png, studyhours_vs_marks.png")