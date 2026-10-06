# Lab 5 - Task 3: Comparing a simulated Normal distribution with real delivery times
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(7)

# ---------- Create the "real" dataset: 97 normal deliveries + 3 very late ones ----------
normal_part = np.random.normal(loc=31, scale=5, size=97).round().clip(min=15)
late_part = np.array([68, 71, 75])
times = np.concatenate([normal_part, late_part])
np.random.shuffle(times)

pd.DataFrame({
    "OrderID": range(1, 101),
    "DeliveryTime(min)": times.astype(int),
}).to_csv("delivery_times.csv", index=False)

df = pd.read_csv("delivery_times.csv")
real = df["DeliveryTime(min)"]

# ---------- (a) Mean and standard deviation of real data ----------
mean = real.mean()
std = real.std()
print(f"Real delivery times -- Mean: {mean:.1f} min   Std: {std:.1f} min")

# ---------- (b) Simulate 100 "expected" times with the same mean & std ----------
simulated = np.random.normal(loc=mean, scale=std, size=100)
print(f"Simulated 100 values from Normal(mean={mean:.1f}, std={std:.1f})")

# ---------- (c) Overlapping histograms ----------
bins = np.arange(0, 85, 5)
plt.figure(figsize=(8, 4.5))
plt.hist(real, bins=bins, alpha=0.6, color="steelblue", edgecolor="black", label="Real")
plt.hist(simulated, bins=bins, alpha=0.6, color="indianred", edgecolor="black", label="Simulated (Normal)")
plt.xlabel("Delivery Time (min)")
plt.ylabel("Number of Deliveries")
plt.title("Real vs. Simulated Delivery Times")
plt.legend()
plt.tight_layout()
plt.savefig("delivery_comparison_histogram.png")
plt.close()

# ---------- (d) Short EDA on real data ----------
print("\nMissing values:", df.isnull().sum().sum())

plt.figure(figsize=(4, 5))
plt.boxplot(real)
plt.ylabel("Delivery Time (min)")
plt.title("Delivery Time Boxplot")
plt.tight_layout()
plt.savefig("delivery_boxplot.png")
plt.close()

q1 = real.quantile(0.25)
q2 = real.quantile(0.50)
q3 = real.quantile(0.75)
print(f"Percentiles -- 25th: {q1}   50th: {q2}   75th: {q3}")

# ---------- (e) IQR outlier detection ----------
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
print(f"IQR = {iqr}   Outlier bounds: {lower} - {upper}")

outliers = df[real > upper]   # unusually LONG deliveries
print(f"\nOutliers detected: {len(outliers)} deliveries "
      f"({sorted(outliers['DeliveryTime(min)'].tolist())} minutes)")
print(outliers.to_string(index=False))
print(f"Percentage flagged as outliers: {len(outliers) / len(df) * 100:.1f}%")

print("\nSaved delivery_comparison_histogram.png, delivery_boxplot.png")