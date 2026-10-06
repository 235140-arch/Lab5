# Lab 5 - Task 1: Simulating a fair six-sided die
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # fixed seed so results are reproducible

# Roll a fair die 1000 times
rolls = np.random.randint(1, 7, size=1000)

# (a) First 10 rolls
print("First 10 rolls:", rolls[:10])

# (b) Count how many times each face appeared
faces, counts = np.unique(rolls, return_counts=True)

# (c) Empirical vs theoretical probability
print(f"\n{'Face':<6}{'Count':<8}{'Empirical P':<14}{'Theoretical P'}")
for face, count in zip(faces, counts):
    print(f"{face:<6}{count:<8}{count / 1000:<14.3f}{1 / 6:.4f}")

# (d) Histogram of the 1000 rolls
plt.figure(figsize=(7, 4))
plt.hist(rolls, bins=np.arange(0.5, 7.5, 1), color="steelblue",
         edgecolor="black", rwidth=0.8)
plt.axhline(1000 / 6, color="red", linestyle="--", label="Expected (1000/6)")
plt.xticks(range(1, 7))
plt.xlabel("Die Face")
plt.ylabel("Frequency")
plt.title("1000 Rolls of a Fair Six-Sided Die")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("die_rolls.png")
plt.close()
print("\nSaved die_rolls.png")