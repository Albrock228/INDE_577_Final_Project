import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [4, 5, 6])
plt.title("Test Plot Saving")

# Save the plot
try:
    plt.savefig("test_plot.png", bbox_inches="tight")
    print("Plot saved successfully as 'test_plot.png'")
except Exception as e:
    print(f"Failed to save plot: {e}")
