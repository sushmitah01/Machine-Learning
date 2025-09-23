import numpy as np
import matplotlib.pyplot as plt

# Example dataset
x = np.random.uniform(1, 100, size=50)

# If you just need a histogram
plt.hist(x, bins=5)
plt.show()
