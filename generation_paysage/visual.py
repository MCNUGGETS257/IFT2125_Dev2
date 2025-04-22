import matplotlib.pyplot as plt
import numpy as np

def show_depth_matrix(depths, title="Depth Matrix"):
    # Convert to a NumPy array for easy handling
    depth_array = np.array(depths)

    # Create the plot
    plt.figure(figsize=(6, 6))
    plt.imshow(depth_array, cmap='terrain', interpolation='nearest')
    plt.colorbar(label="Depth")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().invert_yaxis()  # Optional: top-left is (0,0)
    plt.grid(False)
    plt.show()