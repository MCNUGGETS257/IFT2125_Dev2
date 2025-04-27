import random
import copy
import time
import tri_hybride
import matplotlib.pyplot as plt  # For plotting the graph

def TriFusion(T, d, f, seuil):
    if d>=f:
        return
    else:
        m = (d + f) // 2
        TriFusion(T, d, m, seuil)
        TriFusion(T, m + 1, f, seuil)
        tri_hybride.Fusion(T, d, m, f)
max_size = 100000
# Preparing the arrays
to_sort = []
for i in range(max_size):
    to_sort.append([random.randint(0, 1000) for _ in range(i+1)])

insertion = [0 for _ in range(max_size)]
fusion = [0 for _ in range(max_size)]

to_sort_copy = copy.deepcopy(to_sort)

# Timing insertion sort
for i in range(max_size):
    arr = to_sort[i]
    start = time.perf_counter()
    tri_hybride.insertion_sort(arr)
    end = time.perf_counter()
    insertion[i] = end - start

# Timing fusion (merge) sort
for i in range(max_size):
    arr = to_sort_copy[i]
    start = time.perf_counter()
    TriFusion(arr, 0, len(arr) - 1, 1)
    end = time.perf_counter()
    print(arr)
    fusion[i] = end - start

plt.plot(range(1, max_size + 1), insertion, label="Insertion Sort Time")
plt.plot(range(1, max_size + 1), fusion, label="Fusion Sort Time")
plt.xlabel("Number of Elements")
plt.ylabel("Time (seconds)")
plt.title("Insertion vs Fusion Sort Timing")
plt.legend()
plt.grid(True)
plt.show()
# with open("to_sort.json", "w") as f:
#     json.dump(to_sort, f, indent=4)