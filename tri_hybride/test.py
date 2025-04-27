import random
import copy
import time
import tri_hybride
import matplotlib.pyplot as plt

max_size = 150
# Preparing the arrays
to_sort = []
for i in range(max_size):
    to_sort.append([random.randint(0, max_size) for _ in range(i)])

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
    tri_hybride.merge_sort(arr, 0, len(arr) - 1)
    end = time.perf_counter()
    fusion[i] = end - start

# max_size = 150
# # Preparing the arrays
# to_sort = [random.randint(0, max_size) for _ in range(max_size + 1)]

# insertion = [0 for _ in range(max_size)]
# fusion = [0 for _ in range(max_size)]

# # Timing insertion sort
# for i in range(max_size):
#     arr = list(to_sort[0:i+1])
#     start = time.perf_counter()
#     tri_hybride.insertion_sort(arr)
#     end = time.perf_counter()
#     insertion[i] = end - start

# # Timing fusion (merge) sort
# for i in range(max_size):
#     arr = list(to_sort[0:i+1])
#     start = time.perf_counter()
#     tri_hybride.merge_sort(arr, 0, len(arr) - 1)
#     end = time.perf_counter()
#     fusion[i] = end - start

plt.plot(range(1, max_size + 1), insertion, label="temps du tri par insertion")
plt.plot(range(1, max_size + 1), fusion, label="temps du tri fusion")
plt.xlabel("nombre d'éléments dans la liste")
plt.ylabel("temps (secondes)")
plt.title("Comparaison de la performance du tri par insertion et du tri par fusion\n sur des listes de tailles différentes")
plt.legend()
plt.grid(True)
plt.show()
# with open("to_sort.json", "w") as f:
#     json.dump(to_sort, f, indent=4)