
# Nom(s) étudiant(s) / Name(s) of student(s):

import random
import statistics
import sys
import time

# Espace pour fonctions auxillaires :
# Space for auxilary functions :
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key



# Fusion de deux sous-tableaux
def Fusion(T, d, m, f):
    G = []
    D = []
    for i in range(m - d + 1):
        G.append(T[d + i])
    for j in range(f - m):
        D.append(T[m + 1 + j])

    i = j = 0
    k = d

    while i < len(G) and j < len(D):
        if G[i] <= D[j]:
            T[k] = G[i]
            i += 1
        else:
            T[k] = D[j]
            j += 1
        k += 1

    while i < len(G):
        T[k] = G[i]
        i += 1
        k += 1

    while j < len(D):
        T[k] = D[j]
        j += 1
        k += 1

# Tri fusion hybride
def TriFusionHybride(T, d, f, seuil):
    if f - d + 1 <= seuil:
        insertion_sort(T)
    else:
        m = (d + f) // 2
        TriFusionHybride(T, d, m, seuil)
        TriFusionHybride(T, m + 1, f, seuil)
        Fusion(T, d, m, f)

    # Fonction pour determiner le seuil optimal
    def evaluate_seuil(seuil,size,num_trials):
        times = []
        for _ in range(num_trials):
            arr = [random.randint(0, 10000) for _ in range(size)]
            start = time.perf_counter()
            TriFusionHybride(arr.copy(),0,len(arr)-1, seuil)
            end = time.perf_counter()
            times.append(end - start)
        return statistics.mean(times)
    seuils = range(1, 100, 5)  # seuils à tester
    sizes = [1000, 2000, 5000]

    for size in sizes:
        print(f"\n--- Taille = {size} ---")
        for seuil in seuils:
            avg_time = evaluate_seuil(seuil,size,5)
            print(f"Seuil: {seuil}, Temps moyen: {avg_time:.6f} sec")


# Fonction à compléter / function to complete:
def solve(array):
    seuil = 90  # seuil déterminé expérimentalement 
    TriFusionHybride(array, 0, len(array) - 1, seuil)
    return array
# Ne pas modifier le code ci-dessous :
# Do not modify the code below :

def process_numbers(input_file):
    try:
        # Read integers from the input file
        with open(input_file, "r") as f:
            lines = f.readlines() 
            array = list(map(int, lines[0].split()))  # valeur de chaque noeud  

        return solve(array)
    
    except Exception as e:
        print(f"Error: {e}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python tri_hybride.py <input_file>")
        return

    input_file = sys.argv[1]

    print(f"Input File: {input_file}")
    res = process_numbers(input_file)
    print(f"Result: {res}")

if __name__ == "__main__":
    main()
