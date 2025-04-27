
# Nom(s) étudiant(s) / Name(s) of student(s):
# Ndikumasabo, Ratzi-Chris (20266205)
# Islam, Hudaa Bint Afzal (20278949)

import sys

# Espace pour fonctions auxillaires :
# Space for auxilary functions :

# pour les tests de performances
def merge_sort(arr, d, f):
    if d < f:
        m = (d + f) // 2

        # Recursively sort the left and right halves
        merge_sort(arr, d, m)
        merge_sort(arr, m + 1, f)

        # Merge the sorted halves
        fusion(arr, d, m, f)

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

# Fusion de deux sous-tableaux
def fusion(arr, d, m, f):
    G = []
    D = []
    for i in range(m - d + 1):
        G.append(arr[d + i])
    for j in range(f - m):
        D.append(arr[m + 1 + j])

    i = j = 0
    k = d

    while i < len(G) and j < len(D):
        if G[i] <= D[j]:
            arr[k] = G[i]
            i += 1
        else:
            arr[k] = D[j]
            j += 1
        k += 1

    while i < len(G):
        arr[k] = G[i]
        i += 1
        k += 1

    while j < len(D):
        arr[k] = D[j]
        j += 1
        k += 1

# Tri fusion hybride
def hybrid_merge_sort(arr, d, f, seuil):
    if f - d + 1 <= seuil:
        insertion_sort(arr)
    else:
        m = (d + f) // 2
        hybrid_merge_sort(arr, d, m, seuil)
        hybrid_merge_sort(arr, m + 1, f, seuil)
        fusion(arr, d, m, f)

# Fonction à compléter / function to complete:
def solve(array) :
    seuil = 120  # seuil déterminé expérimentalement 
    merge_sort(array, 0, len(array) - 1)
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
