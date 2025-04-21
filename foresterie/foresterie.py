  
# Nom(s) étudiant(s) / Name(s) of student(s):

import sys

# Espace pour fonctions auxillaires :
# Space for auxilary functions :


# Source du code : https://leetcode.com/problems/house-robber/solutions/6147042/2-solutions-o-n-space-and-o-1-space/
# Vidéo youtbe : https://www.youtube.com/watch?v=ZwDDLAeeBM0&ab_channel=NickWhite
# Fonction à compléter / function to complete:
def solve(cost, forest) :
    n = len(forest)
    if n == 0:
        return 0
    elif n == 1:
        return forest[0]-cost

    # Étape 1 : Calculer les sommes maximales
    dp = [0] * n
    dp[0] = forest[0]-cost
    dp[1] = max(forest[0]-cost, forest[1]-cost)

    for i in range(2, n):
        dp[i] = max(dp[i-1], forest[i]-cost + dp[i-2])
    return dp[-1] 


# Ne pas modifier le code ci-dessous :
# Do not modify the code below :

def process_numbers(input_file):
    try:
        # Read integers from the input file
        with open(input_file, "r") as f:
            lines = f.readlines() 
            cost = int(lines[0].strip())  # cout d'exploitation pour couper un arbre
            forest = list(map(int, lines[1].split()))  # valeur de chaque arbre    

        return solve(cost, forest)
    
    except Exception as e:
        print(f"Error: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python foresterie.py <input_file>")
        return

    input_file = sys.argv[1]

    print(f"Input File: {input_file}")
    res = process_numbers(input_file)
    print(f"Result: {res}")

if __name__ == "__main__":
    main()