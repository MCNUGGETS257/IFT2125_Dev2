
# Nom(s) étudiant(s) / Name(s) of student(s):
# Islam, Hudaa Bint Afzal (20278949)
#

import sys

# Espace pour fonctions auxillaires :
# Space for auxilary functions :
def make_moves(node):
    if node is None:
        return 0
    
    left = make_moves(node.left)
    right = make_moves(node.right)

    if node.val == 0:
        moves = -1 + left + right
    elif node.val == 1:
        moves = left + right
    elif node.val > 1:
        moves = node.val - 1 + left + right

    node.val = moves

    return node.val

def count_moves(node):
    if node is None:
        return 0
    
    left = count_moves(node.left)
    right = count_moves(node.right)

    return abs(node.val) + abs(left) + abs(right)

# Fonction à compléter / function to complete:
def solve(root) :
    make_moves(root)
    return count_moves(root)

# Ne pas modifier le code ci-dessous :
# Do not modify the code below :

# Ne pas modifier le code ci-dessous :

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def process_numbers(input_file):
    try:
        # Read integers from the input file
        with open(input_file, "r") as f:
            lines = f.readlines() 
            tree_list = list(map(int, lines[0].split()))  # valeur de chaque noeud
            root = build_tree(tree_list)

        return solve(root)
    
    except Exception as e:
        print(f"Error: {e}")

def build_tree(lst):
    if not lst:
        return None
    
    root = TreeNode(lst[0])
    queue = [root]
    i = 1
    
    while i < len(lst):
        node = queue.pop(0)
        
        if i < len(lst):
            node.left = TreeNode(lst[i])
            queue.append(node.left)
            i += 1
        
        if i < len(lst):
            node.right = TreeNode(lst[i])
            queue.append(node.right)
            i += 1
    
    return root

def print_tree(root):
    if not root:
        return
    current_level = [root]
    while current_level:
        next_level = []
        values = []
        for node in current_level:
            if node:
                values.append(str(node.val))
                if node.left != None :
                    next_level.append(node.left)
                if node.right != None :
                    next_level.append(node.right)
        print(" ".join(values))
        current_level = next_level

def main():
    if len(sys.argv) != 2:
        print("Usage: python distribution.py <input_file>")
        return

    input_file = sys.argv[1]

    print(f"Input File: {input_file}")
    res = process_numbers(input_file)
    print(f"Result: {res}")

if __name__ == "__main__":
    main()

