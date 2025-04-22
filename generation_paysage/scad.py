def scad(matrix):
    row = len(matrix)
    col = len(matrix[0])

    with open("terrain_mesh.scad", "w") as f:
        # Plan océan bleu
        f.write("color([0, 0.3, 0.7])\n")
        f.write(f"translate([0, 0, 0])\n")
        f.write(f"cube([{col}, {row}, 1]);\n\n")

        f.write("color([0.3, 0.9, 0.3])\n")  # Vert clair pour terrain
        f.write("polyhedron(points=[\n")

        # Liste des sommets
        for y in range(row):
            for x in range(col):
                h = matrix[y][x]
                f.write(f"    [{x}, {y}, {h:.2f}],\n")
        f.write("],\n")

        # Liste des faces
        f.write("faces=[\n")
        for y in range(row - 1):
            for x in range(col - 1):
                # Indices des sommets
                i = y * col + x
                i_right = i + 1
                i_down = i + col
                i_diag = i + col + 1

                # Deux triangles pour chaque cellule
                f.write(f"    [{i}, {i_right}, {i_down}],\n")
                f.write(f"    [{i_right}, {i_diag}, {i_down}],\n")
        f.write("]);\n")
