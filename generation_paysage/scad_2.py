def minecraft(matrix):
    row = len(matrix)
    col = len(matrix[0])

    with open("test.scad", "w") as f:
        for y in range(row):
            for x in range(col):
                h = matrix[y][x]
                translate = f"translate([{x}, {y}, 0])\n"
                cube = f"cube([1, 1, {h:.2f}]);\n"
                f.write(translate)
                f.write(cube)
    
    f.close()

def polyhedral(heights):
    with open("mesh.scad", "w") as f:
        land = landscape(heights)
        f.write(land)
    f.close()

def landscape(heights, cell_size=1.0, z_scale=1.0):
    rows = len(heights)
    cols = len(heights[0])
    scad = ""

    flat = [z * z_scale for row in heights for z in row]
    z_min = min(flat)
    z_max = max(flat)
    sea_level = 2.0 * z_scale  # ajusté au relief

    def get_color(z):
        if z <= sea_level:
            # Dégradé de bleu selon la profondeur
            shade = (z - z_min) / (sea_level - z_min + 1e-6)
            return [0.0, 0.3 + 0.4 * shade, 0.6 + 0.4 * shade]  # bleu foncé → bleu clair
        else:
            # Dégradé terrain vert → brun → blanc
            shade = (z - sea_level) / (z_max - sea_level + 1e-6)
            if shade < 0.33:
                return [0.0, 0.6 + 0.4 * (shade / 0.33), 0.0]  # vert clair → vert
            elif shade < 0.66:
                t = (shade - 0.33) / 0.33
                return [0.5 + 0.3 * t, 0.4 - 0.3 * t, 0.1]      # brun
            else:
                t = (shade - 0.66) / 0.34
                return [1.0, 1.0, 1.0]                         # neige

    for y in range(rows - 1):
        for x in range(cols - 1):
            # Get heights of the 4 corners
            z00 = heights[y][x] * z_scale
            z10 = heights[y][x + 1] * z_scale
            z11 = heights[y + 1][x + 1] * z_scale
            z01 = heights[y + 1][x] * z_scale

            # Points top et bottom
            top = [
                [x * cell_size, y * cell_size, z00],
                [(x + 1) * cell_size, y * cell_size, z10],
                [(x + 1) * cell_size, (y + 1) * cell_size, z11],
                [x * cell_size, (y + 1) * cell_size, z01],
            ]
            bottom = [
                [x * cell_size, y * cell_size, 0],
                [(x + 1) * cell_size, y * cell_size, 0],
                [(x + 1) * cell_size, (y + 1) * cell_size, 0],
                [x * cell_size, (y + 1) * cell_size, 0],
            ]
            points = top + bottom

            faces = [
                [0, 1, 2], [0, 2, 3],       # Top
                [4, 6, 5], [4, 7, 6],       # Bottom
                [0, 4, 5], [0, 5, 1],       # Side 1
                [1, 5, 6], [1, 6, 2],       # Side 2
                [2, 6, 7], [2, 7, 3],       # Side 3
                [3, 7, 4], [3, 4, 0],       # Side 4
            ]

            avg_z = (z00 + z10 + z11 + z01) / 4
            color = get_color(avg_z)

            scad += f"color([{color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f}]) {{\n"
            scad += "  polyhedron(\n"
            scad += "    points = [\n"
            for pt in points:
                scad += f"      {pt},\n"
            scad += "    ],\n    faces = [\n"
            for face in faces:
                scad += f"      {face},\n"
            scad += "    ]\n  );\n"
            scad += "}\n"

    return scad
