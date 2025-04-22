import random
from collections import deque
import math
import visual

def print_matrix(matrix):
    # Print the matrix
    for row in matrix:
        print(' '.join(map(str, row)))
#--------------------------------------------------------------------------------------------------------
#               PERLIN NOISE
#--------------------------------------------------------------------------------------------------------
class Vect2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def dot(self, other):
        return self.x*other.x +self.y*other.y

def shuffle(array):
    """ This function shuffles an array in O(n), it takes it goes through an array and at index i
        choses an element randomly from array[i:] to place at array[i]
        """
    a = 0
    b = len(array) - 1

    for i in range(len(array)):
        idx = random.randint(a,b)
        temp = array[i]
        array[i] = array[idx]
        array[idx] = temp
        a += 1
    return array

def permutation():
    array = [0]*256
    for i in range(256):
        array[i] = i
    shuffle(array)
    array = array*2
    return array

def fade(t):
    return ((6*t-15)*t+10)*t*t*t

def lerp(t, a1, a2):
    return a1 + t*(a2-a1)

def constVect(v):
    h = v%4
    if h == 0:
        return Vect2D(1.0, 1.0)
    if h == 1:
        return Vect2D(-1.0, 1.0)
    if h == 2:
        return Vect2D(-1.0, -1.0)
    else:
        return Vect2D(1.0, -1.0)
    
def noise2D(x, y):
    X = int(x) & 225
    Y = int(y) & 225
    
    xf = x - int(x)
    yf = y - int(y)

    tr = Vect2D(xf - 1.0, yf - 1.0)
    tl = Vect2D(xf, yf - 1.0)
    br = Vect2D(xf - 1.0, yf)
    bl = Vect2D(xf, yf)

    p = permutation()

    vtr = p[p[X+1]+Y+1]
    vtl = p[p[X]+Y+1]
    vbr = p[p[X+1]+Y]
    vbl = p[p[X]+Y]

    # constant vector
    dtr = tr.dot(constVect(vtr))
    dtl = tl.dot(constVect(vtl))
    dbr = br.dot(constVect(vbr))
    dbl = bl.dot(constVect(vbl))

    u = fade(xf)
    v = fade(yf)

    return lerp(u, lerp(v, dbl, dtl), lerp(v, dbr, dtr))

def perlin(array):
    for y in range(len(array)):
        for x in range(len(array[0])):
            a = random.random() * random.randint(1,256) + 0.01
            b = random.random() * random.randint(1,256) + 0.01

            #n = noise2D(a, b)

            n = 0.0
            a = 1
            f = 0.005
            for o in range(3):
                v = a*noise2D(x*f, y*f)
                n += v

                a*= 0.5
                f*= 2.0
            n+= 1
            n*= 0.5

            n_normalized = (n + 1) / 2
            array[y][x] = n_normalized

    return array
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------
#               SCAD    
#--------------------------------------------------------------------------------------------------------
# Version corrigée de la fonction `minecraft` pour générer un dégradé bleu lisible dans OpenSCAD

def clamp(value):
        return max(0.0, min(1.0, value))

def get_color_from_shade(shade, is_water=False):
    shade = clamp(shade)
    if is_water:
        return [0.0, clamp(0.3 + 0.4 * shade), clamp(0.6 + 0.4 * shade)]
    else:
        if shade < 0.33:
            return [0.0, clamp(0.6 + 0.4 * (shade / 0.33)), 0.0]
        elif shade < 0.66:
            t = (shade - 0.33) / 0.33
            return [clamp(0.5 + 0.3 * t), clamp(0.4 - 0.3 * t), 0.1]
        else:
            return [1.0, 1.0, 1.0]
def minecraft(heights, highest, depths, max_depth):
    """ Version corrigée : couleur non multipliée par p mais interpolée directement à partir de shade """
    row = len(heights)
    col = len(heights[0])
    scad = ""
    sea_level = 2.0

    for y in range(row):
        for x in range(col):
            h = heights[y][x]

            if h <= sea_level and depths[y][x] < 0:
                shade = abs(depths[y][x]) / max_depth
                color = get_color_from_shade(shade, is_water=True)
            else:
                shade = h / highest
                color = get_color_from_shade(shade, is_water=False)

            scad += f"color([{color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f}])\n"
            scad += f"translate([{x}, {y}, 0])\n"
            scad += f"cube([1, 1, {h:.2f}]);\n"
    # Ajout du texte RC-HDA IFT2125 
    scad += "color([0, 0, 0])\n"
    scad += "translate([28, 5, -0.1])\n"
    scad += 'mirror([0, 1, 0])\n'
    scad += "linear_extrude(height = 1)\n"
    scad += "text(\"H.I & R-C.N IFT2125\", size = 4, font = \"Liberation Sans\", halign = \"center\");\n"
    return scad

def polyhedral(heights, highest, depths, max_depth, cell_size=1.0, z_scale=1.0):
    """ Returns a string to visualize a heightmap in color 3D in openSCAD in polyhedric mesh style"""
    rows = len(heights)
    cols = len(heights[0])
    scad = ""
    sea_level = 2.0   # ajusté au relief

    # Generate one cube per cell
    for y in range(rows - 1):
        for x in range(cols - 1):
            # Get heights of the 4 corners of the current cell
            z00 = heights[y][x] * z_scale
            z10 = heights[y][x + 1] * z_scale
            z11 = heights[y + 1][x + 1] * z_scale
            z01 = heights[y + 1][x] * z_scale

            # Compute the points for the top surface
            top = [
                [x * cell_size, y * cell_size, z00],
                [(x + 1) * cell_size, y * cell_size, z10],
                [(x + 1) * cell_size, (y + 1) * cell_size, z11],
                [x * cell_size, (y + 1) * cell_size, z01],
            ]

            # Bottom points (z = 0)
            bottom = [
                [x * cell_size, y * cell_size, 0],
                [(x + 1) * cell_size, y * cell_size, 0],
                [(x + 1) * cell_size, (y + 1) * cell_size, 0],
                [x * cell_size, (y + 1) * cell_size, 0],
            ]

            # Combine points
            points = top + bottom

            # Define faces using local indices
            faces = [
                [0, 1, 2], [0, 2, 3],       # Top face
                [4, 6, 5], [4, 7, 6],       # Bottom face (reversed)
                [0, 4, 5], [0, 5, 1],       # Side 1
                [1, 5, 6], [1, 6, 2],       # Side 2
                [2, 6, 7], [2, 7, 3],       # Side 3
                [3, 7, 4], [3, 4, 0],       # Side 4
            ]

            # Use average height for color logic
            max_z = max(z00, z10, z11, z01)
            if max_z <= sea_level and depths[y][x] < 0:
                shade = abs(depths[y][x]) / max_depth
                color = get_color_from_shade(shade, is_water=True)
            else:
                shade = max_z / highest
                color = get_color_from_shade(shade, is_water=False)
            # Generate the OpenSCAD code for this cell
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
    # Ajout du texte RC-HDA IFT2125 
    scad += "color([0, 0, 0])\n"
    scad += "translate([28, 5, -0.8])\n"
    scad += 'mirror([0, 1, 0])\n'
    scad += "linear_extrude(height = 1)\n"
    scad += "text(\"H.I & R-C.N IFT2125\", size = 4, font = \"Liberation Sans\", halign = \"center\");\n"

    return scad

def scad(landscape, dim, highest, file="mesh.scad", adjust=0):
    """ Creates a scad file where the landscape fits in a 50-80mm.
        Adjustments are needed for polyhedric style to fit in that range (put adjust at 1) """
    row = dim[0]
    col = dim[1]
    scad = ""
    max_size = 80    # in mm
    min_size = 50

    # for polyhedric style
    if adjust == 1:
        min_size += 2
        max_size += 1

    x = max((col/100.0)*max_size, min_size)
    y = max((row/100.0)*max_size, min_size)
    x = x/col
    y = y/row 

    p = min(y, x)
    z = max(p*highest, 40)
    z = z/40

    scaling = f"Multiplying row dimension by {x:.2f}, column by {y:.2f} and height by {z:.2f}."
    print(scaling)

    scad += f"scale([{x:.2f}, {y:.2f}, {z:.2f}])"
    scad += "\n union() {" + landscape + "}"
    

    with open(file, "w") as f:
        f.write(scad)
    f.close()
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------
#               AESTHETICS 
#--------------------------------------------------------------------------------------------------------
def noise(heights):
    """ Applies perlin noise to the heights matrix """
    row = len(heights)
    col = len(heights[0])
    
    noise = perlin([[0 for _ in range(col)] for _ in range(row)])

    for y in range(row):
        for x in range(col):
            heights[y][x] = heights[y][x] * noise[y][x]

def kernel(size, sigma):
    """ helper function for gaussian blur """
    kernel = [[0.0 for _ in range(size)] for _ in range(size)]
    sum_val = 0.0
    offset = size // 2

    for y in range(size):
        for x in range(size):
            dx = x - offset
            dy = y - offset
            val = math.exp(-(dx*dx + dy*dy) / (2 * sigma * sigma))
            kernel[y][x] = val
            sum_val += val

    # Normalize the kernel
    for y in range(size):
        for x in range(size):
            kernel[y][x] /= sum_val

    return kernel

def blur(heights, kernel):
    """ Gaussian blur for smooth transitions between heights after applying noise """
    rows = len(heights)
    cols = len(heights[0])
    k_size = len(kernel)
    k_offset = k_size // 2

    # Output matrix
    result = [[0.0 for _ in range(cols)] for _ in range(rows)]

    for y in range(rows):
        for x in range(cols):
            acc = 0.0
            for ky in range(k_size):
                for kx in range(k_size):
                    ny = y + ky - k_offset
                    nx = x + kx - k_offset

                    if 0 <= ny < rows and 0 <= nx < cols:
                        acc += heights[ny][nx] * kernel[ky][kx]
            result[y][x] = acc

    return result
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------
#               LANDSCAPING
#--------------------------------------------------------------------------------------------------------
def neighbours(x, y, row, col, numNeighbours=8):
    """ Returns the neighbours (that exist) of a cell at coords [x,y] in a matrix rowxcol """  
    # Von Newmann neighbourhood  
    if numNeighbours == 4:
        directions = [[-1,0],[1,0],[0,-1],[0,1]]
    # Moore neighbourhood
    elif numNeighbours == 8:
        directions = [[-1,-1],[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1],[-1,0]]
    
    neighbourhood = []
    for i in directions:
        a = x + i[0]
        b = y + i[1]
        if 0 <= a < col and 0 <= b < row:
            neighbourhood.append([a,b])
    return neighbourhood

def draw(matrix, x, y, type, to_draw):
    """ Randomly creates shapes of x cells, x = to_draw, in a matrix 
        by changing the content of the cell to type """
    while to_draw > 0:
        row = len(matrix)
        col = len(matrix[0])
        if 0 <= x < col and 0 <= y < row:
            if matrix[y][x] == type:
                neighbourhood = neighbours(x, y, row, col)
                candidates = []
                for n in neighbourhood:
                    if matrix[n[1]][n[0]] != type:
                        candidates.append(n)
                
                if len(candidates) > 0:
                    cell = candidates[random.randint(0, len(candidates)) - 1]
                    matrix[cell[1]][cell[0]] = type
                    to_draw -= 1
                    continue
                else:
                    radius = 2
                    x = random.randint(max(0, x - radius), min(col - 1, x + radius))
                    y = random.randint(max(0, y - radius), min(row - 1, y + radius))
                    continue
            else:
                matrix[y][x] = type
                to_draw -= 1
                
                dx = random.choice([-1,0,1])
                dy = random.choice([-1,0,1])

                # case (0,0)
                if dx == 0 and dy == 0:
                    d = random.choice([-1,1])
                    x_or_y = random.randint(0,10)
                    if x_or_y%2 == 0:
                        dx = d
                    else:
                        dy = d
                x += dx
                y += dx

        else:
            if x < 0 :
                x = 0
            elif x >= col:
                x = col - 1
            if y < 0:
                y = 0
            elif y >= row:
                y = row - 1
            continue

def blob(matrix, ocean):
    """ Creates irregular shapes, blobs of n cells, n = ocean, in a matrix """
    row = len(matrix)
    col = len(matrix[0])

    nb_cells = int(ocean*row*col)
    
    x = random.randint(0,col)
    y = random.randint(0,row)
    draw(matrix, x, y, 1, nb_cells)
    
def borders(blobs):
    """ Find borders withing a matrix of blobs defined by 0s and 1s
        Returns [
                    [blobs of 0s = [border_cells, in_land_cells], ...],
                    [blobs of 1s = [border_cells, in_land_cells], ...]
                ]"""
    
    row = len(blobs)
    col = len(blobs[0])
    visited = [[False for _ in range(col)] for _ in range(row)]
    shapes = [[],[]]

    for y in range(row):
        for x in range(col):
            if visited[y][x] == False and blobs[y][x] == 0:
                shapes[0].append(shape(blobs, [x,y], visited))
            if visited[y][x] == False and blobs[y][x] == 1:
                shapes[1].append(shape(blobs, [x,y], visited, 1))

    return shapes

def shape(blobs, start, visited, type=0):
    """ From a starting cell [x,y] = start, with a visited matrix containing booleans that 
        signify if this cells has alrady been accounted for, this function finds and return
        the borders of a of the shapes formed by neighbouring cells of the blobs matrix which 
        the cells == type as well as the cells inside of the shape.

        Return format is [[border_cells],[inner_cells]] """
    row = len(visited)
    col = len(visited[0])
    queue = deque([start])
    borders = []
    in_bound = []

    while queue:
        cell = queue.popleft()
        x = cell[0]
        y = cell[1]

        if visited[y][x] == True: 
            continue

        visited[y][x] = True
        # find neighbours
        neighbourhood = neighbours(x, y, row, col)
        border = False
        if len(neighbourhood) < 8:
            border = True
        for n in neighbourhood:
            if blobs[n[1]][n[0]] == type:
                if visited[n[1]][n[0]] == False:
                    queue.append(n)
            else:
                border = True
        if border == True:
            borders.append([x,y])
        else:
            in_bound.append([x,y])

    return [borders, in_bound]

def depth(blobs, shapes):
    """ Return a new matrix where the cells indicate how deep inside the shape the cell is.
        Negative for shapes of 0s and positive for cells of 1s. 
        Returns also the deepest depth for each shape.
        Return output follows [
                                depth_matrix, [
                                                [list of biggest_depth for shapes of 0s], 
                                                [list of biggest_depth for shapes of 0s]
                                                ]
                                ] """
    row = len(blobs)
    col = len(blobs[0])

    depths = [[0 for _ in range(col)] for _ in range(row)]
    shape_max_depth = [[],[]]
    queue = deque()

    for i in [0,1]:
        for s in shapes[i]:
            mask = [[0 for _ in range(col)] for _ in range(row)]
            for j in [0,1]:
                for coord in s[j]:
                    x = coord[0]
                    y = coord[1]
                    mask[y][x] = 1
            if i == 0:
                direction = 1
            elif i == 1:
                direction = -1
            # initialize the borders
            how_deep = direction
            for border_cell in s[0]:
                x = border_cell[0]
                y = border_cell[1]
                if 0 < x < col - 1 and 0 < y < row - 1:
                    depths[y][x] = how_deep
                    queue.append([x,y])

            while queue:
                cell = queue.popleft()
                x = cell[0]
                y = cell[1]

                how_deep = depths[y][x]

                neighbourhood = neighbours(x, y, row, col)
                for n in neighbourhood:
                    nx = n[0]
                    ny = n[1]

                    if mask[ny][nx] == 1 and depths[ny][nx] == 0:
                        depths[ny][nx] = how_deep + 1*direction
                        queue.append([nx,ny])
            max_depth = how_deep
            shape_max_depth[i].append(max_depth)
                
    return [depths, shape_max_depth]

def slope(heights, start, shape, sigma, max_height, threshold):
    """ Forms a slope using the gaussian slope.
        Updates the heights of the slope in the heights matrix. 
        It only updates the cells of the heights matrix that forms the specified shape.
        The steepness of the slope depend on sigma. 
        The slope starts at a height of max_height.
        The heights don't go lower than the threshold. """
    row = len(heights)
    col = len(heights[0])
    cx = start[0]
    cy = start[1]

    for i in [0,1]:
        for x, y in shape[i]:
            dx = x - cx
            dy = y - cy
            dist_sq = dx*dx + dy*dy
            value = max_height * math.exp(-dist_sq / (2 * sigma * sigma))
            if value >= threshold:
                heights[y][x] = value
            else:
                heights[y][x] = threshold

    return heights

def crater(heights, center, shape, radius, crater_floor, rim_height, sigma):
    """ Forms a crater in the specified shape from the center coordinates 
        and updates the heights in the specified heights matrix. The crater
        doesn't go beyound the cells in the radius specified is flat in it center 
        to a radius of crater_floor. The rim of the crater has a height of rim height
        and it's steepness depends on sigma. It's a slightly modified version of slope()
        that also use the gaussian distribution.
        """

    row = len(heights)
    col = len(heights[0])
    cx, cy = center

    for group in shape:
        for x, y in group:
            dx = x - cx
            dy = y - cy
            dist = math.sqrt(dx * dx + dy * dy)

            if dist <= radius:
                if dist <= radius * 0.5:
                    # Flat crater floor
                    heights[y][x] = crater_floor
                else:
                    # Rim with smooth rise
                    norm_dist = (dist - radius * 0.5) / (radius * 0.5)
                    rim = rim_height * math.exp(-((norm_dist - 1) ** 2) / (2 * sigma ** 2))
                    heights[y][x] = crater_floor + rim

    return heights

def layer(main, secondary, p_main=0.7, p_sec=0.3):
    """ This methods layers a secondary matrix on top of the main one at different frequencies specified
        by p_main and p_sec. """
    row = len(main)
    col = len(main[0])

    for y in range(row):
        for x in range(col):
            main[y][x] = max(p_main * main[y][x] + p_sec * secondary[y][x],3)

def relief(blobs, depths, shapes, max_depths):
    """ Randomly reates relief for the land masses in the blobs matrix and return a heightmap """
    row = len(blobs)
    col = len(blobs[0])

    heights = [[0 for _ in range(col)] for _ in range(row)]

    land = shapes[0]        # [ shape 1 = [[borders],[in_land]], shape 2 = [[borders],[in_land]], shape x = ...]
    max_depths_land = max_depths[0]
    max_depth_land = max(max_depths_land)

    heights = [[0 for _ in range(col)] for _ in range(row)]
    mainland = max_depths_land.index(max_depth_land)
    biggest_land = land[mainland] 
    
    # to not have an oversized peak
    # works for heightmap size 30x30 to 100x100
    # have to have the proportion if bigger to get a correct range
    highest = random.randint(int((min(row, col)/100.0))* 40, 40)

    # random start
    repeat = True
    while repeat:
        start = random.choice(biggest_land[0] + biggest_land[1]) 
        x, y = start
        if depths[y][x] >= 0.25*max_depth_land:
            repeat = False

    # start = random.choice(biggest_land[0] + biggest_land[1]) 
    # x, y = start
    max_height = (random.randint(25,100)/100.0)*highest
    min_height = (random.randint(15,25)/100.0)
    sigma = math.sqrt(-max_depth_land**2 / (2 * math.log(min_height)))
    slope(heights, start, biggest_land, sigma, max_height, 10)

    for i in range(len(land)):
        temp = [[0 for _ in range(col)] for _ in range(row)]
        if i == mainland:
            max_height = 40
            start = random.choice(land[i][0] + land[i][1])
            slope(temp, start, land[i], max_depth_land, max_height, 4)
            layer(heights, temp)

            for j in range(4):
                max_height /= 2
                start = random.choice(land[i][0] + land[i][1])
                crater(temp, start, land[i], max_depth_land/4, 2, highest/2, 4)
                layer(heights, temp)

        else:
            h = 0
            if len(land[i][0] + land[i][1]) > 0.2*row*col:
                h = highest*0.5
            p = (random.randint(15,25))/100.0
            max_height = max(highest*p, max_depths_land[i], h)
            start = random.choice(land[i][0] + land[i][1])
            slope(heights, start, land[i], max_depth_land, max_height, 4)
    
    ocean = shapes[1]
    for bodies in ocean:
        for x, y in bodies[0] + bodies[1]:
            heights[y][x] = 2

    return heights

def peak(heights):
    """ Finds the highest point in the matrix """
    row = len(heights)
    col = len(heights[0])

    highest = 0
    for y in range(row):
        for x in range(col):
            highest = max(highest, heights[y][x])
    return highest

def landscape(file="mesh.scad", style=1):
    """ Generates the landscape """
    ocean = random.randint(60,70)/100.0
    row = random.randint(30,100)
    col = random.randint(30,100)

    dimension = f"Heightmap format: {row}x{col}"
    print(dimension)
    ocean_dimension = f"Ocean size: {int(ocean*row*col)} cells " 
    ocean_dimension += f"of {row*col} cells (approx. {ocean:.2f}%)"
    print(ocean_dimension)

    blobs = [[0 for _ in range(col)] for _ in range(row)]

    print("Making shapes...")
    blob(blobs, ocean)
    shapes = borders(blobs)
    num_land = f"Landscape has approx. {len(shapes[0])} islands."
    print(num_land)

    print("Calculating depths...")
    depths, max_depths = depth(blobs, shapes)

    print("Making relief...")
    heights = relief(blobs, depths, shapes, max_depths)

    print("Applying perlin noise for a natural touch...")
    noise(heights)

    print("Smoothing heights...")
    ker = kernel(10, 1)
    heights = blur(heights, ker)

    print(f"Writing {file} file")
    highest = peak(heights)
    max_ocean_depth = min(max_depths[1])
    if style == 1:
        scad(polyhedral(heights, highest, depths, max_ocean_depth), [row,col], highest, file, style)
    else:
        scad(minecraft(heights, highest, depths, max_ocean_depth), [row,col], highest, file)

    print("Landscape done!")

landscape("landscape.scad")

#visual.show_depth_matrix(d)
