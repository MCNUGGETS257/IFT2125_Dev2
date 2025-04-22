import random
from collections import deque
import math
import perlin_noise as perlin
import visual
import scad_2



def print_matrix(matrix):
    # Print the matrix
    for row in matrix:
        print(' '.join(map(str, row)))

ocean = random.randint(25,40)/100.0
row = random.randint(30,100)
col = random.randint(30,100)

print(row,col)
print(ocean)

drawn = []

# neighbours
def neighbours(x, y, row, col, numNeighbours=8):
    #directions += [[-2,-2],[0,-2],[2,-2],[2,0],[2,2],[0,2],[-2,2],[-2,0]]
    
    if numNeighbours == 4:
        directions = [[-1,0],[1,0],[0,-1],[0,1]]
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
                    drawn.append(1)
                    matrix[cell[1]][cell[0]] = type
                    to_draw -= 1
                    continue
                else:
                    radius = 2
                    x = random.randint(max(0, x - radius), min(col - 1, x + radius))
                    y = random.randint(max(0, y - radius), min(row - 1, y + radius))
                    continue
            else:
                drawn.append(1)
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

def blob(blobs):
    row = len(blobs)
    col = len(blobs[0])

    nb_cells = int(ocean*row*col)


    x = random.randint(0,col)
    y = random.randint(0,row)
    draw(blobs, x, y, 1, nb_cells)
    
def borders(blobs):
    row = len(blobs)
    col = len(blobs[0])
    visited = [[False for _ in range(col)] for _ in range(row)]
    shapes = [[],[]]

    for y in range(row):
        for x in range(col):
            if visited[y][x] == False and blobs[y][x] == 0:
                shapes[0].append(shape([x,y], visited))
            if visited[y][x] == False and blobs[y][x] == 1:
                shapes[1].append(shape([x,y], visited, 1))

    return shapes

def shape(start, visited, type=0):
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
    row = len(main)
    col = len(main[0])

    for y in range(row):
        for x in range(col):
            main[y][x] = p_main * main[y][x] + p_sec * secondary[y][x]

def relief(blobs, depths, shapes, max_depths):
    row = len(blobs)
    col = len(blobs[0])

    heights = [[0 for _ in range(col)] for _ in range(row)]
    bassin = []

    land = shapes[0]        # [ shape 1 = [[borders],[in_land]], shape 2 = [[borders],[in_land]], shape x = ...]
    max_depths_land = max_depths[0]
    max_depth_land = max(max_depths_land)

    heights = [[0 for _ in range(col)] for _ in range(row)]
    mainland = max_depths_land.index(max_depth_land)
    # Pour chaque île (forme de terre), on ajoute une pente depuis un sommet
    for i, island in enumerate(land):
        # Créer une matrice temporaire
        temp = [[0 for _ in range(col)] for _ in range(row)]

        # Chercher le sommet de l’île
        island_cells = island[0] + island[1]
        summit = max(island_cells, key=lambda pos: depths[pos[1]][pos[0]])

        # Calcul d’un sigma adapté
        local_depth = max_depths_land[i]
        max_height = 25  # hauteur de départ
        sigma = local_depth / 1.5

        # Appliquer la pente depuis le sommet
        slope(temp, summit, island, sigma, max_height, 2)

        # Fusionner avec la carte finale
        layer(heights, temp, p_main=0.6, p_sec=0.4)

    biggest_land = land[mainland] 
    
    # to not have an oversized peak
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
                bassin.append(start)

        else:
            max_height = max(4, max_depths_land[i])
            start = random.choice(land[i][0] + land[i][1])
            slope(heights, start, land[i], max_depth_land, max_height, 4)
    
    ocean = shapes[1]
    for bodies in ocean:
        for x, y in bodies[0] + bodies[1]:
            heights[y][x] = 2

    print("slopped")
    return [heights, bassin]




def kernel(size, sigma):
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


def noise(heights):
    row = len(heights)
    col = len(heights[0])

    noise = perlin.generate([[0 for _ in range(col)] for _ in range(row)])

    for y in range(row):
        for x in range(col):
            heights[y][x] = heights[y][x] * noise[y][x]

    


blobs = [[0 for _ in range(col)] for _ in range(row)]
blobs = [[0 for _ in range(30)] for _ in range(30)]
blob(blobs)
#print_matrix(blobs)
b = borders(blobs)
d, max_d = depth(blobs, b)
print(max_d)
heights, bassin = relief(blobs, d, b, max_d)

noise(heights)
k = kernel(10, 1)
heights = blur(heights, k)

#print(max_d)
scad_2.polyhedral(heights)
print(len(drawn))
print(len(b[0]))
print(len(b[1]))
#visual.show_depth_matrix(d)
