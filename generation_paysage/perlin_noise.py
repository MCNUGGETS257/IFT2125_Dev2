import random

class Vect2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def dot(self, other):
        return self.x*other.x +self.y*other.y

# O(n) shuffling algorithm
def shuffle(array):

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

def fbm(x, y, numOct):
    result = 0.0
    amp = 5.0
    freq = 0.005

    for i in range(numOct):
        n = amp*noise2D(x*freq, y*freq)
        result += n

        amp *= 0.5
        freq *= 2.0

    return result


def inverse_lerp(a, b, value):
    if a == b:
        return 0.0  # Avoid division by zero
    return (value - a) / (b - a)


def generate(array):
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


