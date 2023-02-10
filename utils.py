
class colors:
    WHITE = (255, 255, 255)
    RED = (200,0,0)
    BLUE1 = (0, 0, 255)
    BLUE2 = (0, 100, 255)
    BLACK = (0,0,0)
    GREEN = (0, 200, 0)

class size:
    BLOCK_SIZE = 50
    CONTOUR_SIZE = 1
    SPEED = 100
    
def distance_between(a,b, rows, cols):
    return abs(a[0]-b[0])/rows + abs(b[1]-a[1])/cols
