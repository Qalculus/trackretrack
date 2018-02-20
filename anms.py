import numpy as np

def anms_sdc(keypoints, k, image_size, *, eps_r=0.25, eps_k=0.1):
    """Adaptive non-maxima suppression using disk coverage

    Implements the method by Gauglitz et.al. in
    "Efficiently selecting spatially distributed keypoints for visual tracking"

    Keypoints are OpenCV keypoints with response value set.
    """
    if len(keypoints) < k:
        return keypoints

    keypoints.sort(key=lambda kp: kp.response, reverse=True) # Largest first
    max_iter = 2 * int(np.floor(np.log2(max(image_size))))
    xy = np.vstack([kp.pt for kp in keypoints])

    left = 0
    right = max(image_size) - 1

    class Grid:
        def __init__(self, r):
            self.c = eps_r * r / np.sqrt(2) # cell width
            self.r = r
            height, width = image_size
            nrows = int(np.ceil(height / self.c))
            ncols = int(np.ceil(width / self.c))
            self.grid = np.zeros((nrows, ncols), dtype='bool')

            # Create mask for coverage
            self.patch_radius = p = int(np.floor(self.r / self.c))

        def cell_coords(self, x, y):
            cx = int(np.floor(x / self.c))
            cy = int(np.floor(y / self.c))
            return cy, cx

        def is_covered(self, p):
            x, y = p.pt
            return self.grid[self.cell_coords(x, y)]

        def cover(self, p):
            cy, cx = self.cell_coords(*p.pt)
            height, width = self.grid.shape
            xmin = max(0, cx-self.patch_radius)
            xmax = min(width-1, cx + self.patch_radius)
            ymin = max(0, cy-self.patch_radius)
            ymax = min(height-1, cy + self.patch_radius)
            mx, my = np.meshgrid(range(xmin,xmax+1), range(ymin,ymax+1))
            distance_squared = (mx - cx)**2 + (my - cy)**2
            cells_per_radius = self.r / self.c
            mask = distance_squared < cells_per_radius**2
            self.grid[my[mask], mx[mask]] = 1

    for it in range(max_iter):
        r = 0.5 * (left + right)
        grid = Grid(r)
        result = []
        for kp in keypoints:
            if not grid.is_covered(kp):
                result.append(kp)
                grid.cover(kp)

        #print('Found {:d}'.format(len(result)))
        #print('{:3d} {:5d} {:.2f}'.format(it, len(result), r))
        if k <= len(result) <= (1 + eps_k)*k:
            #print('Solution found on iteration', it+1)
            return result[:k]
        elif len(result) < k:
            right = r
            #print('Too few, decreasing radius by setting right={:.2f}'.format(r))
        else:
            left = r
            #print("Too many, increasing radius by setting left={:.2f}".format(r))

    raise ValueError("Reached max iterations without finding solution")

