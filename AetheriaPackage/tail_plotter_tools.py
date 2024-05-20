import numpy as np
import matplotlib.pyplot as plt

from basic_functions import Linear

rect_corner_basis = np.array([[1,  1,  0],
                              [1,  1, -1],
                              [1, -1, -1],
                              [1, -1, 0]])


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().

    modified from https://stackoverflow.com/a/31364297 and
    https://github.com/matplotlib/matplotlib/issues/17172/
        
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    try:
        ax.set_aspect('equal')
    except NotImplementedError:
        pass

def plot_rectangle(ax, x, h, w, percent=None, **style):
    '''
    Given an array of x, h and w, with h and w being the heights and widths of the rectangles along
    the length x, plot the rectangle at a particular percentage [0,1] along the length.
    '''
    if percent is not None:
        i = int(percent*(np.size(x)-1))
    else:
        x = [x]
        h = [h]
        w = [w]
        i = 0

    scaling = np.array([x[i], w[i]/2, h[i]])
    p1 = scaling * rect_corner_basis[0]
    p2 = scaling * rect_corner_basis[1]
    p3 = scaling * rect_corner_basis[2]
    p4 = scaling * rect_corner_basis[3]

    rect = np.vstack((p1, p2, p3, p4, p1)).swapaxes(0, 1)

    ax.plot(rect[0], rect[1], rect[2], **style)


def plot_tail(ax, x, h, w):
    '''
    Plot the tail, given an array of x, h and w.
    '''
    line_style = {'lw': 0.8} #TODO import this in parameters
    sh = np.shape(x)

    for corner in rect_corner_basis:
        xi, yi, zi = corner
        ax.plot(x*xi, w/2*yi, h*zi, **line_style, color='black')

    # rectangle at the beginning of the tail
    plot_rectangle(ax, x, h, w, 0, **line_style, color='green')
   
    # rectangle at the end of tail
    plot_rectangle(ax, x, h, w, 1, **line_style, color='red')


def plot_cuboid(ax, x, h, w, **style):
    '''
    Plot a cuboid of length x, height h, width h
    '''
    if len(style)==0:
        style = {'lw':0.8, 'color':'purple'}

    # plot lines in the tailwise direction
    lin = np.ones(2)
    for corner in rect_corner_basis:
        xi, yi, zi = corner
        ax.plot([0, x], w/2*yi*lin, h*zi*lin, **style)
    
    # rectangle at the beginning of the tail
    plot_rectangle(ax, 0, h, w, **style)
    
    # plot rectangle at the end of the cuboid
    plot_rectangle(ax, x, h, w, **style)

def plot_cylinder(ax, r, l, x0, y0, z0, caps=True, n=2, res=0.01, **style):
    '''
    Plot a cylinder of radius r, length l at x, y, z.
    If caps is set to True, a set of spherical caps will be drawn within the cylinder.
    n specifies the number of drawn sides (by default True)
    '''
    assert r>0, f'Negative radius={r} not possible!'
    assert l>2*r, f'Cylinder geometry impossible: l<2r'
    

    lon = np.arange(0, l+res, res)
    shp = np.shape(lon)
    rad = r*np.ones(shp)

    if caps:
        rad[lon<r]     = np.sqrt(r**2-(lon[lon<r]-r)**2)
        rad[lon>(l-r)] = np.sqrt(r**2-(lon[lon>(l-r)]-(l-r))**2)
    else:
        phi = np.arange(0, 2*np.pi+res, res)
        for x_init in [x0, x0+l]:
            x = x_init*np.ones(np.shape(phi))
            y = r*np.cos(phi)
            z = r*np.sin(phi)
            ax.plot(x, y, z, **style)

    theta_lst = np.arange(0, 2*np.pi, 2*np.pi/(2*n))

    for theta in theta_lst:
        transform = np.matrix([[            0, 1],
                               [np.cos(theta), 0],
                               [np.sin(theta), 0]])
        
        coord = np.matrix([rad, lon])
        x, y, z = np.array(transform*coord) # matrix multiplication (and then cast back into array)

        ax.plot(x+x0, y+y0, z+z0, **style)


def plot_complete_tail(l_tail, l_tank, h0, b0, hc, bc, hf, bf, r_tank, linear_rel, n_lines=20, ax=None):
    tank_style = {'color': 'purple', 'lw': 0.8}
        
    x = np.linspace(0, l_tail, 500)

    if ax is None:
        ax = plt.figure().add_subplot(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    h = Linear(x[0], x[-1], h0, hf)(x)
    if linear_rel == 'AR':
        AR = Linear(x[0], x[-1], b0/h0, bf/hf)(x)
        w = AR*h
    elif linear_rel == 'b':
        w = Linear(x[0], x[-1], b0, bf)(x)

    plot_cylinder(ax, r=r_tank, l=l_tank, x0=0, y0= r_tank, z0=-r_tank, **tank_style, n=n_lines)
    plot_cylinder(ax, r=r_tank, l=l_tank, x0=0, y0=-r_tank, z0=-r_tank, **tank_style, n=n_lines)
    plot_rectangle(ax, x, h, w, l_tank/l_tail, color='orange')
    plot_cuboid(ax, l_tank, hc, bc, color='red', ls='--')
    plot_tail(ax, x, h, w)

    set_axes_equal(ax)

if __name__ == '__main__':

    # ax = plt.figure().add_subplot(projection='3d')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')

    # plot_cylinder(ax,   0.18106105610457982, 4.97, 0, 0, 0, n=50, caps=True, color='purple', lw=0.8)
    # set_axes_equal(ax)

    plot_tail(5, 2, 1.5, 1.5, 0.4, 0.4, 0.01, 0.01, 0.2, 'b')

    plt.show()