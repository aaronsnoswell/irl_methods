"""
Various plotting utilities

(c) Aaron Snoswell 2018
"""


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def plot_trajectory_4d(trajectory, title, duration):
    """Plot a trajectory of 3D points with color indicating time

    Args:
        trajectory (numpy array): Numpy array of shape mx3, where m is the number of
            3d points to plot
        title (string): Plot title
        duration (float): Duration in seconds of the trajectory (used to give
            the colorbar a legend)

    Returns:
        Nothing
    """

    # Get a color map object
    cmap = cm.get_cmap("viridis")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Number of color segements to split the trajectory into
    segs = 100
    seg_length = len(trajectory) / segs
    for j in range(segs):

        start = int(j*seg_length)
        stop = int((j+1)*seg_length)
        ax.plot(
            xs=trajectory[start:stop, 0],
            ys=trajectory[start:stop, 1],
            zs=trajectory[start:stop, 2],
            color=cmap(float(j / segs))
        )

    _colormappable = cm.ScalarMappable(cmap=cmap)
    _colormappable.set_array([0, duration])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    cbar = plt.colorbar(mappable=_colormappable, ax=ax)
    cbar.set_label('Time (s)')
    plt.title(title)
    plt.show()
