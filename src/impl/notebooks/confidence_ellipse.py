import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import cv2
def confidence_ellipse(loc, cov, ax, n_std=3.0, facecolor='none', **kwargs):

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = loc[0]

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = loc[1]
    print(scale_x, scale_y)
    print(ell_radius_x, ell_radius_y)
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)


    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def cv2_confidence_ellipse(center, cov_matrix, image, n_std=3.0, color=(0, 0, 255), thickness=1):
    # Extracting the 2x2 covariance matrix
    cov_2x2 = np.float32(cov_matrix[:2, :2])


    # Calculating eigenvalues and eigenvectors
    _, eigenvalues, eigenvectors = cv2.eigen(cov_2x2)

    # Extracting major and minor axes lengths
    major_axis_length = n_std * np.sqrt(eigenvalues[0])
    minor_axis_length = n_std * np.sqrt(eigenvalues[1])

    # Extracting the rotation angle in radians
    angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])

    # Convert angle to degrees
    angle_deg = np.degrees(angle)

    # OpenCV ellipse parameters
    center = (int(center[0]), int(center[1]))
    axes = (int(major_axis_length), int(minor_axis_length))

    # Draw ellipse using cv2.ellipse
    ellipse = cv2.ellipse(image, center, axes, angle_deg, 0, 360, color, thickness)
    return ellipse
