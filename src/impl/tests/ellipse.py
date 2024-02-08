import matplotlib.pyplot as plt
import numpy as np
import cv2
from confidence_ellipse import confidence_ellipse, cv2_confidence_ellipse


if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(10, 10))
    img = plt.imread("/home/michal/Documents/FIT/DP/dp/imgs/people.jpeg")
    ax.imshow(img)
    d = 100
    P = np.array([[d,-60,0,0],
         [30,2*d,0,0],
         [0,0,d,0],
         [0,0,0,d]])
    confidence_ellipse([150, 80], P, ax=ax,
                       edgecolor="red")
    plt.show()

    path = r'/home/michal/Documents/FIT/DP/dp/imgs/people.jpeg'

    # Reading an image in default mode
    image = cv2.imread(path)

    window_name="tmp"
    image = cv2_confidence_ellipse([150, 80], P, image=image)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()