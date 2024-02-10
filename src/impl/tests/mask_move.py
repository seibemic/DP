import numpy as np
import cv2  # You may need to install OpenCV using: pip install opencv-python

def move_mask(mask, dx, dy):
    rows, cols = mask.shape[:2]
    print(rows,cols)

    # Create an empty mask of the same size
    moved_mask = np.zeros_like(mask)

    # Define the transformation matrix for translation
    M = np.float32([[1, 0, dx], [0, 1, dy]])

    # Apply the translation to the mask
    moved_mask = cv2.warpAffine(mask, M, (cols, rows))

    return moved_mask

if __name__ == "__main__":
    # Example usage:
    # Load your mask image
    img = cv2.imread('/home/michal/Documents/FIT/DP/dp/imgs/people.jpeg')
    print(img.shape)
    mask = np.zeros_like(img.transpose(2,0,1)[0])
    mask[20:50, 40:60] = 1
    print(mask.shape)
    # Specify the amount to move in x and y directions
    dx = 20  # Change this to your desired value
    dy = 30  # Change this to your desired value

    # Move the mask
    moved_mask = move_mask(mask, dx, dy)

    # Display the original and moved masks
    cv2.imshow('Original Mask', mask)
    cv2.imshow('Moved Mask', moved_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
