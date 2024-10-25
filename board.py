import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import pickle
import cv2
import numpy as np
import os

import matplotlib
matplotlib.use('TkAgg')

def plot_scatter(img):

    # Convert to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Create array of pixel colours (in format and scaling needed for matplotlib scatter plot)
    pixel_colours = img.reshape((img.shape[0]*img.shape[1], 3))
    norm = colors.Normalize(vmin=-1.0, vmax=1.0)
    norm.autoscale(pixel_colours)
    pixel_colours = norm(pixel_colours).tolist()

    # # Visualize the colours in an RGB colour space
    H, S, V = cv2.split(img_hsv)

    # Display scatter plot
    fig = plt.figure()  # Create another figure for the 3D plot
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.scatter(H.flatten(), S.flatten(), V.flatten(), facecolors=pixel_colours, marker='.')
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    axis.view_init(30, 70, 0) # (elevation, azimuth, roll): try adjusting to view from different perspectives

    plt.show()

    return


def threshold(img, lower_hsv, upper_hsv):

    # Convert to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Create mask of pixels inside the range of lower and upper colours
    mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)

    # Apply mask to visualize
    img_threshold = cv2.bitwise_and(img, img, mask=mask)

    # Optional visualisation which is useful for guaging threshold robustness
    plt.figure()  # Create a new figure for the mask and applied image

    plt.subplot(121)
    plt.imshow(img_hsv)
    plt.xlabel('HSV')

    plt.subplot(122)
    plt.imshow(mask, 'gray')
    plt.xlabel('Mask Image')

    # plt.subplot(123)
    # plt.imshow(img_threshold)
    # plt.xlabel('Applied to original image')

    plt.tight_layout()
    plt.show()

    return img_threshold


# Displays all 15 boards at once. Useful for checking intermediate filtering steps
def show_boards():

    # Create grid of plots
    fig, axes = plt.subplots(3, 5, figsize=(15, 15))  # 5x3 grid
    plt.subplots_adjust(wspace=0.005, hspace=0.11)

    num_cols = 7

    # show each board in its respective position
    for i in range(15):
        ax = axes[i // 5, i % 5]

        height, width, depth = boards[i].shape

        num_cols = 7
        num_rows = 6

        # Draw vertical grid lines
        for col in range(1, num_cols):
            ax.axvline(x=col*(width/num_cols), color='pink', linestyle='--', linewidth=1)

        # Draw horizontal grid lines
        for row in range(1, num_rows):
            ax.axhline(y=row*(height/num_rows), color='pink', linestyle='--', linewidth=1)

        ax.imshow(boards[i])
        ax.axis('off')
        ax.set_title(f'Image {i+1}', loc='left')

    plt.show()


def order_points(points):
    
    # Initialize the ordered corner points
    corrected = np.zeros((4, 2), dtype='float32')

    # The top-left point will have the smallest sum, the bottom-right the largest sum
    sum = points.sum(axis=1)
    corrected[0] = points[np.argmin(sum)]
    corrected[3] = points[np.argmax(sum)]

    # The top-right point will have the smallest difference, the bottom-left will have the largest difference
    diff = np.diff(points, axis=1)
    corrected[1] = points[np.argmin(diff)]
    corrected[2] = points[np.argmax(diff)]

    return corrected


def is_yellow(pxl):

    # Make sure it has enough red colour >50 (so that the blue board does not pass as 'red')
    # Yellow is characterised by not much blue <50
    # But sufficient green >55 to make sure it's not a red token!
    yellow_rgb_lower = (50, 55, 0) 
    yellow_rgb_upper = (255, 255, 43) 
    # 147 91  44
    # 176 189 51

    lower_check = pxl[0] > yellow_rgb_lower[0] and pxl[1] > yellow_rgb_lower[1] and pxl[2] > yellow_rgb_lower[2]
    upper_check = pxl[0] < yellow_rgb_upper[0] and pxl[1] < yellow_rgb_upper[1] and pxl[2] < yellow_rgb_upper[2]

    return lower_check & upper_check


def is_red(pxl):

    # Make sure it has enough red colour >50 (so that the blue board does not pass as 'red')
    # Red is characterised by not much blue nor green <55
    red_rgb_lower = (50, 0, 0)
    red_rgb_upper = (255, 55, 55)

    lower_check = pxl[0] > red_rgb_lower[0] and pxl[1] > red_rgb_lower[1] and pxl[2] > red_rgb_lower[2]
    upper_check = pxl[0] < red_rgb_upper[0] and pxl[1] < red_rgb_upper[1] and pxl[2] < red_rgb_upper[2]

    return lower_check & upper_check


def get_state(board):

    # Initialise output array
    state = np.zeros((6, 7))

    # Calculate the locations of each possible red or yellow token
    token_locs = np.zeros((6, 7, 2))
    height, width, depth = np.shape(board)
    num_rows = 6
    num_cols = 7
    row_spacing = height // num_rows
    col_spacing = width // num_cols

    for row in range(num_rows):
        for col in range(num_cols):
            token_locs[row, col, 0] = int(row*row_spacing + int(row_spacing//2)) # we add spacing/2 to get the CENTRE of each cell, not just its corner
            token_locs[row, col, 1] = int(col*col_spacing + int(col_spacing//2))

    # Determine the token colour
    for row in range(num_rows):
        for col in range(num_cols):
            x = token_locs[row, col, 0]
            y = token_locs[row, col, 1]
            pxl = board[int(x)][int(y)]

            if (is_yellow(pxl)):
                state[row, col] = 1

            elif (is_red(pxl)):
                state[row, col] = 2

            else:
                state[row, col] = 0

    return state


# Our meaty function
# Digests an image of a connect 4 board, and reads the board to understand the state of the game and its cells
# Takes an array of pixels (heigh, width, depth), and returns:
    # board: the input image but manipulated and warped to show how the function has thesholded and projected the board
    # state: a numpy array for which colours occupy each cell in the board. Empty = 0, yellow = 1, red = 2
    # corners: a numpy array (4, 2) listing the corners from top-left, top-right, bottom-left, bottom-right and their x and y coordinates
def interpret_image(img):

    # Make a copy for intermittent filters
    board = img

    # Visually determine threshold values by observing the HSV scatter plot
    # plot_scatter(img)

    # Threshold each image using maticulously hand-picked values
    hsv_lower = (11, 125, 0) # 10, 6
    hsv_upper = (16, 255, 155) # 16, 22
    board = threshold(img, hsv_lower, hsv_upper)

    # Find contours to the thresholded image
    gray_board = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray_board, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # The largest contour will be the outline of the board
    largest_contour = max(contours, key=cv2.contourArea)

    # The board is rectangular, so approximate the contour as a quadrilateral
    epsilon = 0
    approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
    # while(len(approx_polygon) != 4):
    #     approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
    #     epsilon += 1 # increase distance from contour-edge to approx-polygon-edge until a quadrilateral is found

    # Optional drawing functions which will display the largest contour found, or the polygon which the board has been approximated to
    cv2.drawContours(img, [largest_contour], 0, (255,0,0), 2)
    plt.imshow(img)
    plt.show()
    # cv2.drawContours(img, [approx_polygon], 0, (255,0,0), 30)

    # Find the current board corners and desired board corner positions in order: top-left, top-right, bottom-left, bottom-right
    width, height = 350, 300  # A 7x6 shaped rectangle to keep hole proportions correct
    corners = order_points(np.array([point[0] for point in approx_polygon]))
    endpoints = np.array([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]], dtype='float32')

    # Calculate transformation matrix
    transform = cv2.getPerspectiveTransform(corners, endpoints)

    # Project the board to a rectangle, taking up the entire figure
    board = cv2.warpPerspective(img, transform, (width, height))

    # Read the newly transformed board to understand the cell states
    state = get_state(board)

    # return img, state, corners
    return img, state, corners


# For some reason, the ground truth corners appear to be off by a factor of 1/2
# Take '001.jpg' in the ground truths for example: the bottom right corner is supposedly at x=982, y=1081
# As the image is in fact 3472 pixels high, this implies that the corner is not even a third of the way down the image
# By visually inspecting 001.jpg, you can see this is wrong: The corner is clearly in the bottom half of the image (y >= 1736)

# Perhaps I have interpreted the ground truths wrong. In any case, the mismatch is solved as long as
# I multiply all ground truth values by 2.

def assess_corners(estimate, truth):

    # Correction needed. See above
    correction_factor = 2

    # The corner will not be found exactly accurately
    # Create a parameter to allow some leniancy in the accuracy test
    margin = 30

    # Test for each corner
    # If at any point a difference is outside the allowed margin, the corner test for this board has failed immediately
    for i in range(4):
        x_diff = estimate[i, 0] - correction_factor * truth[i][0]
        y_diff = estimate[i, 1] - correction_factor * truth[i][1]
        if (abs(x_diff) > margin): return False
        if (abs(y_diff) > margin): return False

    return True


def assess_states(estimate, truth):

    # Make a tally for counting how many are right
    num_correct = 0

    # Tally the number of correct cells in the board
    for x in range(6):
        for y in range(7):
            if (estimate[x, y] == truth[x][y]):
                num_correct += 1

    # Return a percentage accuracy
    ret = num_correct / (6*7)

    return ret


def display_results(corner_accuracies, state_accuracies):

    # Display results for each image
    print()
    print("RESULTS:")
    print("img no.      CORNERS FOUND?  CELL ACCURACY (%)")
    for i in range(15):
        print(f"{i+1:03d}.jpg:     {bool(corner_accuracies[i])}            {state_accuracies[i]:.3f}")

    # Calculate average and overall accuracy
    average_accuracy = sum(state_accuracies) / 15
    overall_accuracy = sum(state_accuracies == 1.) / 15

    # Print neatly
    print()
    print(f"Average accuracy: {average_accuracy:.3f}")
    print(f"Overall accuracy: {overall_accuracy:.3f}")
    print()

    return

# Load each image of a board
imgs = []
imgs_dir_path = 'Chess_CV\data'
for filename in os.listdir(imgs_dir_path)[:40]:
    if filename.endswith('.png'):
        file_path = os.path.join(imgs_dir_path, filename)
        img = cv2.imread(file_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)

print(len(imgs))

# Initialise arrays for storing board states, and board corners: the results
board_states  = np.zeros((15, 6, 7))
board_corners = np.zeros((15, 4, 2))

# Analyse each image to get the board corner positions, and board state
boards = [] # used for visualising intermittent steps
for i in range(15):
    # plt.imshow(imgs[i])
    # plt.show()
    processed_board, board_states[i], board_corners[i] = interpret_image(imgs[i])
    boards.append(processed_board)
show_boards()
