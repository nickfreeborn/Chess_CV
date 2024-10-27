import cv2
from matplotlib import pyplot as plt

im = cv2.imread('test_images/image1.jpg')
im_colour = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

im_gray_blur = cv2.GaussianBlur(im_gray, (5,5), 0) 


sobel_edges_x = cv2.Sobel(im_gray_blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_edges_y = cv2.Sobel(im_gray_blur, cv2.CV_64F, 0, 1, ksize=3)
canny_edges = cv2.Canny(im_gray_blur, 100, 100)
sobel_magnitude = ((sobel_edges_x)**2+(sobel_edges_y)**2)**(1/2)


plt.figure(figsize = (20,80))
plt.subplot(131)
plt.imshow(im_gray)
plt.title('Image')
plt.xticks([]), plt.yticks([])

plt.subplot(132)
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny Edges')
plt.xticks([]), plt.yticks([])

plt.subplot(133)
plt.imshow(sobel_magnitude, cmap='gray')
plt.title('Sobel Magnitude')
plt.xticks([]), plt.yticks([])

plt.show()