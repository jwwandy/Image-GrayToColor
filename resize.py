import cv2

img = cv2.imread('./images/mountain_color.jpg')
img = cv2.resize(img,(400,267))
cv2.imwrite('./images/mountain_color_resize400.jpg',img)