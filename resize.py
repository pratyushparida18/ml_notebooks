import cv2

img = cv2.imread('/home/pratyush/Downloads/download.jpeg')

scale_percent = 0.5
width = int(img.shape[1]*scale_percent)
height = int(img.shape[0]*scale_percent)
dimension = (width,height)

resized = cv2.resize(img,dimension,interpolation=cv2.INTER_LANCZOS4)


print(resized.shape)

cv2.imshow('input',img)
cv2.imshow('output',resized)
cv2.imwrite('resized_image.jpg',resized)

cv2.waitKey(0)
cv2.destroyAllWindows()