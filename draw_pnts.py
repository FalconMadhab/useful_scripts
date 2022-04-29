import cv2

image = cv2.imread("/home/ninad/hpcl_images_labelme/ABSENCE_OF_BIKE_CONES.jpg")
image = cv2.circle(image, (int(612.1951219512196),int(297.0650406504065)), radius=2, color=(0, 255, 255), thickness=2)
image = cv2.circle(image, (int(802.439024390244),int(284.0569105691057)), radius=2, color=(0, 255, 255), thickness=2)
image = cv2.circle(image, (int(982.9268292682927),int(453.9756097560976)), radius=2, color=(0, 255, 255), thickness=2)
image = cv2.circle(image, (int(681.30081300813),int(521.4552845528456)), radius=2, color=(0, 255, 255), thickness=2)

cv2.imwrite("ress.jpg",image)