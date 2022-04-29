import cv2

# Opens the Video file
cap= cv2.VideoCapture('/home/ninad/comco_mar14_decan.mp4')
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    if i%500==0:
        cv2.imwrite('decan_frames/comco_belgaum_decan/comco_belgaum_decan14_'+str(i)+'.jpg',frame)
    i+=1

cap.release()
cv2.destroyAllWindows()

