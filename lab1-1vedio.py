import cv2
import math
import numpy as np

class MotionDetect(object):
    """docstring for MotionDetect"""
    def __init__(self, shape):
        super(MotionDetect, self).__init__()

        self.shape = shape
        self.avg_map = np.zeros((self.shape[0], self.shape[1],self.shape[2]), np.uint8)
        self.alpha = 0.8 # you can ajust your value
        self.threshold = 40 # you can ajust your value

        print("MotionDetect init with shape {}".format(self.shape))

    def getMotion(self, img):
        assert img.shape == self.shape, "Input image shape must be {}, but get {}".format(self.shape, img.shape)

        #灰階化
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #模糊化
        gray = cv2.blur(gray,(4,4))


        
        # Extract motion part (hint: motion part mask = difference between image and avg > threshold)
        moving = cv2.absdiff(self.avg_map.astype(np.uint8), gray.astype(np.uint8))

        # Mask out unmotion part (hint: set the unmotion part to 0 with mask)
        moving_map = np.zeros((img.shape[0], img.shape[1], img.shape[2]), np.uint8)
        height = img.shape[0]
        width = img.shape[1]

        for i in range(0, height):
            for j in range(0, width):
                if moving[i, j].sum() < self.threshold: #移動量比設定值小，表示沒有移動
                    moving_map[i, j] = [0, 0, 0]
                else:
                    moving_map[i, j] = img[i, j]
                    

        # Update avg_map
        self.avg_map = self.avg_map * self.alpha + img * (1 -self.alpha)

        return moving_map


# ------------------ #
#  Video Read/Write  #
# ------------------ #
name = "../data.mp4"
# Input reader
cap = cv2.VideoCapture(name)
fps = cap.get(cv2.CAP_PROP_FPS)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Output writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output1.avi', fourcc, fps, (w, h), True)

# Motion detector
mt = MotionDetect(shape=(h,w,3))

# Read video frame by frame
while True:
    # Get 1 frame
    success, frame = cap.read()

    if success:
        motion_map = mt.getMotion(frame)

        # Write 1 frame to output video
        out.write(motion_map)
    else:
        break

# Release resource
cap.release()
out.release()
