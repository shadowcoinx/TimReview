import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse

cap = cv2.VideoCapture('Resources/2020-12-02 19-43-31_unfish.mkv')


if not cap.isOpened():
    print("Error opening video")

while(cap.isOpened()):

    w = cap.get(3)
    h = cap.get(4)
    frameArea = h * w
    areaTH = frameArea / 280

    status, frame = cap.read()
    status, frame2 = cap.read()

    def ThresholdIMAGE(frame):


        # Difference between frames
        diff = cv2.absdiff(frame, frame2)

        # Converting to grayscale
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Blurring grayscale image
        blur = cv2.GaussianBlur(gray, (1,1), 0)

        # Thresholding
        _, thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)

        # Dilating
        dilated = cv2.dilate(thresh, None, iterations=3)

        return dilated


    dilated = ThresholdIMAGE(frame)


    def FindDrawContours(frame, frame2, h, w, frameArea, areaTH):

        # Finding contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            up_limit = int(1 * (h / 7))
            down_limit = int(4 * (h / 7))
            area = cv2.contourArea(contour)
            if area > areaTH:
                m = cv2.moments(contour)
                cx = int(m['m10'] / m['m00'])
                cy = int(m['m01'] / m['m00'])
                (x, y, w, h) = cv2.boundingRect(contour)


                if cy in range(up_limit, down_limit):
                    # If area of contour is less than 700
                    if area < 1000:
                        continue


                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


        # Drawing contours to original frame, -1 applies to every contour
        # cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

        frame = frame2

        # status, frame2 = cap.read()

        return frame

    FindDrawContours(frame, frame2, h, w, frameArea, areaTH)
    cv2.imshow('s', dilated)
    cv2.imshow('as', frame)


    def final(frame):

        # Set frame to 4 channel, same as overlay
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        # Determine frame sizes for blank image
        frame_h, frame_w, frame_c = frame.shape

        # Draw blank image and draw box overlay inside it
        overlay1 = np.zeros((frame_h, frame_w, 4), dtype='uint8')
        pts = np.array([[300, 395], [82, 470], [600, 710], [730, 499]])
        box = cv2.fillConvexPoly(overlay1, pts, (110, 51, 45))

        # Canny detect edges and fill any holes
        edges = cv2.Canny(overlay1, 105, 220)
        edges = cv2.dilate(edges, None)
        edges = cv2.erode(edges, None)

        # Finds contours from canny image
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # Convert border overlay to 4 channel
        overlayBoxBorder = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGRA)

        # Variables to drawing border
        contour_id = 0
        border_thickness = 2
        border_color = (185, 115, 72)

        # Draw border around detected canny edges
        overlayBoxBorder = cv2.drawContours(overlayBoxBorder, contours, contour_id, border_color, border_thickness)

        # Add together overlay border and box fill
        overlay1 = cv2.addWeighted(overlayBoxBorder, 1, box, 1.1, 0)

        # Alpha channel value for overlay
        alpha = 0.5

        # Add overlay box over the road to video
        image_new = cv2.addWeighted(overlay1, alpha, frame, 1.1, 0)

        return image_new



    image_new = final(frame)


    if status:
        cv2.imshow('Output', image_new)
    key = cv2.waitKey(999999999)

    if key == 32:
        cv2.waitKey()
    elif key == ord('q'):
        break














