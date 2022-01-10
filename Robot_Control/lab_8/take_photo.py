#!/usr/bin/env python3

import cv2
from cars_camera import Camera, RESOLUTIONS
import os

def main():

    cv2.namedWindow("calibration")
    cam = Camera()

    i = 0
    dir = 'robot_control\\camera_calibration\\calib_photos'

    while True:
        cam.keep_stream_alive()
        img = cam.get_frame()

        cv2.imshow("demo", img)
        keypress = cv2.pollKey() & 0xFF
        
        if keypress == ord('q'):
            break
            
        elif keypress == ord('+'):
            q = cam.get_quality()
            if not q == max(RESOLUTIONS.keys()):
                cam.set_quality(q + 1)
                
        elif keypress == ord('-'):
            q = cam.get_quality()
            if not q == min(RESOLUTIONS.keys()):
                cam.set_quality(q - 1)
                
        elif keypress % 256 == 32:
            base_filename = str(i)
            filename = os.path.join(dir, base_filename + '.jpg')
            cv2.imwrite(filename, img)
            i += 1

if __name__ == "__main__":
    main()
