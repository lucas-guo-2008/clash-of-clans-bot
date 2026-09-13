import cv2 
import numpy as np
import struct
import adbutils
import time

if __name__ == "__main__":
    frame = cv2.imread("template matching/ss.png")
    template = cv2.imread("template matching/attack.png")
    h, w, c = template.shape

    methods = [
        cv2.TM_SQDIFF,
        cv2.TM_SQDIFF_NORMED,
        cv2.TM_CCORR,
        cv2.TM_CCORR_NORMED,
        cv2.TM_CCOEFF,
        cv2.TM_CCOEFF_NORMED
    ]

    for method in methods:
        img2 = frame.copy()
        result = cv2.matchTemplate(img2, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        print(max_val)
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            location = min_loc
        else: location = max_loc

        cv2.rectangle(img2, location, (location[0]+w, location[1]+h), 255, 5)
        cv2.putText(img2, f"model {method} and val {min_val} {max_val}", (10, 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        cv2.imshow('Match', img2)

        if cv2.waitKey(3000) == ord("q"):
                cv2.destroyAllWindows()
