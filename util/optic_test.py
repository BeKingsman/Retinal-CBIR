import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import os


for name in os.listdir("images"):
    image = cv2.imread('images/'+name)
    template = cv2.imread('optic_nerve.png')
    heat_map = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

    h, w, _ = template.shape
    y, x = np.unravel_index(np.argmax(heat_map), heat_map.shape)

    cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 5)

    img=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imshow("Converted",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

