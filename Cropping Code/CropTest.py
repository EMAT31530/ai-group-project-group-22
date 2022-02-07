# import OS module
import os
from PIL import Image
import matplotlib.pyplot as plt

# Get the list of all files and directories
path = r"C:\Users\OliWo\OneDrive\University\EMAT\Intro to AI\Coursework\histopathologic-cancer-detection\test"

# to store files in a list
list = []

# dirs=directories
for (root, dirs, file) in os.walk(path):
    for f in file:
        if '.tif' in f:
            list.append(f)

list.sort()

for i in range(len(list)):
    name = list[i]
    path = "C:/Users/OliWo/OneDrive/University/EMAT/Intro to AI/Coursework/histopathologic-cancer-detection/test/"
    im = Image.open(path+name)

    width, height = im.size

    # Setting the points for cropped image
    left = width/3
    top = height/3
    right = (width/3)*2
    bottom = (height/3)*2

    # Cropped image of above dimension
    # (It will not change original image)
    im1 = im.crop((left, top, right, bottom))

    file_name = str('test/test'+str(i))

    im.save(file_name+'.tif')