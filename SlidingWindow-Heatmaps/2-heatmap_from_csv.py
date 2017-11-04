import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

#Image size (width, height, color)
img = Image.new('F', (1900, 1900), 0)

# open csv
reader = csv.reader(open('output_class.csv', 'rb'))
#bucle principal
for index,row in enumerate(reader):
    
    ImageDraw.Draw(img).polygon(((float(row[0]), float(row[1])), (float(row[2]),\
    float(row[3])), (float(row[4]), float(row[5])), (float(row[6]), \
    float(row[7]))), fill=float(row[8]))

#All images in array
myimg = np.ma.masked_equal(np.array(img), 0.)
#nearest interpolation array plot
plt.imshow(myimg, interpolation="nearest")
#legend 
plt.colorbar()
#figure show
plt.show()