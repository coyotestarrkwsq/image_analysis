import csv
from scipy import misc
from matplotlib import pyplot as plt

path = '/home/wangsq/data_set/mitos14/A17/mitosis/A17_01Ad_not_mitosis.csv'


with open(path, 'rb') as csvfile:
    content = csv.reader(csvfile)
    content = list(content)
    
#content = map(lambda x : map(int, x), content)

img = misc.imread('/home/wangsq/data_set/mitos14/A17/frames/x40/A17_01Ad.tiff')

j=0
x = int(content[j][0])
y = int(content[j][1])
print (x,y)

i=32
img = img[y-i:y+i, x-i:x+17]

plt.imshow(img)
plt.show()

