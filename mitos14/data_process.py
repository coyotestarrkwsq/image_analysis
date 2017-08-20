import csv
from scipy import misc
from matplotlib import pyplot as plt

path = '/home/wangsq/A03_00Ab_mitosis.csv'


with open(path, 'rb') as csvfile:
    content = csv.reader(csvfile)
    content = list(content)
    
#content = map(lambda x : map(int, x), content)

img = misc.imread('/home/wangsq/A03_00Ab.tiff')

j=0
x = int(content[j][0])
y = int(content[j][1])
print (x,y)

i=32
img = img[y-i:y+i, x-i:x+i]

plt.imshow(img)
plt.show()

