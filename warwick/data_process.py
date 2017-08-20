import scipy.io as sio
from scipy import misc
from matplotlib import pyplot as plt

classification = 'Classification/'
detection = 'Detection/'
path = '/home/wangsq/CRCHistoPhenotypes_2016_04_28/'

content = sio.loadmat(path + classification + 'img1/img1_others.mat')

content = map(lambda x : map(round, x), content.get('detection'))
content = map(lambda x : map(int, x), content)

img = misc.imread(path + classification + 'img1/img1.bmp')
i=1
x = content[i][0]
y = content[i][1]
print (x,y)
img = img[y-10:y+10, x-10:x+10]
plt.imshow(img)
plt.show()
