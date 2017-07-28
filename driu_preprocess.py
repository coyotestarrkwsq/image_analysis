from scipy import misc
import glob
import numpy as np
import cv2
import argparse
from matplotlib import pyplot as plt
import os


def mean_subtract(array, r_mean, g_mean, b_mean):
    array[:,:,0]-=r_mean
    array[:,:,1]-=g_mean
    array[:,:,2]-=b_mean
    return array

def crop(image):
    return image[30:566, 10:546]

def padding(image, pad_height_l, pad_height_r, pad_length_l, pad_length_r):
    if image.ndim>2:
        return np.pad(image, ((pad_height_l, pad_height_r), (pad_length_l, pad_length_r), (0,0)),
            mode = 'constant', constant_values=0)
    else:
        return np.pad(image, ((pad_height_l, pad_height_r), (pad_length_l, pad_length_r)),
            mode = 'constant', constant_values=0)

def rotate(image, angle):
    (h, l) = (536,536)
    center = (h/2, l/2)    

    image = crop(image)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, M, (h, l))        
    image = padding(image, 24, 24, 14, 15)  
    return image

def scale(image, scaling):

    if scaling == 'enlarge':
        large = cv2.resize(image, (0,0), fx=1.5, fy=1.5)
        
        large_ul = large[0:584, 0:565]
        large_ur = large[0:584, 283:848]
        large_ll = large[292:876, 0:565]
        large_lr = large[292:876, 283:848]
    
        return [large_ul, large_ur, large_ll, large_lr]
    
    elif scaling =='shrink':
        image = crop(image)
        small = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
        
        small = padding(small, 158, 158, 148, 149)
        return small

def main():
    
    image_path = "/home/wangsq/data_set/DRIVE/training/images/*.tif" 
    seg_path = "/home/wangsq/data_set/DRIVE/training/1st_manual/*.gif"
    image_data_path = sorted(glob.glob(image_path))
    seg_data_path = sorted(glob.glob(seg_path))
    save_path = '/home/wangsq/data_set/DRIVE/training/final_full_aug_no_scale_data/'
    file = open(save_path + 'train_pair.txt', 'w')

    
    #r_mean = np.mean([np.mean(misc.imread(image)[:,:,0]) for image in data_path])
    #g_mean = np.mean([np.mean(misc.imread(image)[:,:,1]) for image in data_path])
    #b_mean = np.mean([np.mean(misc.imread(image)[:,:,2]) for image in data_path])

    for im, seg in zip(image_data_path, seg_data_path):
        image = misc.imread(im)
        segm = misc.imread(seg)
        
        filepath, filename = os.path.split(im)
        im_name, exts = os.path.splitext(filename)


        filepath, filename = os.path.split(seg)
        seg_name, exts = os.path.splitext(filename)





        for i in range(0,16):
            angle = i*22.5
            rotate_im = rotate(image, angle)
            misc.imsave(save_path + im_name + '_rotate' + str(angle) + '.png', rotate_im)
            file.write(im_name + '_rotate' + str(angle) + '.png ')


            rotate_seg = rotate(segm, angle)
            misc.imsave(save_path + seg_name + '_rotate' + str(angle) + '.jpeg', rotate_seg)
            file.write(seg_name + '_rotate' + str(angle) + '.jpeg' + '\n')


            flip_im = np.flipud(rotate_im)
            misc.imsave(save_path + im_name + '_rotate' + str(angle) + 'fup.png', flip_im)
            file.write(im_name + '_rotate' + str(angle) + 'fup.png ')

            flip_seg = np.flipud(rotate_seg)
            misc.imsave(save_path + seg_name + '_rotate' + str(angle) + 'fup.jpeg', flip_seg)
            file.write(seg_name + '_rotate' + str(angle) + 'fup.jpeg' + '\n')



            flip_im = np.fliplr(rotate_im)
            misc.imsave(save_path + im_name + '_rotate' + str(angle) + 'flr.png', flip_im)
            file.write(im_name + '_rotate' + str(angle) + 'flr.png ')

            flip_seg = np.fliplr(rotate_seg)
            misc.imsave(save_path + seg_name + '_rotate' + str(angle) + 'flr.jpeg', flip_seg)
            file.write(seg_name + '_rotate' + str(angle) + 'flr.jpeg' + '\n')

       




if __name__=='__main__':

    main()


'''
        enlarge_im = scale(image, 'enlarge')

        enlarge_seg = scale(segm, 'enlarge')
        
        for i in range(len(enlarge_im)):
            misc.imsave(save_path + im_name + '_enlarge' + str(i) + '.tif', enlarge_im[i])
            misc.imsave(save_path + seg_name + '_enlarge' + str(i) + '.jpeg', enlarge_seg[i])
        
        shrink_im = scale(image, 'shrink')
        #plt.imshow(shrink_im)
        #plt.show()
        #sys.exit()
        misc.imsave(save_path + im_name + '_shrink.tif', shrink_im)        
        
        shrink_seg = scale(segm, 'shrink')
        misc.imsave(save_path + seg_name + '_shrink.jpeg', shrink_seg)
        
'''
