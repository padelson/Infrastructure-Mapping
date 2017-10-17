"""
Author:  Xiao Chen
Ref: GDAL tutorial code 
"""

import argparse
import time

import gdal
import numpy as np
#from scipy.misc import imshow, imresize
from scipy.misc import imresize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os 



def read(tif_path, H, W):
    '''
    Reads the middle HxW image from the tif given by tif_path
    '''
    gdal_dataset = gdal.Open(tif_path)
    # x_size and y_size and the width and height of the entire tif in pixels
    x_size, y_size = gdal_dataset.RasterXSize, gdal_dataset.RasterYSize
    print("TIF Size (W, H): ", x_size, y_size)
    # Mid point minus half the width and height we want to read will give us our top left corner
    if W > x_size:
        raise Exception("Requested width exceeds tif width.")
    if H > y_size:
        raise Exception("Requested height exceeds tif height.")
    gdal_result = gdal_dataset.ReadAsArray((x_size - W)//2, (y_size - H)//2, W, H)
    # If a tif file has only 1 band, then the band dimension will be removed.
    if len(gdal_result.shape) == 2:
        gdal_result = np.reshape(gdal_result, [1] + list(gdal_result.shape))
    # gdal_result is a rank 3 tensor as follows (bands, height, width)
    return np.transpose(gdal_result, (1, 2, 0))

if __name__ == "__main__":
    default_path = "/afs/ir/class/cs325b/gdal_tutorial/data/F182013.v4c_web.stable_lights.avg_vis.tif"
    parser = argparse.ArgumentParser(description="Read the middle tile from a tif.")
    parser.add_argument('-p', '--tif_path',
        default=default_path,
        help='The path to the tif')
    parser.add_argument('-w','--width', default=1000, type=int, help='Tile width')
    parser.add_argument('-t','--height', default=5000, type=int, help='Tile height')
    args = parser.parse_args()

    file_input_path=args.tif_path 
    print("==============================")
    print("parsing file -- %s " % file_input_path)
    folder_, fname = os.path.split(file_input_path)    
    fname_wo_type = (fname.split(".")[0])
    

    
    #raise NotImplementedError("----more!----")  
    start_time = time.time()
    img = read(args.tif_path, args.height, args.width)
    img_time = time.time() - start_time
    start_time = time.time()
    img2 = read(args.tif_path, args.height, args.width*10)
    img2_time = time.time() - start_time
    print("The image you requested took {} seconds to read, an image with ten times the width to {} seconds to read".format(
        img_time, img2_time))

    if args.tif_path == default_path:
        # Something to make the nightlights look good in images
        def clean(x):
            x = np.log(x + 1)
            x[x > 3] = 3
            return x
        img = clean(img)
        img2 = clean(img2)
    
    
    num_bands = np.squeeze(img).shape[2]
    
    print(num_bands)
    img_tensor = np.squeeze(img) 
    img2_tensor = np.squeeze(img2)
    print(img2_tensor[:,:,:3].shape)
    fig=plt.figure()
    plt.imshow(np.dstack([img2_tensor[:,:,2].astype(np.uint8), 
                         img2_tensor[:,:,1].astype(np.uint8),
                         img2_tensor[:,:,0].astype(np.uint8)] ))
    
    #plt.tight_layout()
    #plt.show()
    
    save_path_dir = "../img/"+fname_wo_type 
    if os.path.exists(save_path_dir) is False :
        os.mkdir(save_path_dir)
    
    fig.savefig(save_path_dir+"/rgb_overall_view.png", width=12, height=12)
    # plot overall histogram of bands value

    img2_arr = img2_tensor.flatten()
    img2_arr = img2_arr[np.where( img2_arr>0.1)]
    
    weights_norm = np.ones_like(img2_arr)/float(len(img2_arr)) 
    fig2=plt.figure(figsize=(12, 9))
    plt.hist(img2_arr, bins=500, weights=weights_norm)
    plt.xlabel("mean: %.2f; var: %.2f" %(np.mean(img2_arr), np.var(img2_arr) ), fontsize=16 )
    plt.ylabel("Proportion (%)", fontsize=16)
    plt.tick_params(labelsize=13)
    plt.title("All bands histogram")
    #plt.show()
    fig2.savefig(save_path_dir+"hist_overall_bands.png", width=12, height=9)
    plt.close('all')
    print("=========finish overall_view fig generation!=============")
    #############################
    # def sub func to vis hist 
    #############################
    def visHistOnBand(np_mat, band_idx, b_key, color_="darkcyan", path_def=".", displayImg=False):
        arr_ = np_mat.flatten()
        arr_ = arr_[np.where(arr_ > 0.1)]
        bin_n = np.minimum(100, arr_.shape[0])
        weights_ = np.ones_like(arr_)/float(len(arr_))
        fig_ = plt.figure()
        plt.hist(arr_, bins=bin_n, weights=weights_, color=color_)
        plt.xlabel("mean: %.2f ; var: %.2f " % (np.mean(arr_), np.var(arr_) ), fontsize=16 )
        plt.ylabel("Proportion (%)", fontsize=16)
        plt.title("Band %d: %s" % (band_idx, b_key) )
        #plt.tight_layout()    
        if displayImg is True:
            plt.show()
        
        fig_.savefig(path_def+"/band_%d_hist.png"%band_idx, width=10, height=8)

         
    MultiBand_keyword = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
    color_name = ["steelblue", "forestgreen", "tomato", "violet", "saddlebrown", "darkorange"]      
    for k in range(num_bands): 
        # uncomment if you want to see the small tile in img 
        #img = imresize(np.squeeze(img_tensor)[:,:,k], 0.9)
        img2 = imresize(np.squeeze(img2_tensor)[:,:,k], 0.9)
        visHistOnBand(np.squeeze(img2_tensor)[:,:,k], k, MultiBand_keyword[k], 
                      color_ = color_name[k],  path_def=save_path_dir, displayImg=False)

        #plt.imshow(img, cmap='gray')
        #plt.imshow(img)
        #plt.title("band %d" % k )
        #plt.show()
        #plt.imshow(img2, cmap='gray')
        fig3 = plt.figure(figsize=(10,10))
        plt.imshow(img2)
        plt.title("Band %d: %s" % (k, MultiBand_keyword[k] ))	
        #plt.show()
        #plt.tight_layout()
        fig3.savefig(save_path_dir+"/band_%d_view.png" % k, width=10, height=10 )
        print("=====band %d figs are generated!======" % k)   
