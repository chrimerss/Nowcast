import h5py
import numpy as np
from osgeo import gdal
from glob import glob
import os

EVENTS= ['20170604', '20170624', '20170709','20170922','20180704','20181207','20190507','20190522','20190823','20190919']
N_FOERCAST=10

def make_h5(pth):
    fnames= sorted(glob(pth))
    print(fnames)
    name= pth.split('/')[-2]
    h5= h5py.File(os.path.join('data', name+'.h5'), 'w')
    n_splits= len(fnames)//N_FOERCAST
    sample_splits= [fnames[i*N_FOERCAST:(i+1)*N_FOERCAST] for i in range(n_splits)]
    for i in range(n_splits-1):
        print('%d/%d'%(i,n_splits))
        tmp_arr_train= np.zeros((N_FOERCAST,648,800), dtype=np.float16)
        tmp_arr_test= np.zeros((N_FOERCAST,648,800), dtype=np.float16)
        for j in range(N_FOERCAST):
            fname= sample_splits[i][j]
            arr= gdal.Open(fname).ReadAsArray()[:-1,:]
            assert arr.shape==(648, 800), 'expected shape (649,800), but got %s'%arr.shape
            tmp_arr_train[j]= arr
            fname= sample_splits[i+1][j]
            arr= gdal.Open(fname).ReadAsArray()[:-1,:]
            assert arr.shape==(648, 800), 'expected shape (649,800), but got %s'%arr.shape
            tmp_arr_test[j]=arr


        h5.create_dataset('train-%d'%i, data=tmp_arr_train)
        h5.create_dataset('test-%d'%i,data=tmp_arr_test)

    h5.close()

    
if __name__=='__main__':
    for event in EVENTS:
        target_path= '/home/allen/drive/geotiff' + '/' +event+ '/'+ '*.tif'
        make_h5(target_path)