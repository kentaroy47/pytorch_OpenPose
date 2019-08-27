# CODING=UTF-8
import os
import urllib.request
import zipfile
import tarfile


data_dir = "./data/"
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

weights_dir = "./weights/"
if not os.path.exists(weights_dir):
    os.mkdir(weights_dir)

# MSCOCO??2014
url =  "http://images.cocodataset.org/zips/val2014.zip"
target_path = os.path.join(data_dir, "val2014.zip") 

if not os.path.exists(target_path):
    urllib.request.urlretrieve(url, target_path)
    
    zip = zipfile.ZipFile(target_path)
    zip.extractall(data_dir)  # ZIP??????
    zip.close()  
    

