import os
from shutil import copyfile

path1 = '/Users/qiuhaonan/Downloads/gtFine_trainvaltest/final_target'
path2 = '/Users/qiuhaonan/Downloads/gtFine_trainvaltest/final_left'
path3 = '/Users/qiuhaonan/Downloads/gtFine_trainvaltest/val'

folders = os.listdir(path1)
for folder in folders:
    path1_sub = path1 + '/'+folder
    path2_sub = path2 + '/'+folder
    path3_sub = path3 + '/' + folder
    if not os.path.exists(path2_sub):
        os.makedirs(path2_sub)
    files = os.listdir(path1_sub)
    for file in files:
        file_name = path1_sub + '/' + file
        copyfile(path3_sub+'/'+file[:-17]+'_leftImg8bit.png',path2_sub+'/'+file[:-17]+'_leftImg8bit.png')
        print(file_name)
