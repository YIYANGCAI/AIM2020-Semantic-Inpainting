import shutil
import os

img_tags = ['with_holes', 'mask', 'merge_es']
tgt_folders = ['./datasets/test_inputs/', './datasets/test_masks/', './datasets/test_merge_es/']
source_folder = './for_users/'

for root, dirs, files in os.walk(source_folder, topdown=False):
    for name in files:
        print("copying the file:\t{}".format(name))
        if img_tags[0] in name:
            shutil.copy(os.path.join(source_folder, name), os.path.join(tgt_folders[0], name))
        elif img_tags[1] in name:
            shutil.copy(os.path.join(source_folder, name), os.path.join(tgt_folders[1], name))
        elif img_tags[2] in name:
            shutil.copy(os.path.join(source_folder, name), os.path.join(tgt_folders[2], name))