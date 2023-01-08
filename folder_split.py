from random import seed
import splitfolders
splitfolders.ratio('D:/dataset/PASCAL_VOC_2007/VOC2007/trainval_original', output='D:/dataset/PASCAL_VOC_2007/VOC2007/trainval',
                   seed=42, ratio=(0.8, 0.2))
