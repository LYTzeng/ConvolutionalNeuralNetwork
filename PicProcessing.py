import numpy as np
from PIL import Image


class Picture():
    def __init__(self, image_source=None, image_array=None):
        """
        圖片矩陣處理
        :param image_source:string 圖片路徑
        :param image_array: 2D numpy array (list of lists) 圖片2D矩陣
        """
        self.image_source = image_source
        self.image_np_array = image_array

    def pic2array(self):
        pic = Image.open(self.image_source)
        # width, height = pic.size
        pic_2d_list = np.array(pic)
        return pic_2d_list

    def array2pic(self):
        height = len(self.image_np_array)
        width = len(self.image_np_array[0])
        pic = Image.fromarray(self.image_np_array, 'RGB')
        pic.show()
        return pic
