from PicProcessing import Picture
import numpy as np
from FakeData import Fake


class Convolution():
    def __init__(self, image_source=None, filter=None):
        """
        CNN 捲積神經網路
        :param image_source: string 圖片路徑
        :param filter: list of lists (2D) 用來做 Featured Map
        """
        if image_source is None:
            self.image_source_dir = 'D:/NTUT/智慧型代理人/作業/CNN/test_data/pic_1.jpg'
        else:
            self.image_source_dir = image_source
        self.picture = Picture(self.image_source_dir)
        self.image_source_arr = self.picture.pic2array()
        self.image_np = np.array(self.image_source_arr)
        if filter is None:
            self.filter = [[1,0,-1],[2,0,-2],[1,0,-1]]
        else:
            self.filter = filter
        self.filter = np.array(self.filter)
        self.map = None
        self.map_reLu = None

        # 測資
        self.fake = Fake()

    def feature_map(self):
        """
        依據filter和原始影像陣列計算出Featured Map
        :return:
        """
        height, width = self.size(self.image_np)
        filter_height,filter_width  = self.size(self.filter)
        self.map_height = height - (filter_height - 1)
        self.map_width = width - (filter_width - 1)
        self.map = [[] for i in range(self.map_height)]

        for h in range(0, self.map_height):
            for w in range(0, self.map_width):
                part = self.image_np[h:filter_height+h, w:filter_width+w]
                try:
                    self.multi = np.multiply(part, self.filter)
                except Exception as e:
                    print(e)
                map_element = np.sum(self.multi)
                self.map[h].append(map_element)
        return self.map

    def reLu(self):
        """ReLu函數"""
        map_np = np.array(self.map)
        self.map_reLu = [[] for i in range(self.map_height)]
        for h in range(self.map_height):
            for w in range(self.map_width):
                if map_np[h, w] < 0 :
                    self.map_reLu[h].append(0)
                elif map_np[h, w] >= 0:
                    self.map_reLu[h].append(map_np[h, w])
        return np.array(self.map_reLu)

    def size(self, array):
        """
        :param array: list
        :return: height, width
        """
        return len(array),len(array[0])

    def test_featured_map(self):
        """測試featured_map"""
        self.image_np  = np.array(self.fake.arr3x3)
        self.filter = np.array(self.fake.arr2x2filter)
        print(self.feature_map())
        print(self.reLu())
