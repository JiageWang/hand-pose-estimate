import cv2
import torch
import numpy as np
from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])
Point.__add__ = lambda self, pair: Point(x=self.x + pair[0], y=self.y + pair[1])
Point.__sub__ = lambda self, pair: Point(x=self.x - pair[0], y=self.y - pair[1])
Point.__mul__ = lambda self, pair: Point(x=self.x * pair[0], y=self.y * pair[1])

Bbox = namedtuple("Bbox", ["x1", "y1", "x2", "y2"])

Ratio = namedtuple("Ratio", ["w", 'h'])
Ratio.__truediv__ = lambda self, pair: Ratio(w=self.w / pair[0], h=self.h / pair[1])
Ratio.__mul__ = lambda self, pair: Ratio(w=self.w * pair[0], h=self.h * pair[1])

Size = namedtuple("Size", ["w", 'h'])
Size.__truediv__ = lambda self, pair: Ratio(w=self.w / pair[0], h=self.h / pair[1])
Size.__mul__ = lambda self, pair: Size(w=self.w * pair[0], h=self.h * pair[1])


class HandCapture(object):
    CAP_SIZE = Size(640, 480)
    NET_SIZE = Size(256, 256)
    RATIO_CAP_TO_NET = NET_SIZE / CAP_SIZE
    RATIO_NET_DOWNSAMPLE = Ratio(4, 4)

    THRESHOLD = 0.4
    BLOCK_WIDTH = 2

    def __init__(self, model_path):
        if torch.cuda.is_available():
            model = torch.jit.load("hand.pts")
            model = model.cuda()
        else:
            model = torch.jit.load("hand.pts", map_location='cpu')
        self.model = model.eval()

    def to_tensor(self, img, bbox=None):
        input = img.copy()
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            input = input[y1:y2, x1:x2, :]
            ratio = self.NET_SIZE / Size(input.shape[1], input.shape[0])  # size/size=ratio
            M = np.array([[min(ratio), 0, 0], [0, min(ratio), 0]])
            input = cv2.warpAffine(input, M, self.NET_SIZE, borderMode=1, borderValue=128)
            # cv2.imshow('warp', input)
        else:
            ratio = self.RATIO_CAP_TO_NET
            input = cv2.resize(input, self.NET_SIZE)
        input = input.astype(float)
        input = input / 255 - 0.5
        tensor = torch.tensor(input, dtype=torch.float32)
        tensor = tensor.permute((2, 0, 1))
        tensor = tensor.unsqueeze(0)
        return tensor, ratio

    def forward(self, tensor):
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        featuremap = self.model(tensor)[3].cpu().data.numpy()  # (n,24,64,64)
        return featuremap

    def detectBbox(self, img):
        tensor, _ = self.to_tensor(img)
        featuremaps = self.forward(tensor)

        region_map = featuremaps[0, 21:, :, :]
        locations = self.nmsLocation(region_map[0])  # 找极大值点
        # 合成bbox
        bboxs = self.getBBox(region_map, locations)
        return bboxs

    def detectHand(self, img, bbox):
        if not bboxs: return []
        tensors = []
        ratios = []
        for bbox in bboxs:
            tensor, ratio = self.to_tensor(img, bbox)
            tensors.append(tensor)
            ratios.append(ratio)
        input_tensors = torch.cat(tensors, 0)
        featuremaps = self.forward(input_tensors)

        keypoints = [[] for i in range(len(bboxs))]
        for i in range(len(bboxs)):
            for j in range(21):  # 21个关键点
                locations = self.nmsLocation(featuremaps[i, j, :, :])
                if locations:
                    point = locations[0][1] * self.RATIO_NET_DOWNSAMPLE
                    x = int(point.x / min(ratios[i]) + bboxs[i][0])
                    y = int(point.y / min(ratios[i]) + bboxs[i][1])
                    keypoints[i].append(Point(x, y))
        return keypoints

    def getBBox(self, region_map, locations):
        """

        :param region_map: (3,64,464)
        :param locations: ((value, Point(x,y)),..)
        :param ratio: Ratio(w,h)
        :return:
        """
        bboxs = []
        for location in locations:
            point = location[1]  # (x, y)
            ratio_width = 0.  # 累加5x5内的ratio_width
            ratio_height = 0.  # 累加5x5内的ratio_height
            pixcount = 0  # 累加个数
            for m in range(max(point.y - 2, 0), min(point.y + 3, region_map.shape[1])):
                for n in range(max(point.x - 2, 0), min(point.x + 3, region_map.shape[2])):
                    ratio_width += region_map[1, m, n]
                    ratio_height += region_map[2, m, n]
                    pixcount += 1
            if pixcount > 0:
                ratio = Ratio(
                    min(max(ratio_width / pixcount, 0), 1),
                    min(max(ratio_height / pixcount, 0), 1)
                )  # 长宽相对于图像比例
                center = point * (self.RATIO_NET_DOWNSAMPLE / self.RATIO_CAP_TO_NET)  # (x,y)
                size = self.NET_SIZE * (ratio / self.RATIO_CAP_TO_NET)  # (w,h)
                x_min = int(max(center.x - size.w / 2, 0))
                y_min = int(max(center.y - size.h / 2, 0))
                x_max = int(min(center.x + size.w / 2, self.CAP_SIZE.w - 1))
                y_max = int(min(center.y + size.h / 2, self.CAP_SIZE.h - 1))
                bboxs.append(Bbox(x_min, y_min, x_max, y_max))  # (x, y, x, y)
        return bboxs

    def nmsLocation(self, featuremap):
        """
        :param featuremap:  特征图 64x64
        :return: locations ()
        """
        # set the local window size: 5*5
        locations = []
        blockwidth = self.BLOCK_WIDTH
        threshold = self.THRESHOLD
        for i in range(blockwidth, featuremap.shape[1] - blockwidth):  #
            for j in range(blockwidth, featuremap.shape[0] - blockwidth):
                value = featuremap[j][i]
                point = Point(i, j)  # (x,y)
                if value < threshold: continue
                localmaximum = True
                for m in range(min(i - blockwidth, 0), min(i + blockwidth, featuremap.shape[1] - 1) + 1):
                    for n in range(max(j - blockwidth, 0), min(j + blockwidth, featuremap.shape[0] - 1) + 1):
                        if featuremap[n][m] > value:
                            localmaximum = False
                            break
                    if not localmaximum: break
                if localmaximum:
                    locations.append((value, point))
        sorted(locations, key=lambda a: a[0], reverse=True)
        return locations

    @staticmethod
    def drawBbox(img, bboxs):
        if not bboxs: return
        for bbox in bboxs:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))

    def drawKeypoints(self, img, keypoints):
        if not keypoints: return
        for i in range(len(keypoints)):
            for keypoint in keypoints[i]:
                cv2.circle(img, keypoint, 2, (255, 0, 0))


if __name__ == "__main__":
    hand = HandCapture("hand.pts")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    while True:
        img = cap.read()[1]
        bboxs = hand.detectBbox(img)
        keypoints = hand.detectHand(img, bboxs)
        hand.drawBbox(img, bboxs)
        hand.drawKeypoints(img, keypoints)
        cv2.imshow("img", img)
        if cv2.waitKey(1) == ord('q'):
            break
