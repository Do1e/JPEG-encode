import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='input file', required=True)
parser.add_argument('-o', '--output', help='output file', required=True)
args = parser.parse_args()

BGR = cv2.imread(args.input)
# 色彩空间变换
YCrCb = cv2.cvtColor(BGR, cv2.COLOR_BGR2YCrCb)
h, w, d = YCrCb.shape

# 转换图片大小，必须能被切分成8*8的小块
if((h % 8 == 0) and (w % 8 == 0)):
	nblock = int(h * w / 64)
else:
	h = h // 8 * 8
	w = w // 8 * 8
	YCrCb = cv2.resize(YCrCb, [h, w], cv2.INTER_CUBIC)
	nblock = int(h * w / 64)
# 后续运算会转换变为有符号数，因此做数据转换
npdata = np.array(YCrCb, np.int16)
del BGR
del YCrCb