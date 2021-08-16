# -*- coding: utf-8 -*-
import cv2
import sys
import numpy as np
from jpegdef import *
from struct import pack

# 读取一张未经压缩的bmp图片
# 并使用cv2输出为jpg图片
BGR = cv2.imread("tstsimg.bmp")
cv2.imwrite("sys_out.jpg", BGR)

# 色彩空间变换
YCrCb = cv2.cvtColor(BGR, cv2.COLOR_BGR2YCrCb)
h, w, d = YCrCb.shape
npdata = np.array(YCrCb, np.uint8)
for i in range(0,3):
	print(BGR[:,:,i])
print('')
for i in range(0,3):
	print(YCrCb[:,:,i])
exit()

# 转换图片大小
if((h % 8 == 0) and (w % 8 == 0)):
	h = h
	w = w
	nblock = int(h * w / 64)
else:
	h = h // 8 * 8
	w = w // 8 * 8
	YCrCb = cv2.resize(YCrCb, [h, w], cv2.INTER_CUBIC)
	nblock = int(h * w / 64)

write_head("my_out.jpg", h, w)

last_block_dc = 0
for i in range(0, h, 8):
	for j in range(0, w, 8):
		for k in range(0, 3):
			now_block = npdata[i:i+8, j:j+8, k] - 128
			now_block_dct = cv2.dct(np.float32(now_block))
			now_block_qut = quantize(now_block_dct, k)
			now_block_zz = block2zz(now_block_qut)
			now_block_dc = now_block_zz[0] - last_block_dc
			last_block_dc = now_block_zz[0]
			now_block_ac = RLE(now_block_zz[1:])
