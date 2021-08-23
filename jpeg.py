# -*- coding: utf-8 -*-
import cv2
import sys
import numpy as np
from jpegdef import *
from struct import pack

infilename = "tstimg.bmp"
outfilename = "my_out.jpg"
# 读取一张未经压缩的bmp图片
# 并使用cv2输出为jpg图片
BGR = cv2.imread(infilename)
cv2.imwrite("sys_out.jpg", BGR)

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
# 写入jpeg格式的译码信息
write_head(outfilename, h, w)
# 换算出哈夫曼字典
DC0 = DHT2tbl(std_huffman_DC0)
DC1 = DHT2tbl(std_huffman_DC1)
AC0 = DHT2tbl(std_huffman_AC0)
AC1 = DHT2tbl(std_huffman_AC1)

s = ""
# 二进制写入形式打开文件(追加)
fp = open(outfilename, "ab")
last_block_ydc = 0
last_block_cbdc = 0
last_block_crdc = 0
p = 0
for i in range(0, h, 8):
	for j in range(0, w, 8):
		# 编码速度较慢，输出运行百分比
		print("%5.2f"%(p/nblock*100),"%",end="\b\b\b\b\b\b\b\b")
		p += 1

		# Y通道
		now_block = npdata[i:i+8, j:j+8, 0] - 128		# 取出一个8*8块并减去128
		now_block_dct = cv2.dct(np.float32(now_block))	# DCT变换
		now_block_qut = quantize(now_block_dct, 0)		# 量化
		now_block_zz = block2zz(now_block_qut)			# zigzag扫描
		now_block_dc = now_block_zz[0] - last_block_ydc # 直流分量差分形式记录
		last_block_ydc = now_block_zz[0]				# 记录本次量
		now_block_ac = RLE(now_block_zz[1:])			# 交流分量对0进行行程编码
		s = write_num(s, -1, now_block_dc, DC0)			# 根据编码方式写入直流数据
		for l in range(0, len(now_block_ac), 2):		# 写入交流数据
			s = write_num(s, now_block_ac[l], now_block_ac[l+1], AC0)
			while(len(s) >= 8):							# 记录数据太长会导致爆内存
				num = int(s[0:8], 2)					# 运行速度变慢
				fp.write(pack(">B", num))
				if(num == 0xff):						# 为防止标志冲突
					fp.write(pack(">B", 0))				# 数据中出现0xff需要在后面补两个0x00
				s = s[8:len(s)]

		# Cb通道
		now_block = npdata[i:i+8, j:j+8, 2] - 128
		now_block_dct = cv2.dct(np.float32(now_block))
		now_block_qut = quantize(now_block_dct, 2)
		now_block_zz = block2zz(now_block_qut)
		now_block_dc = now_block_zz[0] - last_block_cbdc
		last_block_cbdc = now_block_zz[0]
		now_block_ac = RLE(now_block_zz[1:])
		s = write_num(s, -1, now_block_dc, DC1)
		for l in range(0, len(now_block_ac), 2):
			s = write_num(s, now_block_ac[l], now_block_ac[l+1], AC1)
			while(len(s) >= 8):
				num = int(s[0:8], 2)
				fp.write(pack(">B", num))
				if(num == 0xff):
					fp.write(pack(">B", 0))
				s = s[8:len(s)]

		# Cr通道
		now_block = npdata[i:i+8, j:j+8, 1] - 128
		now_block_dct = cv2.dct(np.float32(now_block))
		now_block_qut = quantize(now_block_dct, 1)
		now_block_zz = block2zz(now_block_qut)
		now_block_dc = now_block_zz[0] - last_block_crdc
		last_block_crdc = now_block_zz[0]
		now_block_ac = RLE(now_block_zz[1:])
		s = write_num(s, -1, now_block_dc, DC1)
		for l in range(0, len(now_block_ac), 2):
			s = write_num(s, now_block_ac[l], now_block_ac[l+1], AC1)
			while(len(s) >= 8):
				num = int(s[0:8], 2)
				fp.write(pack(">B", num))
				if(num == 0xff):
					fp.write(pack(">B", 0))
				s = s[8:len(s)]
# 写入残留数据
if(len(s) != 0):
	while(len(s) != 8):
		s += '0'
	fp.write(pack(">B", int(s, 2)))
# EOI
fp.write(pack(">H", 0xffd9))
fp.close()