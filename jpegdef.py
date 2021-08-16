# -*- coding: utf-8 -*-
import numpy as np
import os
from struct import pack

std_luminance_quant_tbl = np.array(
	[
		[16, 11, 10, 16, 24, 40, 51, 61],
		[12, 12, 14, 19, 26, 58, 60, 55],
		[14, 13, 16, 24, 40, 57, 69, 56],
		[14, 17, 22, 29, 51, 87, 80, 62],
		[18, 22, 37, 56, 68,109,103, 77],
		[24, 35, 55, 64, 81,104,113, 92],
		[49, 64, 78, 87,103,121,120,101],
		[72, 92, 95, 98,112,100,103, 99]
	],
	np.uint8
)
std_chrominance_quant_tbl = np.array(
	[
		[ 34, 36, 48, 94,198,198,198,198],
		[ 36, 42, 52,132,198,198,198,198],
		[ 48, 52,112,198,198,198,198,198],
		[ 94,132,198,198,198,198,198,198],
		[198,198,198,198,198,198,198,198],
		[198,198,198,198,198,198,198,198],
		[198,198,198,198,198,198,198,198],
		[198,198,198,198,198,198,198,198]
	],
	np.uint8
)
std_huffman_DC0 = np.array(
	[0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
	 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
	np.uint8
)
std_huffman_DC1 = np.array(
	[0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
	 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
	np.uint8
)
std_huffman_AC0 = np.array(
	[0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0,
	 1, 125, 1, 2, 3, 0, 4, 17, 5, 18, 33, 49,
	 65, 6, 19, 81, 97, 7, 34, 113, 20, 50, 129,
	 145, 161, 8, 35, 66, 177, 193, 21, 82, 209,
	 240, 36, 51, 98, 114, 130, 9, 10, 22, 23,
	 24, 25, 26, 37, 38, 39, 40, 41, 42, 52, 53,
	 54, 55, 56, 57, 58, 67, 68, 69, 70, 71, 72,
	 73, 74, 83, 84, 85, 86, 87, 88, 89, 90, 99,
	 100, 101, 102, 103, 104, 105, 106, 115, 116,
	 117, 118, 119, 120, 121, 122, 131, 132, 133,
	 134, 135, 136, 137, 138, 146, 147, 148, 149,
	 150, 151, 152, 153, 154, 162, 163, 164, 165, 
	 166, 167, 168, 169, 170, 178, 179, 180, 181, 
	 182, 183, 184, 185, 186, 194, 195, 196, 197, 
	 198, 199, 200, 201, 202, 210, 211, 212, 213, 
	 214, 215, 216, 217, 218, 225, 226, 227, 228, 
	 229, 230, 231, 232, 233, 234, 241, 242, 243, 
	 244, 245, 246, 247, 248, 249, 250],
	np.uint8
)
std_huffman_AC1 = np.array(
	[0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1,
	 2, 119, 0, 1, 2, 3, 17, 4, 5, 33, 49, 6,
	 18, 65, 81, 7, 97, 113, 19, 34, 50, 129,
	 8, 20, 66, 145, 161, 177, 193, 9, 35, 51,
	 82, 240, 21, 98, 114, 209, 10, 22, 36, 52,
	 225, 37, 241, 23, 24, 25, 26, 38, 39, 40,
	 41, 42, 53, 54, 55, 56, 57, 58, 67, 68, 69,
	 70, 71, 72, 73, 74, 83, 84, 85, 86, 87, 88,
	 89, 90, 99, 100, 101, 102, 103, 104, 105,
	 106, 115, 116, 117, 118, 119, 120, 121, 122,
	 130, 131, 132, 133, 134, 135, 136, 137, 138,
	 146, 147, 148, 149, 150, 151, 152, 153, 154,
	 162, 163, 164, 165, 166, 167, 168, 169, 170,
	 178, 179, 180, 181, 182, 183, 184, 185, 186,
	 194, 195, 196, 197, 198, 199, 200, 201, 202,
	 210, 211, 212, 213, 214, 215, 216, 217, 218,
	 226, 227, 228, 229, 230, 231, 232, 233, 234,
	 242, 243, 244, 245, 246, 247, 248, 249, 250],
	np.uint8
)

class Sym_Code():
	def __init__(self, symbol, code, n_bit):
		self.symbol = symbol
		self.code = code
		str_code=''
		mask = 1 << (n_bit - 1)
		for i in range(0, n_bit):
			if(mask & code):
				str_code += '1'
			else:
				str_code += '0'
			mask >>= 1
		self.str_code = str_code
	"""定义输出形式"""
	def __str__(self):
		return "0x{:0>2x}    |  {}".format(self.symbol, self.str_code)
	"""定义排序依据"""
	def __eq__(self, other):
		return self.symbol == other.symbol
	def __le__(self, other):
		return self.symbol < other.symbol
	def __gt__(self, other):
		return self.symbol > other.symbol

def quantize(block, dim):
	if(dim == 0):
		qarr = std_luminance_quant_tbl
	else:
		qarr = std_chrominance_quant_tbl
	return (block / qarr).round().astype(np.int16)

def block2zz(block):
	re = np.empty(64, np.int16)
	pos = np.array([0, 0])
	R = np.array([0, 1])
	LD = np.array([1, -1])
	D = np.array([1, 0])
	RU = np.array([-1, 1])
	for i in range(0, 64):
		# print(i, pos)
		re[i] = block[pos[0], pos[1]]
		if(((pos[0] == 0) or (pos[0] == 7)) and (pos[1] % 2 == 0)):
			pos = pos + R
		elif(((pos[1] == 0) or (pos[1] == 7)) and (pos[0] % 2 == 1)):
			pos = pos + D
		elif((pos[0] + pos[1]) % 2 == 0):
			pos = pos + RU
		else:
			pos = pos + LD
	return re

def RLE(AClist):
	re = []
	cnt = 0
	for i in range(0, 63):
		if(cnt == 15):
			re.append(cnt)
			re.append(0)
			cnt = 0
		elif(AClist[i] == 0):
			cnt += 1
		else:
			re.append(cnt)
			re.append(AClist[i])
			cnt = 0
	while(re[-1] == 0):
		re.pop()
		re.pop()
		if(len(re) == 0):
			break
	re.append(0)
	re.append(0)
	if(len(re) > 100):
		print("Error")
		exit()
	return np.array(re, np.int16)
	# return np.append(np.append(np.array(re, np.int16), np.zeros(63-len(re), np.int16)), np.array(len(re), np.int16))

def DHT2tbl(data):
	numbers = data[0:16]
	symbols = data[16:len(data)]
	# print(numbers)
	# print(symbols)
	if(sum(numbers) != len(symbols)):
		print("Wrong DHT!")
		exit()
	code = 0
	SC = []
	for n_bit in range(1, 17):
		n_bit += 1
		for symbol in symbols[sum(numbers[0:n_bit-1]):sum(numbers[0:n_bit])]:
			SC.append(Sym_Code(symbol, code, n_bit))
			code += 1
			# print(SC[-1])
			# f.write(str(SC[-1]))
		code <<= 1
	return sorted(SC)

def write_head(filename, h, w):
	fp = open(filename, "wb")

	# SOI
	fp.write(pack(">H", 0xffd8))
	# APP0
	fp.write(pack(">H", 0xffe0))
	fp.write(pack(">H", 16))			# APP0字节数
	fp.write(pack(">L", 0x4a464946))	# JFIF
	fp.write(pack(">B", 0))				# 0
	fp.write(pack(">H", 0x0101))		# 版本号: 1.1
	fp.write(pack(">B", 0x00))			# XY无密度单位
	fp.write(pack(">L", 0x00010001))	# XY方向像素密度
	fp.write(pack(">H", 0x0000))		# 无缩略图信息
	# DQT_0
	fp.write(pack(">H", 0xffdb))
	fp.write(pack(">H", 64+3))			# 量化表字节数
	fp.write(pack(">B", 0x00))			# 量化表精度: 8bit(0)  量化表ID: 0
	for it in std_luminance_quant_tbl:
		for item in it:
			fp.write(pack(">B", item))	# 量化表0内容
	# DQT_1
	fp.write(pack(">H", 0xffdb))
	fp.write(pack(">H", 64+3))			# 量化表字节数
	fp.write(pack(">B", 0x01))			# 量化表精度: 8bit(0)  量化表ID: 1
	for it in std_chrominance_quant_tbl:
		for item in it:
			fp.write(pack(">B", item))	# 量化表0内容
	# SOF0
	fp.write(pack(">H", 0xffc0))
	fp.write(pack(">H", 17))			# 帧图像信息字节数
	fp.write(pack(">B", 8))				# 精度: 8bit
	fp.write(pack(">H", h))				# 图像高度
	fp.write(pack(">H", w))				# 图像宽度
	fp.write(pack(">B", 3))				# 颜色分量数: 3(YCrCb)
	fp.write(pack(">B", 1))				# 颜色分量ID: 1
	fp.write(pack(">H", 0x1100))		# 水平垂直采样因子: 1  使用的量化表ID: 0
	fp.write(pack(">B", 2))				# 颜色分量ID: 2
	fp.write(pack(">H", 0x1101))		# 水平垂直采样因子: 1  使用的量化表ID: 1
	fp.write(pack(">B", 3))				# 颜色分量ID: 3
	fp.write(pack(">H", 0x1101))		# 水平垂直采样因子: 1  使用的量化表ID: 1
	# DHT_DC0
	fp.write(pack(">H", 0xffc4))
	fp.write(pack(">H", len(std_huffman_DC0)+3))	# 哈夫曼表字节数
	fp.write(pack(">B", 0x00))						# DC0
	for item in std_huffman_DC0:
		fp.write(pack(">B", item))					# 哈夫曼表内容
	# DHT_AC0
	fp.write(pack(">H", 0xffc4))
	fp.write(pack(">H", len(std_huffman_AC0)+3))	# 哈夫曼表字节数
	fp.write(pack(">B", 0x10))						# DC0
	for item in std_huffman_AC0:
		fp.write(pack(">B", item))					# 哈夫曼表内容
	# DHT_DC1
	fp.write(pack(">H", 0xffc4))
	fp.write(pack(">H", len(std_huffman_DC1)+3))	# 哈夫曼表字节数
	fp.write(pack(">B", 0x01))						# DC0
	for item in std_huffman_DC1:
		fp.write(pack(">B", item))					# 哈夫曼表内容
	# DHT_AC1
	fp.write(pack(">H", 0xffc4))
	fp.write(pack(">H", len(std_huffman_AC1)+3))	# 哈夫曼表字节数
	fp.write(pack(">B", 0x11))						# DC0
	for item in std_huffman_AC0:
		fp.write(pack(">B", item))					# 哈夫曼表内容
	# SOS
	fp.write(pack(">H", 0xffda))
	fp.write(pack(">H", 12))			# 扫描开始信息字节数
	fp.write(pack(">B", 3))				# 颜色分量数: 3
	fp.write(pack(">H", 0x0100))		# 颜色分量1 DC、AC使用的哈夫曼表ID
	fp.write(pack(">H", 0x0211))		# 颜色分量2 DC、AC使用的哈夫曼表ID
	fp.write(pack(">H", 0x0311))		# 颜色分量3 DC、AC使用的哈夫曼表ID
	fp.write(pack(">B", 0x00))
	fp.write(pack(">B", 0x3f))
	fp.write(pack(">B", 0x00))

	fp.close()