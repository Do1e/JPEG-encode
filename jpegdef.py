# -*- coding: utf-8 -*-
import numpy as np
import os
from struct import pack
from gen_jpeg_standard_quantization_table import *
# # 亮度量化表
# std_luminance_quant_tbl = np.array(
# 	[
# 		[16, 11, 10, 16, 24, 40, 51, 61],
# 		[12, 12, 14, 19, 26, 58, 60, 55],
# 		[14, 13, 16, 24, 40, 57, 69, 56],
# 		[14, 17, 22, 29, 51, 87, 80, 62],
# 		[18, 22, 37, 56, 68,109,103, 77],
# 		[24, 35, 55, 64, 81,104,113, 92],
# 		[49, 64, 78, 87,103,121,120,101],
# 		[72, 92, 95, 98,112,100,103, 99]
# 	],
# 	np.uint8
# )
# # 色度量化表
# std_chrominance_quant_tbl = np.array(
# 	[
# 		[17, 18, 24, 47, 99, 99, 99, 99],
# 		[18, 21, 26, 66, 99, 99, 99, 99],
# 		[24, 26, 56, 99, 99, 99, 99, 99],
# 		[47, 66, 99, 99, 99, 99, 99, 99],
# 		[99, 99, 99, 99, 99, 99, 99, 99],
# 		[99, 99, 99, 99, 99, 99, 99, 99],
# 		[99, 99, 99, 99, 99, 99, 99, 99],
# 		[99, 99, 99, 99, 99, 99, 99, 99]
# 	],
# 	np.uint8
# )
std_luminance_quant_tbl ,std_chrominance_quant_tbl = gen_quant_table_by_quality(30)
# 亮度直流量范式哈夫曼编码表
std_huffman_DC0 = np.array(
	[0, 0, 7, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
	 4, 5, 3, 2, 6, 1, 0, 7, 8, 9, 10, 11],
	np.uint8
)
# 色度直流量范式哈夫曼编码表
std_huffman_DC1 = np.array(
	[0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
	 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
	np.uint8
)
# 亮度交流量范式哈夫曼编码表
std_huffman_AC0 = np.array(
	[0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125,
	 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
	 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
	 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
	 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
	 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
	 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
	 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
	 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
	 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
	 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
	 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
	 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
	 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
	 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
	 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
	 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
	 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
	 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
	 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
	 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
	 0xf9, 0xfa],
	np.uint8
)
# 色度交流量范式哈夫曼编码表
std_huffman_AC1 = np.array(
	[0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119,
	 0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
	 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
	 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
	 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
	 0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
	 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
	 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
	 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
	 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
	 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
	 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
	 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
	 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
	 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
	 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
	 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
	 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
	 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
	 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
	 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
	 0xf9, 0xfa],
	np.uint8
)

# 记录哈夫曼字典的类
# symbol: 原始数据
# code: 对应的编码数据
# n_bit: 编码的二进制位数
# str_code: 编码的二进制数据
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

# 量化
# block: 当前8*8块的数据
# dim: 维度  0:Y  1:Cr  2:Cb
def quantize(block, dim):
	if(dim == 0):
		# 使用亮度量化表
		qarr = std_luminance_quant_tbl
	else:
		# 使用色度量化表
		qarr = std_chrominance_quant_tbl
	return (block / qarr).round().astype(np.int16)

# zigzag扫描
# block: 当前8*8块的数据
def block2zz(block):
	re = np.empty(64, np.int16)
	# 当前在block的位置
	pos = np.array([0, 0])
	# 定义四个扫描方向
	R = np.array([0, 1])
	LD = np.array([1, -1])
	D = np.array([1, 0])
	RU = np.array([-1, 1])
	for i in range(0, 64):
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

# 0的行程编码
# AClist: 要编码的交流数据
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
	# 删除末尾的所有[15 0]
	while(re[-1] == 0):
		re.pop()
		re.pop()
		if(len(re) == 0):
			break
	# 在结尾添加两个0作为结束标记
	if(AClist[-1] == 0):
		re.extend([0, 0])
	return np.array(re, np.int16)

# 将范式哈夫曼编码表转换为哈夫曼字典
# data: 定义的范式哈夫曼编码表
def DHT2tbl(data):
	numbers = data[0:16]				# 1~16bit长度的编码对应的个数
	symbols = data[16:len(data)]		# 原数据
	if(sum(numbers) != len(symbols)):	# 判断是否为正确的范式哈夫曼编码表
		print("Wrong DHT!")
		exit()
	code = 0
	SC = []								# 记录字典的列表
	for n_bit in range(1, 17):
		# 按范式哈夫曼编码规则换算出字典
		for symbol in symbols[sum(numbers[0:n_bit-1]):sum(numbers[0:n_bit])]:
			SC.append(Sym_Code(symbol, code, n_bit))
			code += 1
		code <<= 1
	return sorted(SC)

# 写入jpeg格式的译码信息
# filename: 输出文件名
# h: 图片高度
# w: 图片宽度
def write_head(filename, h, w):
	# 二进制写入形式打开文件(覆盖)
	fp = open(filename, "wb")

	# SOI
	fp.write(pack(">H", 0xffd8))
	# APP0
	fp.write(pack(">H", 0xffe0))
	fp.write(pack(">H", 16))			# APP0字节数
	fp.write(pack(">L", 0x4a464946))	# JFIF
	fp.write(pack(">B", 0))				# 0
	fp.write(pack(">H", 0x0101))		# 版本号: 1.1
	fp.write(pack(">B", 0x01))			# 像素密度单位: 像素/英寸
	fp.write(pack(">L", 0x00480048))	# XY方向像素密度
	fp.write(pack(">H", 0x0000))		# 无缩略图信息
	# DQT_0
	fp.write(pack(">H", 0xffdb))
	fp.write(pack(">H", 64+3))			# 量化表字节数
	fp.write(pack(">B", 0x00))			# 量化表精度: 8bit(0)  量化表ID: 0
	tbl = block2zz(std_luminance_quant_tbl)
	for item in tbl:
		fp.write(pack(">B", item))	# 量化表0内容
	# DQT_1
	fp.write(pack(">H", 0xffdb))
	fp.write(pack(">H", 64+3))			# 量化表字节数
	fp.write(pack(">B", 0x01))			# 量化表精度: 8bit(0)  量化表ID: 1
	tbl = block2zz(std_chrominance_quant_tbl)
	for item in tbl:
		fp.write(pack(">B", item))	# 量化表1内容
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
	fp.write(pack(">B", 0x10))						# AC0
	for item in std_huffman_AC0:
		fp.write(pack(">B", item))					# 哈夫曼表内容
	# DHT_DC1
	fp.write(pack(">H", 0xffc4))
	fp.write(pack(">H", len(std_huffman_DC1)+3))	# 哈夫曼表字节数
	fp.write(pack(">B", 0x01))						# DC1
	for item in std_huffman_DC1:
		fp.write(pack(">B", item))					# 哈夫曼表内容
	# DHT_AC1
	fp.write(pack(">H", 0xffc4))
	fp.write(pack(">H", len(std_huffman_AC1)+3))	# 哈夫曼表字节数
	fp.write(pack(">B", 0x11))						# AC1
	for item in std_huffman_AC1:
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
	fp.write(pack(">B", 0x00))			# 固定值
	fp.close()

# 特殊的二进制编码格式
# num: 待编码的数字
def tobin(num):
	s = ""
	if(num > 0):
		while(num != 0):
			s += '0' if(num % 2 == 0) else '1'
			num = int(num / 2)
		s = s[::-1]
	elif(num < 0):
		num = -num
		while(num != 0):
			s += '1' if(num % 2 == 0) else '0'
			num = int(num / 2)
		s = s[::-1]
	return s

# 根据编码方式写入数据
# s: 未写入文件的二进制数据
# n: 数据前面0的个数(-1代表DC)
# num: 待写入的数据
# tbl: 范式哈夫曼编码字典
def write_num(s, n, num, tbl):
	bit = 0
	tnum = num
	while(tnum != 0):
		bit += 1
		tnum = int(tnum / 2)
	if(n == -1):					# DC
		tnum = bit
		if(tnum > 11):
			print("Write DC data Error")
			exit()
	else:							# AC
		if((n > 15) or (bit > 11) or (((n != 0) and (n != 15)) and (bit == 0))):
			print("Write AC data Error")
			exit()
		tnum = n * 10 + bit + (0 if(n != 15) else 1)
	# 范式哈夫曼编码记录0的个数(AC)以及num的bit长度
	s += tbl[tnum].str_code
	# 特殊形式的数据存储num
	s += tobin(num)
	return s
