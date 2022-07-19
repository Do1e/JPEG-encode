import numpy as np
from struct import pack

import tables

# 记录哈夫曼字典的类
# symbol: 原始数据
# code: 对应的编码数据
# n_bit: 编码的二进制位数
class Sym_Code():
	def __init__(self, symbol: int, code: int, n_bit: int) -> None:
		self.symbol = symbol
		self.code = code
		self.n_bit = n_bit
	"""定义输出形式"""
	def __str__(self):
		return "0x{:0>2x}    |  {}".format(self.symbol, str(self.code))
	"""定义排序依据"""
	def __eq__(self, other):
		return self.symbol == other.symbol
	def __le__(self, other):
		return self.symbol < other.symbol
	def __gt__(self, other):
		return self.symbol > other.symbol

# zigzag扫描
# block: 当前8*8块的数据
def block2zz(block: np.ndarray) -> np.ndarray:
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

# 写入jpeg格式的译码信息
# filename: 输出文件名
# h: 图片高度
# w: 图片宽度
def write_head(filename: str, h: int, w: int) -> None:
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
	tbl = block2zz(tables.std_luminance_quant_tbl)
	for item in tbl:
		fp.write(pack(">B", item))		# 量化表0内容
	# DQT_1
	fp.write(pack(">H", 0xffdb))
	fp.write(pack(">H", 64+3))			# 量化表字节数
	fp.write(pack(">B", 0x01))			# 量化表精度: 8bit(0)  量化表ID: 1
	tbl = block2zz(tables.std_chrominance_quant_tbl)
	for item in tbl:
		fp.write(pack(">B", item))		# 量化表1内容
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
	fp.write(pack(">H", len(tables.std_huffman_DC0)+3))	# 哈夫曼表字节数
	fp.write(pack(">B", 0x00))							# DC0
	for item in tables.std_huffman_DC0:
		fp.write(pack(">B", item))						# 哈夫曼表内容
	# DHT_AC0
	fp.write(pack(">H", 0xffc4))
	fp.write(pack(">H", len(tables.std_huffman_AC0)+3))	# 哈夫曼表字节数
	fp.write(pack(">B", 0x10))							# AC0
	for item in tables.std_huffman_AC0:
		fp.write(pack(">B", item))						# 哈夫曼表内容
	# DHT_DC1
	fp.write(pack(">H", 0xffc4))
	fp.write(pack(">H", len(tables.std_huffman_DC1)+3))	# 哈夫曼表字节数
	fp.write(pack(">B", 0x01))							# DC1
	for item in tables.std_huffman_DC1:
		fp.write(pack(">B", item))						# 哈夫曼表内容
	# DHT_AC1
	fp.write(pack(">H", 0xffc4))
	fp.write(pack(">H", len(tables.std_huffman_AC1)+3))	# 哈夫曼表字节数
	fp.write(pack(">B", 0x11))							# AC1
	for item in tables.std_huffman_AC1:
		fp.write(pack(">B", item))						# 哈夫曼表内容
	# SOS
	fp.write(pack(">H", 0xffda))
	fp.write(pack(">H", 12))			# 扫描开始信息字节数
	fp.write(pack(">B", 3))				# 颜色分量数: 3
	fp.write(pack(">H", 0x0100))		# 颜色分量1 DC、AC使用的哈夫曼表ID
	fp.write(pack(">H", 0x0211))		# 颜色分量2 DC、AC使用的哈夫曼表ID
	fp.write(pack(">H", 0x0311))		# 颜色分量3 DC、AC使用的哈夫曼表ID
	fp.write(pack(">B", 0x00))			# 固定值
	fp.write(pack(">B", 0x3f))
	fp.write(pack(">B", 0x00))
	fp.close()

# 量化
# block: 当前8*8块的数据
# dim: 维度  0:Y  1:Cr  2:Cb
def quantize(block: np.ndarray, dim: int) -> np.ndarray:
	if(dim == 0):
		# 使用亮度量化表
		qarr = tables.std_luminance_quant_tbl
	else:
		# 使用色度量化表
		qarr = tables.std_chrominance_quant_tbl
	return (block.astype(np.float32) / qarr.astype(np.float32)).round().astype(np.int16)

# 0的行程编码
# AClist: 要编码的交流数据
def RLE(AClist: np.ndarray) -> np.ndarray:
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
	re.append(0)
	re.append(0)
	return np.array(re, np.int16)