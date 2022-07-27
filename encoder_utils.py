from __future__ import annotations
import numpy as np
from struct import pack

import tables
import common_utils

# 比特缓存，每8位保存到列表中
class Byte_Buffer():
	def __init__(self) -> None:
		self.buffer = []
		self.temp = ""
	def size(self) -> int:
		return len(self.buffer)
	def tolist(self) -> list:
		return self.buffer
	def tonumpy(self) -> np.ndarray:
		return np.array(self.buffer)
	def append_bit(self, data: int) -> None:
		if data is None:
			return
		if data != 0 and data != 1:
			raise ValueError("data must be 0 or 1")
		self.temp += str(data)
		if len(self.temp) == 8:
			self.buffer.append(int(self.temp, 2))
			self.temp = ""
	def append_byte(self, data: int) -> None:
		if data is None:
			return
		if data < 0 or data > 255:
			raise ValueError("data must be 0 ~ 255")
		if len(self.temp) == 0:
			self.buffer.append(data)
		else:
			lenTemp = len(self.temp)
			self.temp += (bin(data >> lenTemp)[2:]).zfill(8 - lenTemp)
			data &= (1 << lenTemp) - 1
			self.buffer.append(int(self.temp, 2))
			self.temp = (bin(data)[2:]).zfill(lenTemp)
	def append_str(self, data: str) -> None:
		if data is None:
			return
		lenByte = len(data) // 8
		for i in range(lenByte):
			self.append_byte(int(data[i * 8 : (i + 1) * 8], 2))
		for i in range(8 * lenByte, len(data)):
			if data[i] == '0':
				self.append_bit(0)
			elif data[i] == '1':
				self.append_bit(1)
			else:
				raise ValueError("data must be 0 or 1")
	def append_buffer(self, data: Byte_Buffer) -> None:
		if data is None:
			return
		if len(self.temp) == 0:
			self.buffer.extend(data.buffer)
			self.temp = data.temp
		else:
			if data.size() == 0:
				self.temp += data.temp
				if len(self.temp) >= 8:
					self.buffer.append(int(self.temp[:8], 2))
					self.temp = self.temp[8:]
				return
			lenTemp = len(self.temp)
			self.temp += (bin(data.buffer[0] >> lenTemp)[2:]).zfill(8 - lenTemp)
			self.buffer.append(int(self.temp, 2))
			for i in range(0, data.size() - 1):
				data.buffer[i] = (data.buffer[i] & ((1 << lenTemp) - 1)) << (8 - lenTemp)
				data.buffer[i] |= data.buffer[i + 1] >> lenTemp
				self.buffer.append(data.buffer[i])
			data.buffer[-1] &= ((1 << lenTemp) - 1)
			self.temp = (bin(data.buffer[-1])[2:] + data.temp).zfill(lenTemp + len(data.temp))
			if len(self.temp) >= 8:
				self.buffer.append(int(self.temp[:8], 2))
				self.temp = self.temp[8:]
	def flush(self, fp: open, header=False) -> None:
		for item in self.buffer:
			fp.write(pack('>B', item))
			if item == 255 and not header:
				fp.write(pack('>B', 0))
		self.buffer = []
	def flush_api(self, List: list, header=False) -> None:
		for item in self.buffer:
			List.append(item)
			if item == 255 and not header:
				List.append(0)
		self.buffer = []
	def __str__(self) -> str:
		return str([hex(item) for item in self.buffer]) + " " + self.temp

# zigzag扫描
# block: 当前8*8块的数据
def block2zz(block: np.ndarray) -> np.ndarray:
	re = np.empty(64, block.dtype)
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
def write_head(h: int, w: int, gray=False) -> Byte_Buffer:
	res = Byte_Buffer()
	# SOI
	res.append_str(bin(0xffd8)[2:].zfill(16))
	# APP0
	res.append_str(bin(0xffe0)[2:].zfill(16))
	res.append_str(bin(16)[2:].zfill(16))			# APP0字节数
	res.append_str(bin(0x4a464946)[2:].zfill(32))	# JFIF
	res.append_byte(0)								# 0
	res.append_str(bin(0x0101)[2:].zfill(16))		# 版本号: 1.1
	res.append_byte(0)								# 像素密度单位: 像素/英寸
	res.append_str(bin(0x00010001)[2:].zfill(32))	# XY方向像素密度
	res.append_str(bin(0x0000)[2:].zfill(16))		# 无缩略图信息
	# DQT_0
	res.append_str(bin(0xffdb)[2:].zfill(16))
	res.append_str(bin(64+3)[2:].zfill(16))			# 量化表字节数
	res.append_byte(0x00)							# 量化表精度: 8bit(0)  量化表ID: 0
	tbl = block2zz(tables.std_luminance_quant_tbl)
	for item in tbl:
		res.append_byte(item)						# 量化表0内容
	if not gray:
		# DQT_1
		res.append_str(bin(0xffdb)[2:].zfill(16))
		res.append_str(bin(64+3)[2:].zfill(16))			# 量化表字节数
		res.append_byte(0x01)							# 量化表精度: 8bit(0)  量化表ID: 1
		tbl = block2zz(tables.std_chrominance_quant_tbl)
		for item in tbl:
			res.append_byte(item)						# 量化表1内容
	# SOF0
	res.append_str(bin(0xffc0)[2:].zfill(16))
	res.append_str(bin(11 if gray else 17)[2:].zfill(16))	# 帧图像信息字节数
	res.append_byte(8)										# 精度: 8bit
	res.append_str(bin(h)[2:].zfill(16))					# 图像高度
	res.append_str(bin(w)[2:].zfill(16))					# 图像宽度
	res.append_byte(1 if gray else 3)						# 颜色分量数: 3(YCrCb)
	res.append_byte(1)										# 颜色分量ID: 1
	res.append_str(bin(0x1100)[2:].zfill(16))				# 水平垂直采样因子: 1  使用的量化表ID: 0
	if not gray:
		res.append_byte(2)									# 颜色分量ID: 2
		res.append_str(bin(0x1101)[2:].zfill(16))			# 水平垂直采样因子: 1  使用的量化表ID: 1
		res.append_byte(3)									# 颜色分量ID: 3
		res.append_str(bin(0x1101)[2:].zfill(16))			# 水平垂直采样因子: 1  使用的量化表ID: 1
	# DHT_DC0
	res.append_str(bin(0xffc4)[2:].zfill(16))
	res.append_str(bin(len(tables.std_huffman_DC0)+3)[2:].zfill(16))	# 哈夫曼表字节数
	res.append_byte(0x00)												# DC0
	for item in tables.std_huffman_DC0:
		res.append_byte(item)											# 哈夫曼表内容
	# DHT_AC0
	res.append_str(bin(0xffc4)[2:].zfill(16))
	res.append_str(bin(len(tables.std_huffman_AC0)+3)[2:].zfill(16))	# 哈夫曼表字节数
	res.append_byte(0x10)												# AC0
	for item in tables.std_huffman_AC0:
		res.append_byte(item)											# 哈夫曼表内容
	if not gray:
		# DHT_DC1
		res.append_str(bin(0xffc4)[2:].zfill(16))
		res.append_str(bin(len(tables.std_huffman_DC1)+3)[2:].zfill(16))	# 哈夫曼表字节数
		res.append_byte(0x01)												# DC1
		for item in tables.std_huffman_DC1:
			res.append_byte(item)											# 哈夫曼表内容
		# DHT_AC1
		res.append_str(bin(0xffc4)[2:].zfill(16))
		res.append_str(bin(len(tables.std_huffman_AC1)+3)[2:].zfill(16))	# 哈夫曼表字节数
		res.append_byte(0x11)												# AC1
		for item in tables.std_huffman_AC1:
			res.append_byte(item)											# 哈夫曼表内容
	# SOS
	res.append_str(bin(0xffda)[2:].zfill(16))
	res.append_str(bin(8 if gray else 12)[2:].zfill(16))	# 扫描开始信息字节数
	res.append_byte(1 if gray else 3)						# 颜色分量数: 3
	res.append_str(bin(0x0100)[2:].zfill(16))				# 颜色分量1 DC、AC使用的哈夫曼表ID
	if not gray:
		res.append_str(bin(0x0211)[2:].zfill(16))			# 颜色分量2 DC、AC使用的哈夫曼表ID
		res.append_str(bin(0x0311)[2:].zfill(16))			# 颜色分量3 DC、AC使用的哈夫曼表ID
	res.append_str(bin(0x003f00)[2:].zfill(24))				# 固定值
	return res

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
		if(AClist[i] == 0 and cnt != 15):
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

# 特殊的二进制编码格式
# num: 待编码的数字
def tobin(num: int) -> Byte_Buffer:
	res = Byte_Buffer()
	if num == 0:
		return None
	elif num > 0:
		res.append_str(bin(num)[2:])
	else:
		lenStr = len(bin(-num)[2:])
		num = (1 << lenStr) - 1 + num
		res.append_str(bin(num)[2:].zfill(lenStr))
	return res

# 根据编码方式写入数据
# n: 数据前面0的个数(-1代表DC)
# num: 待写入的数据
# tbl: 范式哈夫曼编码字典
def write_num(n: int, num: int, tbl: list) -> None:
	res = Byte_Buffer()
	bit = len(bin(num)[2:])
	if num <= 0:
		bit -= 1
	# 如果是DC编码
	if n == -1:
		if bit > 11 or bit < 0:
			raise ValueError("DC编码长度数值超出范围")
		tnum = bit
	# 如果是AC编码
	else:
		if (n > 15) or (n < 0) or (bit < 0) or (bit > 11) \
			or (((n != 0) and (n != 15)) and (bit == 0)):
			raise ValueError("AC编码长度数值超出范围")
		tnum = n * 10 + bit + (0 if(n != 15) else 1)
	res.append_str(tbl[tnum].str_code)
	res.append_buffer(tobin(num))
	return res
