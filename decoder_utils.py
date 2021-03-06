import numpy as np
import common_utils

global h, w, d
global quant_dict, color_quant_dict
quant_dict = {}
color_quant_dict = {}
global DHTs_DC, DHTs_AC
DHTs_DC = {}
DHTs_AC = {}
global DC_color_huffman_dict, AC_color_huffman_dict
DC_color_huffman_dict = {}
AC_color_huffman_dict = {}

def zz2block(block: list) -> np.ndarray:
	re = np.empty((8, 8), np.int16)
	# 当前在block的位置
	pos = np.array([0, 0])
	# 定义四个扫描方向
	R = np.array([0, 1])
	LD = np.array([1, -1])
	D = np.array([1, 0])
	RU = np.array([-1, 1])
	for i in range(0, 64):
		# re[i] = block[pos[0], pos[1]]
		re[pos[0], pos[1]] = block[i]
		if(((pos[0] == 0) or (pos[0] == 7)) and (pos[1] % 2 == 0)):
			pos = pos + R
		elif(((pos[1] == 0) or (pos[1] == 7)) and (pos[0] % 2 == 1)):
			pos = pos + D
		elif((pos[0] + pos[1]) % 2 == 0):
			pos = pos + RU
		else:
			pos = pos + LD
	return re

def decode_DHT(data: np.ndarray) -> int:
	global DHTs_DC, DHTs_AC
	if data[0] >> 4 == 0:
		nowtbl = common_utils.DHT2tbl(data[1:])
		nowdict = {}
		for item in nowtbl:
			nowdict[item.str_code] = item.symbol
		DHTs_DC[data[0] & 0x0f] = nowdict.copy()
	elif data[0] >> 4 == 1:
		nowtbl = common_utils.DHT2tbl(data[1:])
		nowdict = {}
		for item in nowtbl:
			nowdict[item.str_code] = item.symbol
		DHTs_AC[data[0] & 0x0f] = nowdict.copy()
	else:
		return None
	return 0

def decode_head(data: np.ndarray) -> int:
	gray=False
	nowIdx = 0
	# SOI
	assert data[nowIdx] == 0xff and data[nowIdx+1] == 0xd8, 'SOI not found'
	# APP0
	nowIdx += 2
	assert data[nowIdx] == 0xff and data[nowIdx+1] == 0xe0, 'APP0 not found'
	nowLen = (data[nowIdx+2] << 8) | data[nowIdx+3]
	nowIdx += 2 + nowLen
	while data[nowIdx] == 0xff and data[nowIdx+1] >= 0xe1 and data[nowIdx+1] <= 0xef:
		nowIdx += 2
		nowLen = (data[nowIdx+2] << 8) | data[nowIdx+3]
		nowIdx += 2 + nowLen
	# DQT_0
	assert data[nowIdx] == 0xff and data[nowIdx+1] == 0xdb, 'DQT_0 not found'
	nowLen = (data[nowIdx+2] << 8) | data[nowIdx+3]
	assert nowLen == 67 and data[nowIdx+4] >> 4 == 0, 'only support 8bit quantization table'
	global quant_dict
	nowtable = zz2block(data[nowIdx+5:nowIdx+69])
	quant_dict[data[nowIdx+4] & 0x0f] = nowtable.copy()
	nowIdx += 2 + nowLen
	if data[nowIdx] != 0xff or data[nowIdx+1] != 0xdb:
		gray = True
	if not gray:
		# DQT_1
		assert data[nowIdx] == 0xff and data[nowIdx+1] == 0xdb, 'DQT_1 not found'
		nowLen = (data[nowIdx+2] << 8) | data[nowIdx+3]
		assert nowLen == 67 and data[nowIdx+4] >> 4 == 0, 'only support 8bit quantization table'
		nowtable = zz2block(data[nowIdx+5:nowIdx+69])
		quant_dict[data[nowIdx+4] & 0x0f] = nowtable.copy()
		nowIdx += 2 + nowLen
	# SOF0
	assert data[nowIdx] == 0xff and data[nowIdx+1] == 0xc0, 'SOF0 not found'
	nowLen = (data[nowIdx+2] << 8) | data[nowIdx+3]
	assert data[nowIdx+4] == 8, 'only support 8bit color'
	global h, w, d
	h = (data[nowIdx+5] << 8) | data[nowIdx+6]
	w = (data[nowIdx+7] << 8) | data[nowIdx+8]
	d = data[nowIdx+9]
	if gray:
		assert h > 0 and w > 0 and d == 1, 'invalid image size'
		assert data[nowIdx+11] == 17, 'invalid sampling'
	else:
		assert h > 0 and w > 0 and d == 3, 'invalid image size'
		assert data[nowIdx+11] == 17 and data[nowIdx+14] == 17 and data[nowIdx+17] == 17, 'invalid sampling'
	global color_quant_dict
	color_quant_dict[data[nowIdx+10]] = quant_dict[data[nowIdx+12]]
	if not gray:
		color_quant_dict[data[nowIdx+13]] = quant_dict[data[nowIdx+15]]
		color_quant_dict[data[nowIdx+16]] = quant_dict[data[nowIdx+18]]
	nowIdx += 2 + nowLen
	# DHT
	while data[nowIdx] == 0xff and data[nowIdx+1] == 0xc4:
		nowLen = (data[nowIdx+2] << 8) | data[nowIdx+3]
		status = decode_DHT(data[nowIdx+4:nowIdx+2+nowLen])
		assert status is not None, 'DHT decode error'
		nowIdx += 2 + nowLen
	# SOS
	assert data[nowIdx] == 0xff and data[nowIdx+1] == 0xda, 'SOS not found'
	nowLen = (data[nowIdx+2] << 8) | data[nowIdx+3]
	if gray:
		assert data[nowIdx+4] == 1, 'only support RGB and Gray image'
	else:
		assert data[nowIdx+4] == 3, 'only support RGB and Gray image'
	global DC_color_huffman_dict
	for i in range(d):
		DC_color_huffman_dict[data[nowIdx+5+2*i]] = DHTs_DC[data[nowIdx+6+2*i] >> 4]
		AC_color_huffman_dict[data[nowIdx+5+2*i]] = DHTs_AC[data[nowIdx+6+2*i] & 0x0f]
	nowIdx += 2 + nowLen
	return nowIdx

def load_buffer(data:np.ndarray ,buffer: str, nowIdx: int) -> list:
	loadLen = min(data.shape[0] - nowIdx, 8)
	for i in range(nowIdx, nowIdx + loadLen):
		if data[i] != 0 or data[i-1] != 0xff:
			buffer += bin(data[i])[2:].zfill(8)
	return nowIdx + loadLen, buffer