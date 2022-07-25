from base64 import decode
import cv2
import numpy as np
import argparse

import decoder_utils
import common_utils


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='input file', required=True)
parser.add_argument('-o', '--output', help='output file', required=True)
args = parser.parse_args()

with open(args.input, 'rb') as f:
	data = f.read()
data = np.frombuffer(data, np.uint8)
nowIdx = decoder_utils.decode_head(data)
if nowIdx is None or data[-2] != 0xff or data[-1] != 0xd9:
	print("Error: unsupported jpeg format")
	exit(1)
data = data[nowIdx:-2]

resimg = np.zeros((decoder_utils.h, decoder_utils.w, decoder_utils.d), np.uint8)
last_block_dc = [0, 0, 0]
nowLeft = 0
nowRight = 0
nowx = 0
nowy = 0
nowIdx, buffer = decoder_utils.load_buffer(data, '', 0)

while nowIdx < data.shape[0]:
	nowblock_decoded = np.zeros((8, 8, decoder_utils.d), np.int16)
	for i in range(1, decoder_utils.d+1):
		nowblock = np.zeros((64), np.int16)
		nowblockIdx = 0
		# decode DC
		while 1:
			try:
				var = decoder_utils.DC_color_huffman_dict[i][buffer[nowLeft:nowRight+1]]
				num = int(buffer[nowRight+1:nowRight+var+1], 2)
				if buffer[nowRight+1] == '0':
					num = -((1<<var) - 1) + num
				nowblock[nowblockIdx] = num
				nowblockIdx += 1
				nowLeft = nowRight + var + 1
				nowRight = nowLeft
				if nowRight >= len(buffer):
					nowIdx, buffer = decoder_utils.load_buffer(data, buffer, nowIdx)
				break
			except KeyError:
				nowRight += 1
				if nowRight >= len(buffer):
					nowIdx, buffer = decoder_utils.load_buffer(data, buffer, nowIdx)
		last_block_dc[i-1] = nowblock[0]
		buffer = buffer[nowRight:]
		nowLeft = 0
		nowRight = 0
		# decode AC
		while 1:
			try:
				var = decoder_utils.AC_color_huffman_dict[i][buffer[nowLeft:nowRight+1]]
				num0 = var >> 4
				num1 = var & 0xf
				nowblockIdx += num0
				if num0 == 15 and num1 == 0:
					nowblockIdx += 1
				elif num0 == 0 and num1 == 0:
					break
				else:
					num = int(buffer[nowRight+1:nowRight+num1+1], 2)
					if buffer[nowRight+1] == '0':
						num = -((1<<num1) - 1) + num
					nowblock[nowblockIdx] = num
					nowblockIdx += 1
				nowLeft = nowRight + num1 + 1
				nowRight = nowLeft
				if nowRight >= len(buffer):
					nowIdx, buffer = decoder_utils.load_buffer(data, buffer, nowIdx)
				if nowblockIdx == 64:
					break
				assert nowblockIdx < 64
			except KeyError:
				nowRight += 1
				if nowRight >= len(buffer):
					nowIdx, buffer = decoder_utils.load_buffer(data, buffer, nowIdx)
		nowblock_dezz = decoder_utils.zz2block(nowblock)
		nowblock_dequt = nowblock_dezz * decoder_utils.color_quant_dict[i]
		nowblock_idct = cv2.idct(nowblock_dequt).round() + 128
		nowblock_idct[nowblock_idct < 0] = 0
		nowblock_idct[nowblock_idct > 255] = 255
		nowblock_decoded[:, :, i-1] = nowblock_idct.astype(np.uint8)
	resimg[nowx:nowx+8, nowy:nowy+8, :] = cv2.cvtColor(nowblock_idct, cv2.COLOR_YCrCb2BGR)