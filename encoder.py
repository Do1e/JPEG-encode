import cv2
import numpy as np
import argparse

import tables
import encoder_utils


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='input file', required=True)
parser.add_argument('-o', '--output', help='output file', required=True)
args = parser.parse_args()

BGR = cv2.imread(args.input)
# cv2.imwrite('./test/ref.jpg', BGR, [cv2.IMWRITE_JPEG_QUALITY, 100, cv2.IMWRITE_JPEG_PROGRESSIVE, 1])
# 色彩空间变换
YCrCb = cv2.cvtColor(BGR, cv2.COLOR_BGR2YCrCb)
h, w, d = YCrCb.shape

# 转换图片大小，必须能被切分成8*8的小块
if((h % 8 == 0) and (w % 8 == 0)):
	nblock = h * w // 64
else:
	h = h // 8 * 8
	w = w // 8 * 8
	YCrCb = cv2.resize(YCrCb, [h, w], cv2.INTER_CUBIC)
	nblock = h * w // 64
# 后续运算会转换变为有符号数，因此做数据转换
npdata = np.array(YCrCb, np.int16)
del BGR
del YCrCb

# 写入jpeg格式的译码信息
encoder_utils.write_head(args.output, h, w)
# 换算出哈夫曼字典
DC0 = encoder_utils.DHT2tbl(tables.std_huffman_DC0)
DC1 = encoder_utils.DHT2tbl(tables.std_huffman_DC1)
AC0 = encoder_utils.DHT2tbl(tables.std_huffman_AC0)
AC1 = encoder_utils.DHT2tbl(tables.std_huffman_AC1)

buffer = encoder_utils.Byte_Buffer()
fp = open(args.output, 'ab')
last_block_ydc = 0
last_block_cbdc = 0
last_block_crdc = 0
now = 0

for i in range(0, h, 8):
	for j in range(0, w, 8):
		print("%5.2f%%" % (now / nblock*100), end="")
		now += 1

		# Y通道
		now_block = npdata[i:i+8, j:j+8, 0] - 128								# 取出一个8*8块并减去128
		now_block_dct = cv2.dct(np.float32(now_block))							# DCT变换
		now_block_qut = encoder_utils.quantize(now_block_dct, 0)				# 量化
		now_block_zz = encoder_utils.block2zz(now_block_qut)					# zigzag扫描
		now_block_dc = now_block_zz[0] - last_block_ydc 						# 直流分量差分形式记录
		last_block_ydc = now_block_zz[0]										# 记录本次量
		now_block_ac = encoder_utils.RLE(now_block_zz[1:])						# 交流分量对0进行行程编码
		buffer.append_buffer(encoder_utils.write_num(-1, now_block_dc, DC0))	# 根据编码方式写入直流数据
		for l in range(0, len(now_block_ac), 2):								# 根据编码方式写入交流数据
			buffer.append_buffer(encoder_utils.write_num( \
				now_block_ac[l], now_block_ac[l+1], AC0))
		buffer.flush(fp)														# 将缓存写入文件

		# Cb通道
		now_block = npdata[i:i+8, j:j+8, 2] - 128
		now_block_dct = cv2.dct(np.float32(now_block))
		now_block_qut = encoder_utils.quantize(now_block_dct, 2)
		now_block_zz = encoder_utils.block2zz(now_block_qut)
		now_block_dc = now_block_zz[0] - last_block_cbdc
		last_block_cbdc = now_block_zz[0]
		now_block_ac = encoder_utils.RLE(now_block_zz[1:])
		buffer.append_buffer(encoder_utils.write_num(-1, now_block_dc, DC1))
		for l in range(0, len(now_block_ac), 2):
			buffer.append_buffer(encoder_utils.write_num( \
				now_block_ac[l], now_block_ac[l+1], AC1))
		buffer.flush(fp)

		# Cr通道
		now_block = npdata[i:i+8, j:j+8, 1] - 128
		now_block_dct = cv2.dct(np.float32(now_block))
		now_block_qut = encoder_utils.quantize(now_block_dct, 1)
		now_block_zz = encoder_utils.block2zz(now_block_qut)
		now_block_dc = now_block_zz[0] - last_block_crdc
		last_block_crdc = now_block_zz[0]
		now_block_ac = encoder_utils.RLE(now_block_zz[1:])
		buffer.append_buffer(encoder_utils.write_num(-1, now_block_dc, DC1))
		for l in range(0, len(now_block_ac), 2):
			buffer.append_buffer(encoder_utils.write_num( \
				now_block_ac[l], now_block_ac[l+1], AC1))
		buffer.flush(fp)
		print("\b\b\b\b\b\b\b\b\b\b", end="")

if len(buffer.temp) != 0:
	buffer.append_str('0' * (8 - len(buffer.temp)))
	buffer.flush(fp)
# EOI
fp.write(b'\xff\xd9')
fp.close()