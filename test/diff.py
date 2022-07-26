import cv2
import numpy as np
from PIL import Image
import os

import sys
sys.path.append('../')
sys.path.append('./')
from encoder_api import JPEG_encoder
from decoder_api import JPEG_decoder

def PSNR(img1: np.ndarray, img2: np.ndarray) -> float:
	mse = np.mean((img1 - img2) ** 2)
	if mse == 0:
		return 99.00
	return 10 * np.log10(255 ** 2 / mse)

for id in range(1, 6):
	fpsnr = open('./test/' + str(id) + '_psnr.txt', 'w')
	fsize = open('./test/' + str(id) + '_size.txt', 'w')

	quat = 5
	while quat <= 100:
		img = Image.open('./test/testimg/' + str(id) + '.png')
		img = img.convert('RGB')
		img.save('./test/' + str(id) + '.jpg', quality=quat, subsampling=0)
		del img

		img = cv2.imread('./test/testimg/' + str(id) + '.png')
		imgref = cv2.imread('./test/' + str(id) + '.jpg')
		data = JPEG_encoder(img, quat, True)
		imgmy = JPEG_decoder(data, True)
		psnr_ref = PSNR(img, imgref)
		psnr_my = PSNR(img, imgmy)
		size_ref = os.path.getsize('./test/' + str(id) + '.jpg')
		size_my = data.shape[0]
		print('质量为%3d时，库函数的PSNR为%.2fdB，自己实现的编解码函数的PSNR为%.2fdB' % (quat, psnr_ref, psnr_my))
		print('质量为%3d时，库函数的文件大小为%dB，自己实现的编解码函数的文件大小为%dB' % (quat, size_ref, size_my))
		fpsnr.write(f'{quat:3}    {psnr_ref:5.2f}    {psnr_my:5.2f}\n')
		fsize.write(f'{quat:3}    {size_ref}    {size_my}\n')
		os.remove('./test/' + str(id) + '.jpg')
		if quat < 90:
			quat += 5
		else:
			quat += 1

	fpsnr.close()
	fsize.close()