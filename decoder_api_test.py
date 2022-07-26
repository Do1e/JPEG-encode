import cv2
import numpy as np
import argparse

from decoder_api import JPEG_decoder

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='input file', required=True)
args = parser.parse_args()

with open(args.input, 'rb') as f:
	data = f.read()
data = np.frombuffer(data, np.uint8)
data = JPEG_decoder(data, show=True)
cv2.imshow('reBuild', data)
cv2.waitKey(0)
cv2.destroyAllWindows()