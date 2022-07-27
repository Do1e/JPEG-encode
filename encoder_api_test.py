import cv2
import numpy as np
import argparse

from encoder_api import JPEG_encoder

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='input file', required=True)
parser.add_argument('-g', '--gray', help='encode one channel if set', action='store_true')
args = parser.parse_args()

BGR = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE if args.gray else cv2.IMREAD_COLOR)
data = JPEG_encoder(BGR, show=True, gray=args.gray)
reBuild = cv2.imdecode(data, cv2.IMREAD_COLOR)
cv2.imshow('reBuild', reBuild)
cv2.waitKey(0)
cv2.destroyAllWindows()