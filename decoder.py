import cv2
import numpy as np
import argparse

import tables
import decoder_utils


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='input file', required=True)
parser.add_argument('-o', '--output', help='output file', required=True)
args = parser.parse_args()

with open(args.input, 'rb') as f:
	data = f.read()
