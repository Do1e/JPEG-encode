# -*- coding: utf-8 -*-
import argparse
from jpegdef_decoder import *
import numpy as np
from struct import unpack
import cv2

import matplotlib.pyplot as plt

class JPEG:
    def __init__(self, file):
        with open(file, "rb") as f:
            self.data = f.read()
        self.quant_tbl={}
        self.quant_id = []
        self.huffman = {}
        self.huffmantbl={}

    def decode(self):
        data = self.data
        while 1:
            (marker,) = unpack(">H", data[0:2])
            # print('%#x'%marker)
            if marker == 0xffd8:
                data = data[2:]  # 起始
            elif marker == 0xffd9:
                return
            else:
                (length,) = unpack(">H", data[2:4])
                length = length + 2  # marker长度2
                if marker == 0xffdb:  # 量化表
                    self.decode_quant_tbl(data[4:length])
                elif marker == 0xffc0:  # frame
                    self.get_frame(data[4:length])
                elif marker == 0xffc4:  # 霍夫曼表
                    self.decode_huffman(data[4:length])
                elif marker == 0xffda:  # 图像数据
                    length = self.scan(data, length)  # 所有图像数据的长度
                data = data[length:]
            if len(data) == 0:
                break

    def decode_quant_tbl(self, data):
        (id,) = unpack("B", data[0:1])
        quant_list = list(unpack("B" * 64, data[1:65]))
        self.quant_tbl[id]=inverse_z(quant_list[0],quant_list[1:])

    def get_frame(self, data):
        # print(unpack("B", data[0:1]))
        (self.h,) = unpack(">H", data[1:3])
        # print(self.h)
        (self.w,) = unpack(">H", data[3:5])
        # print(self.w)
        (self.channel,) = unpack("B", data[5:6])

        # self.h, self.w, channel = unpack("HHB", data[1:6])
        for i in range(3):
            (id,) = unpack("B", data[(8 + 3 * i):(9 + 3 * i)])
            # print(id)
            self.quant_id.append(id)

    def decode_huffman(self, data):
        (id,) = unpack("B", data[0:1])
        # id: 00 DC0; 01 DC1; 10 AC0; 11AC1
        self.huffman[id] = list(unpack("B" * 16, data[1:17]))
        elements = []
        n = 17
        for i in self.huffman[id]:
            elements += list(unpack("B" * i, data[n:n + i]))
            n += i
        self.huffman[id] += elements
        self.huffmantbl[id] = DHT2tbl(self.huffman[id])
        # for i in self.huffmantbl[id]:
        #     print(i)


    def scan(self, data, length):
        # 把FF后面的00删掉
        n = length
        s=""
        while 1:
            num, numnext = unpack("BB", data[n:n + 2])
            if num == 0xff:
                if numnext != 0:  # 到结束ffd9
                    break
                s+=inttostr(num)
                n += 2  # 跳过00
            else:
                s+=inttostr(num)
                n += 1
        # f=open("test.txt",'a')
        # f.write('\n')
        # f.write(s)
        # n 图片信息的总长度
        dc_y=[0 for index in range(self.h*self.w//64)]
        dc_cr=[0 for index in range(self.h*self.w//64)]
        dc_cb=[0 for index in range(self.h*self.w//64)]
        self.block_y=np.zeros((self.h,self.w), dtype=np.uint16)
        self.block_cr=np.zeros((self.h,self.w), dtype=np.uint16)
        self.block_cb=np.zeros((self.h,self.w), dtype=np.uint16)
        k=0
        for i in range(0, self.h, 8):
            for j in range(0, self.w, 8):
                dc_y[k], self.block_y[i:i+8,j:j+8], s = self.decode_imagedata(s,0 if k==0 else dc_y[k-1], 0)
                dc_cr[k],self.block_cr[i:i+8,j:j+8], s = self.decode_imagedata(s,0 if k==0 else dc_cr[k-1], 1)
                dc_cb[k], self.block_cb[i:i + 8, j:j + 8], s = self.decode_imagedata(s,0 if  k==0 else dc_cb[k-1], 1)
                k+=1
        return n

    def decode_imagedata(self,s,dc0,id):
        # f=open("dcac.txt","a")
        # f.write('\n')
        dc1=dc0
        ac=[]
        if id==0:
            huffmantbl_dc=self.huffmantbl[0]
            huffmantbl_ac=self.huffmantbl[16]
        else:
            huffmantbl_dc = self.huffmantbl[1]
            huffmantbl_ac = self.huffmantbl[17]
        n,dc_now, s1=get_num(s[0:32],-1, huffmantbl_dc)
        s=s[32:]
        dc1+=dc_now
        # f.write(str(dc1)+"  ")
        t=0
        while(1):
            if (s1[0:4]=='1010') & (id==0):
                s1=s1[4:]
                break
            if (s1[0:2]=='00') & (id==1):
                s1=s1[2:]
                break
            t+=1
            if len(s1)<32:
                if(len(s)<32):
                    s1=s1+s[:]
                    s=''
                else:
                    s1=s1+s[0:32]
                    s=s[32:]
            
            n,num,s1=get_num(s1,1, huffmantbl_ac)
            # f.write(str(n)+"  ")
            # f.write(str(num)+"  ")
            if(n!=0):
                while(n!=0):
                    ac.append(0)
                    n-=1
            
            ac.append(num)
            if len(ac)>=63:
                break
        s=s1+s
        #if (len(ac)<63):
            # f.write("0  0  ")
        while(len(ac)<63):
            ac.append(0)
        block_xz=np.zeros((8,8))

        block_xz=inverse_z(dc1,ac)
        block_xquan=block_xz*self.quant_tbl[id]
        # for i in range(8):
        #     for k in range(8):
        #         f.write(str(int(block_xz[i][k]))+' ')
        #*******************************************************************************************************

        block_xdct=cv2.idct(block_xquan).round().astype(np.int16) + 128
        for i in range(8):
            for k in range(8):
                if block_xdct[i][k]>=256:
                    block_xdct[i][k]=255
                elif block_xdct[i][k]<0:
                    block_xdct[i][k]=0

        return dc1,block_xdct,s



parser = argparse.ArgumentParser()
parser.add_argument("input", help="path to the input file")
args = parser.parse_args()

input_jpg = args.input
img = JPEG(input_jpg)
img.decode()
bmp_decode=np.zeros((img.h,img.w,3), dtype=np.uint8)
bmp_decode[:,:,0]=img.block_y
bmp_decode[:,:,1]=img.block_cb
bmp_decode[:,:,2]=img.block_cr
BGR = cv2.cvtColor(bmp_decode, cv2.COLOR_YCrCb2BGR)

cv2.imshow('pic', BGR)
cv2.waitKey(0)