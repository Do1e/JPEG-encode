# -*- coding: utf-8 -*
import numpy as np

class SymCode:
    def __init__(self, symbol, code, n_bit):
        self.symbol = symbol
        self.code = code
        str_code = ''
        mask = 1 << (n_bit - 1)
        for i in range(0, n_bit):
            if mask & code:
                str_code += '1'
            else:
                str_code += '0'
            mask >>= 1

        self.str_code = str_code

    def __str__(self):
        return "0x{:0>2x}    |  {}".format(self.symbol, self.str_code)

    def __eq__(self, other):
        return self.symbol == other.symbol

    def __le__(self, other):
        return self.symbol < other.symbol

    def __gt__(self, other):
        return self.symbol > other.symbol


def DHT2tbl(data):
    numbers = data[0:16]  # 1~16bit长度的编码对应的个数
    symbols = data[16:len(data)]  # 原数据
    if sum(numbers) != len(symbols):  # 判断是否为正确的范式哈夫曼编码表
        print("Wrong DHT!")
        exit()
    code = 0
    SC = []  # 记录字典的列表
    for n_bit in range(1, 17):
        # 按范式哈夫曼编码规则换算出字典
        for symbol in symbols[sum(numbers[0:n_bit - 1]):sum(numbers[0:n_bit])]:
            SC.append(SymCode(symbol, code, n_bit))
            code += 1
        code <<= 1
    return sorted(SC)

def inttostr(num):
    s = ""
    while (num != 0):
        s += '0' if (num % 2 == 0) else '1'
        num = int(num / 2)
    s = s[::-1]
    while len(s)<8:
        s='0'+s
    return s

def tobin(num):
    s = ""
    if (num > 0):
        while (num != 0):
            s += '0' if (num % 2 == 0) else '1'
            num = int(num / 2)
        s = s[::-1]
    elif (num < 0):
        num = -num
        while (num != 0):
            s += '1' if (num % 2 == 0) else '0'
            num = int(num / 2)
        s = s[::-1]
    return s

def get_num(s, n, tbl):
    str_tbl=[]
    for i in tbl:
        str_tbl.append(i.str_code)
    bit=0
    for k in range(1,17):
        s1=s[0:k]
        length=len(s1)
        if s1 in str_tbl:
            bit=str_tbl.index(s1)
            break
    s=s[length:]
    if n==-1:
        num=tonum(s[0:bit])
        s=s[bit:]
        # print(bit,num)
        return bit,num, s
    else:
        if(bit%10)==0 & bit!=0 & bit<=150:
            n_zero=(bit-10)//10
            bit=10
        elif bit>150:
            n_zero=15
            bit=bit-n_zero*10-1
        else:
            n_zero=bit//10
            bit=bit%10
        num=tonum(s[0:bit])
        s=s[bit:]
        # print(n_zero,num)
        return n_zero,num, s

def tonum(s):
    bit=len(s)
    num=0
    if s=='':
        return 0
    if s[0]=='1':
        s = s[::-1]
        for i in range(bit):
            num+=int(s[i])*(2**i)
    else:
        s=s[::-1]
        for i in range(bit):
            num-=(1-int(s[i]))*(2**i)
    return num


def inverse_z(dc,list):
    block_xz=np.zeros((8,8))
    block_xz[0][0]=dc
    pos = np.array([0, 1])
    R = np.array([0, 1])
    LD = np.array([1, -1])
    D = np.array([1, 0])
    RU = np.array([-1, 1])
    for i in range(0, 63):
        block_xz[pos[0], pos[1]]=list[i]
        if (((pos[0] == 0) or (pos[0] == 7)) and (pos[1] % 2 == 0)):
            pos = pos + R
        elif (((pos[1] == 0) or (pos[1] == 7)) and (pos[0] % 2 == 1)):
            pos = pos + D
        elif ((pos[0] + pos[1]) % 2 == 0):
            pos = pos + RU
        else:
            pos = pos + LD
    return block_xz

def x00(data):
    s=""
    n=0
    sum=len(data)
    while 1:
        #print(n/sum)
        num, numnext = data[n:n+8],data[n+8:n+16]
        if num == '11111111':
            if int(numnext) != 0:  # 到结束ffd9
                break
            s+=(num)
            n += 16  # 跳过00
        else:
            s+=num
            n += 8
    return s

std_luminance_quant_tbl = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ],
    np.uint8
)
# 色度量化表
std_chrominance_quant_tbl = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ],
    np.uint8
)