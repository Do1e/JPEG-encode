import numpy as np

# 记录哈夫曼字典的类
# symbol: 原始数据
# code: 对应的编码数据
# n_bit: 编码的二进制位数
class Sym_Code():
	def __init__(self, symbol: int, code: int, n_bit: int) -> None:
		self.symbol = symbol
		self.code = code
		self.str_code = bin(code)[2:].zfill(n_bit)
	"""定义输出形式"""
	def __str__(self):
		return "0x{:0>2x}    |  {}".format(self.symbol, self.str_code)
	"""定义排序依据"""
	def __eq__(self, other):
		return self.symbol == other.symbol
	def __le__(self, other):
		return self.symbol < other.symbol
	def __gt__(self, other):
		return self.symbol > other.symbol

# 将范式哈夫曼编码表转换为哈夫曼字典
# data: 定义的范式哈夫曼编码表
def DHT2tbl(data: np.ndarray) -> list:
	numbers = data[0:16]				# 1~16bit长度的编码对应的个数
	symbols = data[16:len(data)]		# 原数据
	if(sum(numbers) != len(symbols)):	# 判断是否为正确的范式哈夫曼编码表
		print("Wrong DHT!")
		exit()
	code = 0
	SC = []								# 记录字典的列表
	for n_bit in range(1, 17):
		# 按范式哈夫曼编码规则换算出字典
		for symbol in symbols[sum(numbers[0:n_bit-1]):sum(numbers[0:n_bit])]:
			SC.append(Sym_Code(symbol, code, n_bit))
			code += 1
		code <<= 1
	return sorted(SC)