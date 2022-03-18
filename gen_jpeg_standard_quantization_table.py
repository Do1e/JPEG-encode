'''
created by riantr on 20210401
Usage: python gen_quantization_table.py [Quality (0~100)]
Example: python gen_quantization_table.py 96
'''
import sys
import cv2
import numpy as np

def gen_quant_table_by_quality(quality=100):

    def gen_zigzag_array_size(matrix_size=8):
        top = list(range(1,matrix_size+1))
        bottom = list(range(matrix_size-1,0,-1))
        zigzag_array_size_list = top + bottom
    
        return zigzag_array_size_list

    def zigzag_slicer(size_list,input_array=None):
        '''
        [lst[i:i+3] for i in range(0,len(lst),3)]
        '''
        type_flag = input_array.__class__
        result = []
        i = 0
        for line_size in size_list:
            result.append(input_array[i:i+line_size])
            i += line_size
            
        if type_flag == np.ndarray:
            return np.array(result)
        elif type_flag == np.matrix:
            return np.matrix(result)
        else:
            return result

    def ZigZag(img):
        def get_lower(img):
            height,width = img.shape[:2]
            gradient = 1.0*height/width
            positions = []
            values = []
            means = []
            for j in range(0,height,1):
                line_points = []
                line_values = []
                for i in range(0,height,1):
                    y = height-1 - i -j
                    x = int(i/gradient)
                    if x*y >= 0:
                        line_points.append([y,x])
                        line_values.append(img[y][x])
                positions.append(line_points)
                values.append(line_values)
                means.append(sum(line_values)/len(line_values))
            return positions,values,means

        def get_higher(img):
            height,width = img.shape[:2]
            gradient = 1.0*height/width
            positions = []
            values = []
            means = []
            for j in range(0,height,1): # 0:479
                line_points = []
                line_values = []
                for i in range(0,width,1): # 0:639
                    x = i        #0:639
                    y = int(i*gradient) + j
    
                    if y < height and x*y >= 0:
                        line_points.append([y,width-1-x])
                        line_values.append(img[y][width-1-x])
                line_points.reverse()
                line_values.reverse()
    
                positions.append(line_points)
                values.append(line_values)
                means.append(sum(line_values)/len(line_values))
    
            positions.reverse()
            values.reverse()
            means.reverse()
    
            return positions,values, means

        points_lower_frequence,values_lower_frequence,means_lower_frequence = get_lower(img)
        points_higher_frequence,values_higher_frequence,means_higher_frequence = get_higher(img)
        points_higher_frequence.pop()
        values_higher_frequence.pop()
        means_higher_frequence.pop()
        positions = points_higher_frequence + points_lower_frequence
        values = values_higher_frequence + values_lower_frequence
        means = means_higher_frequence + means_lower_frequence
    
        positions.reverse()
        values.reverse()
        means.reverse()
        idx = 1
        for P in positions:
            if not (idx % 2) :
                P.reverse()
            idx += 1
        idx = 1
        for P in values:
            if not (idx % 2) :
                P.reverse()
            idx += 1
    
        return positions, values, means
    
    def deZigZag(original_img,zigzaged_positions,zigzaged_values):
        new_img = np.zeros(original_img.shape,dtype=original_img.dtype)
        for i in range(len(zigzaged_values)):
            for j in range(len(zigzaged_values[i])):
                new_img[zigzaged_positions[i][j][0],zigzaged_positions[i][j][1]] = zigzaged_values[i][j]
        return new_img
                    
    def gen_zigzaged_quant_table(quality):
        img = np.arange(64*3).reshape(8,8,3)
        img[:,:,1] = 127
        filetype = '.jpg'
        ret,buff = cv2.imencode(filetype,img,[int(cv2.IMWRITE_JPEG_QUALITY), quality])
    
        img_bytes = np.array(buff).tobytes()
        
        SOF0_flag = b'\xff\xc0'
        if SOF0_flag in img_bytes:
            head = img_bytes.split(SOF0_flag)[0]
        else:
            print("Cannot find SOF0_flag, not a jpeg file")
            sys.exit()
        DQT_flag = b'\xff\xdb'
        if DQT_flag in head:
            index_DQT = head.index(DQT_flag)
            other_info = head[0:index_DQT]
            quant_table_info = head[index_DQT:]
            length = quant_table_info[3] - 3
            start = 5
            QT_info_1 = quant_table_info[start:start+length]
            QT_info_2 = quant_table_info[start+length+5:]
            quant_table_1 = []
            quant_table_2 = []  
            for i in range(length):
                quant_table_1.append(QT_info_1[i])
                try:
                    quant_table_2.append(QT_info_2[i])
                except:
                    pass
            return quant_table_1, quant_table_2
        else:
            print("Cannot find DQT_flag, not a jpeg file")
            return
    
    luminance_quant_table,chrominance_quant_table = gen_zigzaged_quant_table(quality)
    zigzag_array_size_list = gen_zigzag_array_size()
    zigzaged_luminance_quant_table = zigzag_slicer(zigzag_array_size_list,luminance_quant_table)
    zigzaged_chrominance_quant_table = zigzag_slicer(zigzag_array_size_list,chrominance_quant_table)

    temp = np.arange(64).reshape(8,8)
    positions,_values,_means = ZigZag(temp)

    luminance_quant_table_matrix = deZigZag(temp,positions,zigzaged_luminance_quant_table)
    if len(chrominance_quant_table) != 0:
        chrominance_quant_table_matrix = deZigZag(temp,positions,zigzaged_chrominance_quant_table)
    
        return luminance_quant_table_matrix,chrominance_quant_table_matrix
    else:
        return luminance_quant_table_matrix,None

if __name__ == '__main__':
    try:
        quality = int(sys.argv[1])
    except:
        print("Usage: python gen_jpeg_standard_quantization_table.py [Quality (0~100)]")
        print("Example: python gen_jpeg_standard_quantization_table.py 100\n")
        quality = 100
    luminance_quant_table_matrix ,chrominance_quant_table_matrix = gen_quant_table_by_quality(quality)
    print(luminance_quant_table_matrix)
    print(chrominance_quant_table_matrix)
'''
## old version
import sys
import numpy as np

def gen_quant_matrix(Quality,T=0):
  std_luminance_quant_tbl = [
    [16,  11,  10,  16,  24,  40,  51,  61],
    [12,  12,  14,  19,  26,  58,  60,  55],
    [14,  13,  16,  24,  40,  57,  69,  56],
    [14,  17,  22,  29,  51,  87,  80,  62],
    [18,  22,  37,  56,  68, 109, 103,  77],
    [24,  35,  55,  64,  81, 104, 113,  92],
    [49,  64,  78,  87, 103, 121, 120, 101],
    [72,  92,  95,  98, 112, 100, 103,  99]
]
  std_chrominance_quant_tbl = [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ]
  if T == 0:
    StandardMatrix = std_luminance_quant_tbl
  else:
    StandardMatrix = std_chrominance_quant_tbl
  StandardMatrix = np.array(StandardMatrix)
  ScaleFactor = 200 - Quality * 2
  NewMatrix = (StandardMatrix * ScaleFactor + 50)/100
  NewMatrix = np.round(NewMatrix+1e-10).astype(np.uint8)
  return NewMatrix
if __name__ == '__main__':
    try:
        Quality = int(sys.argv[1])
    except:
        Quality = 96
        print()
        print("Usage: python gen_quantization_table.py [Quality (0~100)] [Type (0/1 :0 for luninance 1 for chrominance)]\n")
        print("Example: python gen_quantization_table.py 96 0\n")
        print("when Quality = 96 and Type = 0, the quantization table would be:\n")
    try:
        Type = int(sys.argv[2])
    except:
        Type = 0

    print( gen_quant_matrix(Quality,Type))

'''
