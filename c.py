import math
import cv2
import numpy as np
import time

startTime = time.time()


image_name = 'r.bmp'


# read image
img = cv2.imread(image_name, 1)
height, wedth, channels = img.shape


# read inputs
print("please enter the block size of arithmetic coding (integer number): ")
blockSize = input()

print("please enter the float type of the numpy array ('float64', 'float32', 'float16'): ")
typeOfFloat = input()

arr_width = []
arr_height = []

for x in range(10):
    if height % (x+1) == 0:
        arr_height.append(x+1)
    if wedth % (x+1) == 0:
        arr_width.append(x+1)

print(arr_height)
print("please enter the height size of the average window from the previous array: ")
size_height = input()

print(arr_width)
print("please enter the width size  of the average window from the previous array: ")

size_width = input()

# average the neighbouring pixels with window size of (size_height*size_width)
arrheightRed = []
arrheightGreen = []
arrheightBlue = []

def channel(num):
    i = 0
    arrheight=[]
    while i + size_height - 1 < height:
        j = 0

        arrwedth = []
        while j + size_width - 1 < wedth:
            x = 0
            for y in range(size_height):
                for t in range(size_width):
                    x += img[i + y][j + t][num]
            j += size_width
            arrwedth.append(x / (size_width * size_height))
        i += size_height
        arrheight.append(arrwedth)
    return arrheight


arrheightRed = channel(0)
arrheightGreen = channel(1)
arrheightBlue = channel(2)


# flatten the image
def flattenArrays(arrheight):
    flattenImage=[]
    for items in arrheight:
        for x in items:
            flattenImage.append(x)
    return flattenImage


flattenImageRed = flattenArrays(arrheightRed)
flattenImageBlue = flattenArrays(arrheightBlue)
flattenImageGreen = flattenArrays(arrheightGreen)

# Encoding Part


# first: calculate probability of pixels appearance of each channel(red, green, blue)
# second: calculate ranges of intervals between(0,1) of each channel(red, green, blue)
def startEnd(flattenImage):
    probability = []

    for x in range(256):
        probability.append(0)
    # calculate probability of the image
    for x in flattenImage:
        probability[x] += 1

    for x in range(256):
        probability[x] = probability[x] / (len(flattenImage) * 1.0)

    # calculate the upper and lower ranges of the image
    startD = 0
    start = {}
    end = {}

    for x in range(256):
        start[x] = startD
        end[x] = start[x] + probability[x]
        startD = end[x]
    return start, end


startRed, EndRed = startEnd(flattenImageRed)
startBlue, EndBlue = startEnd(flattenImageBlue)
startGreen, EndGreen = startEnd(flattenImageGreen)


# encode the image
def encode_arithmetic(levels, start, end, dictionary):
    lower = 0
    upper = 1
    for code in levels:
        range = upper - lower
        low = lower
        lower = low + start[code] * range
        upper = low + end[code] * range
    dictionary.append(lower + (upper-lower)/2.0)


def encode_array(array, start, end):

    dictionary = []
    i = 0
    for x in range(((height*wedth)/(size_width*size_height))/blockSize):
        arr = []
        for y in range(blockSize):
            arr.append(array[i])
            i = i+1
        encode_arithmetic(arr, start, end, dictionary)
    return dictionary


dictionaryRed = encode_array(flattenImageRed, startRed, EndRed)
dictionaryBlue = encode_array(flattenImageBlue, startBlue, EndBlue)
dictionaryGreen = encode_array(flattenImageGreen, startGreen, EndGreen)

# save the encoded arrays into numpy arrays
DictionaryRed = np.array(dictionaryRed, dtype=typeOfFloat)
DictionaryBlue = np.array(dictionaryBlue, dtype=typeOfFloat)
DictionaryGreen = np.array(dictionaryGreen, dtype=typeOfFloat)
np.savez('dictionary.npz', array1=DictionaryRed, array2=DictionaryGreen, array3=DictionaryBlue)

print("encoding time in seconds", time.time()-startTime)

# decoding part
startTime = time.time()

# read numpy arrays
data = np.load('dictionary.npz', allow_pickle=True)
dictionaryRed = data['array1']
dictionaryGreen = data['array2']
dictionaryBlue = data['array3']

# decode each code into sequence of pixels using binary search
def binary_search(start, end, l, h, code, lower, upper, blokSize, i, decodingDictionary):
    if i >= blokSize:
        return
    if h >= l:
        mid = l + (h - l) / 2
        d_lower = lower
        rang = upper - lower
        lower = d_lower + start[mid] * rang
        upper = d_lower + end[mid] * rang
        if lower < code < upper:
            decodingDictionary.append(mid)
            i += 1
            binary_search(start, end, 0, 255, code, lower, upper, blokSize, i, decodingDictionary)
        elif lower > code:
            lower = d_lower
            upper = rang + lower
            return binary_search(start, end, l, mid - 1, code, lower, upper, blokSize, i, decodingDictionary)
        else:
            lower = d_lower
            upper = rang + lower
            return binary_search(start, end, mid + 1, h, code, lower, upper, blokSize, i, decodingDictionary)


def decode_arithmetic(dictionary, start, end):
    decodingDictionary = []
    for i in dictionary:
        binary_search(start, end, 0, 255, i, 0, 1, blockSize, 0, decodingDictionary)
    return decodingDictionary


decodingDictionaryRed = decode_arithmetic(dictionaryRed, startRed, EndRed)
decodingDictionaryBlue = decode_arithmetic(dictionaryBlue, startBlue, EndBlue)
decodingDictionaryGreen = decode_arithmetic(dictionaryGreen, startGreen, EndGreen)


while len(decodingDictionaryRed) < height*wedth/(size_width*size_height):
    decodingDictionaryRed.append(decodingDictionaryRed[0])

while len(decodingDictionaryGreen) < height * wedth / (size_width*size_height):
    decodingDictionaryGreen.append(decodingDictionaryGreen[0])

while len(decodingDictionaryBlue) < height*wedth/(size_width*size_height):
    decodingDictionaryBlue.append(decodingDictionaryBlue[0])


# reshape decoded image then return the average pixels then save the image
def reshape():
    imageRed = np.reshape(decodingDictionaryRed, (height / size_height, wedth / size_width))
    imageGreen = np.reshape(decodingDictionaryGreen, (height / size_height, wedth / size_width))
    imageBlue = np.reshape(decodingDictionaryBlue, (height / size_height, wedth / size_width))

    i = 0
    arrheight = []

    while i < height / size_height:

        arrwedth = []
        j = 0
        while j < wedth / size_width:

            for x in range(size_width):
                arrwedth.append([imageRed[i][j], imageGreen[i][j], imageBlue[i][j]])
            j += 1
        i += 1

        for y in range(size_height):

            arrheight.append(arrwedth)

    return arrheight


arrheight = reshape()
image = np.reshape(arrheight, (height, wedth, 3))
im = np.reshape((image-img), (height*wedth*3))

# calculate MSE and PSNR
MSE = 0
for x in im:
    MSE += int(pow(x,2))
MSE = MSE/(height*wedth*3.0)
if MSE != 0.0:
    PSNR = 10 * math.log10((pow(255, 2)) / MSE)
    print("PSNR:", PSNR, " db")


print("decoding time in seconds", time.time()-startTime)

cv2.imwrite('image.png', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
