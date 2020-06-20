import math
import cv2
import numpy as np

# read inputs

blockSize = 3
size=2
typeOfFloat = 'float64'

# inputs validation
image_name = 'r.bmp'


# read image as gray
img = cv2.imread(image_name, 1)
height,wedth,channels=img.shape


arrheightRed=[]
arrheightGreen=[]
arrheightBlue=[]

def channel(num):
    i = 0
    arrheight=[]
    while i + size - 1 < height:
        j = 0

        arrwedth = []
        while j + size - 1 < wedth:
            x = 0
            for y in range(size):
                if size == 1:
                    x += img[i + y][j][num]
                else:
                    x += img[i + y][j][num]
                    x += img[i][j + y][num]

            j += size
            arrwedth.append(x / (size * size))
        i += size
        arrheight.append(arrwedth)
    return arrheight
arrheightRed=channel(0)
arrheightGreen=channel(1)
arrheightBlue=channel(2)


# flatten the image

def flattenArrays(arrheight):
    flattenImage=[]
    for items in arrheight:
        for x in items:
            flattenImage.append(x)
    return flattenImage



flattenImageRed=flattenArrays(arrheightRed)
flattenImageBlue=flattenArrays(arrheightBlue)
flattenImageGreen=flattenArrays(arrheightGreen)






# Encoding Part



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
    return start,end

startRed,EndRed=startEnd(flattenImageRed)
startBlue,EndBlue=startEnd(flattenImageBlue)
startGreen,EndGreen=startEnd(flattenImageGreen)

# encode the image


def encode_arithmetic(levels,start,end,dictionary):
    lower = 0
    upper = 1
    for code in levels:
        range = upper - lower
        low = lower
        lower = low + start[code] * range
        upper = low + end[code] * range
    dictionary.append(lower + (upper-lower)/2.0)



def encode_array(array,start,end):
    dictionary = []
    i = 0
    for x in range(((height*wedth)/(size*size))/blockSize):
        arr = []
        for y in range(blockSize):
            arr.append(array[i])
            i = i+1
        encode_arithmetic(arr,start,end,dictionary)
    return dictionary


dictionaryRed=encode_array(flattenImageRed,startRed,EndRed)
dictionaryBlue=encode_array(flattenImageBlue,startBlue,EndBlue)
dictionaryGreen=encode_array(flattenImageGreen,startGreen,EndGreen)

DictionaryRed = np.array(dictionaryRed, dtype=typeOfFloat)
np.save('dictionaryRed.npy', DictionaryRed)
DictionaryBlue = np.array(dictionaryBlue, dtype=typeOfFloat)
np.save('dictionaryBlue.npy', DictionaryBlue)
DictionaryGreen = np.array(dictionaryGreen, dtype=typeOfFloat)
np.save('dictionaryGreen.npy', DictionaryGreen)

dictionaryRed = np.load('dictionaryRed.npy', allow_pickle=True)
dictionaryBlue = np.load('dictionaryBlue.npy', allow_pickle=True)
dictionaryGreen = np.load('dictionaryGreen.npy', allow_pickle=True)

def binary_search(start,end, l, h, code,lower,upper,blokSize,i,decodingDictionary):
  if i >= blokSize:
      return
  if h >= l:
    mid = l + (h - l)/2
    d_lower = lower
    rang = upper - lower
    lower = d_lower + start[mid] * rang
    upper = d_lower + end[mid] * rang
    if lower<code<upper:
        decodingDictionary.append(mid)
        i+=1
        binary_search(start,end,0,255,code,lower,upper,blokSize,i,decodingDictionary)
    elif lower > code:
      lower = d_lower
      upper = rang + lower
      return binary_search(start,end, l, mid-1, code,lower,upper,blokSize,i,decodingDictionary)
    else:
      lower = d_lower
      upper = rang + lower
      return binary_search(start,end, mid+1, h, code,lower,upper,blokSize,i,decodingDictionary)


def decode_arithmetic(dictionary,start,end):
    decodingDictionary = []
    for i in dictionary:
        binary_search(start, end, 0, 255, i, 0, 1, blockSize, 0,decodingDictionary)
    return decodingDictionary



decodingDictionaryRed=decode_arithmetic(dictionaryRed,startRed,EndRed)
decodingDictionaryBlue=decode_arithmetic(dictionaryBlue,startBlue,EndBlue)
decodingDictionaryGreen=decode_arithmetic(dictionaryGreen,startGreen,EndGreen)



while len(decodingDictionaryRed) < height*wedth/(size*size):
    decodingDictionaryRed.append(decodingDictionaryRed[0])
while len(decodingDictionaryGreen) < height * wedth / (size * size):
        decodingDictionaryGreen.append(decodingDictionaryGreen[0])
while len(decodingDictionaryBlue) < height*wedth/(size*size):
    decodingDictionaryBlue.append(decodingDictionaryBlue[0])




decodingDictionary=np.array(decodingDictionaryRed)
# # reshape decoded image and save it
def reshape():
    imageRed = np.reshape(decodingDictionaryRed, (height / size, wedth / size))
    imageGreen = np.reshape(decodingDictionaryGreen, (height / size, wedth / size))
    imageBlue = np.reshape(decodingDictionaryBlue, (height / size, wedth / size))

    i = 0

    arrheight = []
    while i < height / size:
        arrwedth = []
        j = 0
        while j < wedth / size:
            for x in range(size):
                arrwedth.append([imageRed[i][j],imageGreen[i][j],imageBlue[i][j]])
            j += 1
        i += 1
        for y in range(size):
            arrheight.append(arrwedth)

    return arrheight


arrheight=reshape()
image = np.reshape(arrheight, (height,wedth,3))

cv2.imwrite('image.png', image)
cv2.waitKey(0)
cv2.destroyAllWindows()