import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv

root =os.path.dirname(os.path.abspath("Solution.py"))[:40]+'Datasets\PapilaDB-PAPILA-9c67b80983805f0f886b068af800ef2b507e7dc0'
print(root)
conts = os.listdir(root + "ExpertsSegmentations/Contours/")
dataOG=[]

for i in conts:
  c = np.loadtxt(root + 'ExpertsSegmentations/Contours/'+ i)
  dataOG.append(c)
##just some code to check the nature of the data set  
##lowest = 100
##highest =0
##mean=0
##
##for i in range(len(conts)):
##  if(len(dataOG[i])>highest):
##    highest = len(data[i])
##  if(len(dataOG[i])<lowest):
##    lowest = len(dataOG[i])
##  mean += len(dataOG[i])
##mean = mean/len(conts)
##
##print(lowest)
##print(highest)
##print(mean)

target_size = 16

for i in range(len(dataOG)):
    current_size = dataOG[i].shape[0]

    if current_size > target_size:
        factor = int(current_size / target_size)
        dataOG[i] = resample(dataOG[i], target_size, axis=0)

    elif current_size < target_size:
        factor = int(target_size / current_size)
        dataOG[i] = np.repeat(dataOG[i], factor, axis=0)
        dataOG[i] = resample(dataOG[i], target_size, axis=0)

for i in range(len(conts)):
  np.savetxt((root+'ExpertsSegmentations/Fixed/'+ conts[i]), dataOG[i], delimiter=",")

conts = sorted(os.listdir(root + "ExpertsSegmentations/Fixed/"))

data=[]
for i in conts:
  c = np.loadtxt(root + 'ExpertsSegmentations/Fixed/'+ i ,delimiter=',')
  data.append(c)

concatenated_list = []

for i in range(0, len(data)-1, 4):

    first_array = data[i]
    second_array = data[i+1]
    third_array = data[i+2]
    fourth_array = data[i+3]

    concatenated_array1 = np.concatenate((first_array, third_array), axis=0)
    concatenated_array1 = np.reshape(concatenated_array1, (2, 16, 2))
    concatenated_list.append(concatenated_array1)

    concatenated_array2 = np.concatenate((second_array, fourth_array), axis=0)
    concatenated_array2 = np.reshape(concatenated_array2, (2, 16, 2))
    concatenated_list.append(concatenated_array2)
concatenated_array = np.array(concatenated_list)

##print(len(concatenated_list)) 
##print(concatenated_list[0].shape)


np.savetxt((root + 'ExpertsSegmentations/FixedConcatenated/concatenated_array.csv'), concatenated_array.reshape(-1, 32), delimiter=",")

##resize images but keep the aspectratio the same and modify the contours also
img_names = sorted(os.listdir(root + "FundusImages"))
output = root+'FundusImagesResized'
resized_coords = np.zeros(concatenated_array.shape)
target_size = (256, 256)
done = 0
for i in img_names:
  image = (cv.imread(root + "FundusImages/" + i))

  h, w = image.shape[:2]

  ratio = min(target_size[0] / w, target_size[1] / h)

  new_size = (round(w * ratio), round(h * ratio))

  resized = cv.resize(image, new_size)

  background = np.zeros(target_size[::-1] + (3,), dtype=np.uint8)

  x_offset = (target_size[0] - new_size[0]) // 2
  y_offset = (target_size[1] - new_size[1]) // 2

  background[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0]] = resized
  x_ratio = new_size[0] / w
  y_ratio = new_size[1] / h
  for j in range(concatenated_array.shape[1]):
      for k in range(concatenated_array.shape[2]):
          for l in range(concatenated_array.shape[3]):
              if l == 0:
                  # Adjust X coordinate
                  newX1 = int(concatenated_array[done*2, j, k, l] * x_ratio)
                  newX2 = int(concatenated_array[done*2+1, j, k, l] * x_ratio) 
                  resized_coords[done*2, j, k, l] = newX1
                  resized_coords[done*2+1, j, k, l] = newX2
              else:
                  # Adjust Y coordinate
                  newY1 = int(concatenated_array[done*2, j, k, l] * y_ratio) +32
                  newY2 = int(concatenated_array[done*2+1, j, k, l] * y_ratio)+32 
                  resized_coords[done*2, j, k, l] = newY1
                  resized_coords[done*2+1, j, k, l] = newY2

  cv.imwrite(output+'/'+ i, background)
  done+= 1
  print(str(int(100* (done/488)))+'% done')


resized_coords = np.round(resized_coords).astype(np.int32)
np.savetxt((root + 'ExpertsSegmentations/FixedConcatenatedResized/concatenated_array.csv'), resized_coords.reshape(-1, 32), delimiter=",") 
##resized_coords = np.loadtxt(root + 'ExpertsSegmentations/FixedConcatenatedResized/concatenated_array.csv', delimiter=',')
##resized_coords = resized_coords.reshape(-1, 2, 16, 2)
##print(resized_coords.shape)


output = root+'/ExpertsSegmentations/Masks'
image_size = (256, 256)
done = 0
for i in img_names:
  mask = np.zeros(image_size, dtype=np.uint8)
  cv.fillPoly(mask, [resized_coords[done*2][0]], color=1)
  cup_mask = mask.copy()  # Save a copy of the mask for later use
  mask.fill(0)  # Reset the mask to all zeros
  cv.fillPoly(mask, [resized_coords[done*2][1]], color=1)

  image = np.zeros(image_size + (3,), dtype=np.uint8)  # Create a black image with three color channels
  image[mask == 0] = (1,0,0)  # Set the pixels within the first polygon to red
  image[mask == 1] = (0,1,0)  # Set the pixels within the second polygon to green
  image[cup_mask == 1] = (0,0,1)  # Set the pixels within the first polygon to red
  cv.imwrite((output+'/'+ i[:8]+'_exp1.png'),image)  

  mask.fill(0)  # Reset the mask to all zeros
  cv.fillPoly(mask, [resized_coords[done*2+1][0]], color=1)
  cup_mask = mask.copy()  # Save a copy of the mask for later use
  mask.fill(0)  # Reset the mask to all zeros
  cv.fillPoly(mask, [resized_coords[done*2+1][1]], color=1)

  image = np.zeros(image_size + (3,), dtype=np.uint8)  # Create a black image with three color channels
  image[mask == 0] = (1,0,0)  # Set the pixels within the first polygon to red
  image[mask == 1] = (0,1,0)  # Set the pixels within the second polygon to green
  image[cup_mask == 1] = (0,0,1)  # Set the pixels within the first polygon to red
  cv.imwrite((output+'/'+ i[:8]+'_exp2.png'),image) 

  done+= 1
  print(str(int(100* (done/488)))+'% done')

##mask_names = sorted(os.listdir(root + "ExpertsSegmentations/Masks"))
##masks = np.empty((int(len(mask_names)), 256, 256, 3))
##done = 0
##for i in mask_names:
##  image = cv.imread(output+'/'+ i[:8]+'_exp1.png')
##  image = cv.imread(output+'/'+ i[:8]+'_exp2.png')
##  masks[done] = image
##  done +=1
##  print(str(int(100* (done/976)))+'% done')
