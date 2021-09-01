import os
from PIL import Image
import pandas as pd
import numpy as np

###

walk_train = [elm for elm in os.walk('./large_for_training')]
labels = walk_train[0][1]
folders = walk_train[2:]

label_dict = {}
for cnt,elm in enumerate(labels):
    label_dict[elm] = cnt

train_txt = ''
imageid = 0
for folder in folders:
    if folder[1] == []:
        label = folder[0].split('label@')[-1].split('_')[0]
        labelid = label_dict[label]
        uniqueid = folder[0].split('uniqueid@')[-1].split('_')[0]
        if int(uniqueid)<6:
            for image in folder[2]:
                path = folder[0]+'/'+image
                if "__" in path and 'outlayer' not in path:
                    train_txt += '{} {} {} {}\n'.format(imageid, labelid, label, path)
                    imageid += 1

train_txt = train_txt[0:-1]
with open('./train.txt', 'w') as f:
    f.write(train_txt)

###

walk_test = [elm for elm in os.walk('./small_for_inference')]
folders = walk_test[2:]

test_txt = ''
imageid = 0
for folder in folders:
    if folder[1] == []:
        label = folder[0].split('label@')[-1].split('_')[0]
        labelid = label_dict[label]
        uniqueid = folder[0].split('uniqueid@')[-1].split('_')[0]
        imprint_condition = len([elm for elm in folder[2] if '__' in elm])==0
        if int(uniqueid)>=6:
            for image in folder[2]:
                path = folder[0]+'/'+image
                if (("__" in path and 'outlayer' not in path) or imprint_condition) and '.json' not in path:
                    test_txt += '{} {} {} {}\n'.format(imageid, labelid, label, path)
                    imageid += 1
                    
test_txt = test_txt[0:-1]
with open('./test.txt', 'w') as f:
    f.write(test_txt)

###
    
walk_generate = [elm for elm in os.walk('./small_for_inference')]
folders = walk_generate[2:]

generate_txt = ''
imageid = 0
for folder in folders:
    if folder[1] == []:
        label = folder[0].split('label@')[-1].split('_')[0]
        labelid = label_dict[label]
        uniqueid = folder[0].split('uniqueid@')[-1].split('_')[0]
        imprint_condition = len([elm for elm in folder[2] if '__' in elm])==0
        if int(uniqueid)<6:
            for image in folder[2]:
                path = folder[0]+'/'+image
                if (("__" in path and 'outlayer' not in path) or imprint_condition) and '.json' not in path:
                    generate_txt += '{} {} {} {}\n'.format(imageid, labelid, label, path)
                    imageid += 1
                    
generate_txt = generate_txt[0:-1]
with open('./generate.txt', 'w') as f:
    f.write(generate_txt)
    
    