import re
import ast
import numpy as np
import json
import matplotlib.pyplot as plt

path = "./Unet3_result_recode.txt"

with open(path, 'r') as file:
    raw_data = file.read()

raw_data = raw_data.replace('\n', '')

raw_data = raw_data.replace('}{', '};{')


raw_data = raw_data.replace('\'', '\"')
raw_data = raw_data.replace('array(', '')
raw_data = raw_data.replace(', dtype=float32)', '')
raw_data = raw_data.replace(')', '')
raw_data = raw_data.replace('.,', ',')
raw_data = raw_data.replace('.]', ']')
raw_data = raw_data.replace('nan', '0.0')
raw_data = raw_data.replace(', 0. ', ', 0.0')

entries = raw_data.strip().split(';')

result = []
for entry in entries:
    data_dict = json.loads(entry)
    result.append(data_dict)

bas_loss = [item['loss']   for item in result]    
bas_epochs = list(range(1, len(bas_loss) + 1))

IoU = [item['IOU']   for item in result] 
bas_IoU1 = [item[0] for item in IoU] 
bas_IoU2 = [item[1] for item in IoU] 
bas_IoU3 = [item[2] for item in IoU] 

Dice = [item['Dice']   for item in result] 
bas_Dice1 = [item[0] for item in IoU] 
bas_Dice2 = [item[1] for item in IoU] 
bas_Dice3 = [item[2] for item in IoU] 



path = "./residual-Unet3_result_recode.txt"

with open(path, 'r') as file:
    raw_data = file.read()

raw_data = raw_data.replace('\n', '')

raw_data = raw_data.replace('}{', '};{')


raw_data = raw_data.replace('\'', '\"')
raw_data = raw_data.replace('array(', '')
raw_data = raw_data.replace(', dtype=float32)', '')
raw_data = raw_data.replace(')', '')
raw_data = raw_data.replace('.,', ',')
raw_data = raw_data.replace('.]', ']')
raw_data = raw_data.replace('nan', '0.0')
raw_data = raw_data.replace(', 0. ', ', 0.0')

entries = raw_data.strip().split(';')

result = []
for entry in entries:
    data_dict = json.loads(entry)
    result.append(data_dict)

res_loss = [item['loss']   for item in result]    
res_epochs = list(range(1, len(res_loss) + 1))

IoU = [item['IOU']   for item in result] 
res_IoU1 = [item[0] for item in IoU] 
res_IoU2 = [item[1] for item in IoU] 
res_IoU3 = [item[2] for item in IoU] 

Dice = [item['DCS']   for item in result] 
res_Dice1 = [item[0] for item in IoU] 
res_Dice2 = [item[1] for item in IoU] 
res_Dice3 = [item[2] for item in IoU] 





path = "./attention-Unet3_result_recode.txt"

with open(path, 'r') as file:
    raw_data = file.read()

raw_data = raw_data.replace('\n', '')

raw_data = raw_data.replace('}{', '};{')


raw_data = raw_data.replace('\'', '\"')
raw_data = raw_data.replace('array(', '')
raw_data = raw_data.replace(', dtype=float32)', '')
raw_data = raw_data.replace(')', '')
raw_data = raw_data.replace('.,', ',')
raw_data = raw_data.replace('.]', ']')
raw_data = raw_data.replace('nan', '0.0')
raw_data = raw_data.replace(', 0. ', ', 0.0')

entries = raw_data.strip().split(';')

result = []
for entry in entries:
    data_dict = json.loads(entry)
    result.append(data_dict)

att_loss = [item['loss']   for item in result]    
att_epochs = list(range(1, len(att_loss) + 1))

IoU = [item['IOU']   for item in result] 
att_IoU1 = [item[0] for item in IoU] 
att_IoU2 = [item[1] for item in IoU] 
att_IoU3 = [item[2] for item in IoU] 

Dice = [item['DCS']   for item in result] 
att_Dice1 = [item[0] for item in IoU] 
att_Dice2 = [item[1] for item in IoU] 
att_Dice3 = [item[2] for item in IoU] 

path = "./residual-attention-Unet3_result_recode.txt"

with open(path, 'r') as file:
    raw_data = file.read()

raw_data = raw_data.replace('\n', '')

raw_data = raw_data.replace('}{', '};{')


raw_data = raw_data.replace('\'', '\"')
raw_data = raw_data.replace('array(', '')
raw_data = raw_data.replace(', dtype=float32)', '')
raw_data = raw_data.replace(')', '')
raw_data = raw_data.replace('.,', ',')
raw_data = raw_data.replace('.]', ']')
raw_data = raw_data.replace('nan', '0.0')
raw_data = raw_data.replace(', 0. ', ', 0.0')

entries = raw_data.strip().split(';')

result = []
for entry in entries:
    data_dict = json.loads(entry)
    result.append(data_dict)

res_att_loss = [item['loss']   for item in result]    
res_att_epochs = list(range(1, len(res_att_loss) + 1))

IoU = [item['IOU']   for item in result] 
res_att_IoU1 = [item[0] for item in IoU] 
res_att_IoU2 = [item[1] for item in IoU] 
res_att_IoU3 = [item[2] for item in IoU] 

Dice = [item['DCS']   for item in result] 
res_att_Dice1 = [item[0] for item in IoU] 
res_att_Dice2 = [item[1] for item in IoU] 
res_att_Dice3 = [item[2] for item in IoU] 

plt.figure()
plt.plot(bas_epochs, bas_loss, color='blue', lw=1, label = 'Basic Unet3+')
plt.plot(res_epochs, res_loss, color='green', lw=1, label = 'Residual Unet3+')
plt.plot(att_epochs, att_loss, color='red', lw=1, label = 'Attention Unet3+')
plt.plot(res_att_epochs, res_att_loss, color='yellow', lw=1, label = 'Residual attention Unet3+')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Four different Unet3+ models loss cuve')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(20, 10))

plt.subplot(2, 4, 1)
plt.plot(bas_epochs,bas_Dice1 , color='blue', lw=1, label = "WT")
plt.plot(bas_epochs,bas_Dice2 , color='green', lw=1, label = "TC")
plt.plot(bas_epochs,bas_Dice3 , color='red', lw=1, label = "ED")
plt.xlabel('epochs')
plt.ylabel('Dice')
plt.title('Basic Unet3+ Dice cuve')
plt.legend(loc='lower right')

plt.subplot(2, 4, 2)
plt.plot(res_epochs,res_Dice1 , color='blue', lw=1, label = "WT")
plt.plot(res_epochs,res_Dice2 , color='green', lw=1, label = "TC")
plt.plot(res_epochs,res_Dice3 , color='red', lw=1, label = "ED")
plt.xlabel('epochs')
plt.ylabel('Dice')
plt.title('Residual Unet3+ Dice cuve')
plt.legend(loc='lower right')

plt.subplot(2, 4, 3)
plt.plot(att_epochs,att_Dice1 , color='blue', lw=1, label = "WT")
plt.plot(att_epochs,att_Dice2 , color='green', lw=1, label = "TC")
plt.plot(att_epochs,att_Dice3 , color='red', lw=1, label = "ED")
plt.xlabel('epochs')
plt.ylabel('Dice')
plt.title('Attention Unet3+ Dice cuve')
plt.legend(loc='lower right')

plt.subplot(2, 4, 4)
plt.plot(res_att_epochs,res_att_Dice1 , color='blue', lw=1, label = "WT")
plt.plot(res_att_epochs,res_att_Dice2 , color='green', lw=1, label = "TC")
plt.plot(res_att_epochs,res_att_Dice3 , color='red', lw=1, label = "ED")
plt.xlabel('epochs')
plt.ylabel('Dice')
plt.title('Residual attention Unet3+ Dice cuve')
plt.legend(loc='lower right')

plt.subplot(2, 4, 5)

plt.plot(bas_epochs,bas_IoU1 , color='blue', lw=1, label = "WT")
plt.plot(bas_epochs,bas_IoU2 , color='green', lw=1, label = "TC")
plt.plot(bas_epochs,bas_IoU3 , color='red', lw=1, label = "ED")
plt.xlabel('epochs')
plt.ylabel('IOU')
plt.title('Basic Unet3+ IOU cuve')
plt.legend(loc='lower right')


plt.subplot(2, 4, 6)
plt.plot(res_epochs,res_IoU1 , color='blue', lw=1, label = "WT")
plt.plot(res_epochs,res_IoU2 , color='green', lw=1, label = "TC")
plt.plot(res_epochs,res_IoU3 , color='red', lw=1, label = "ED")
plt.xlabel('epochs')
plt.ylabel('IOU')
plt.title('Residual Unet3+ IOU cuve')
plt.legend(loc='lower right')


plt.subplot(2, 4, 7)
plt.plot(att_epochs,att_IoU1 , color='blue', lw=1, label = "WT")
plt.plot(att_epochs,att_IoU2 , color='green', lw=1, label = "TC")
plt.plot(att_epochs,att_IoU3 , color='red', lw=1, label = "ED")
plt.xlabel('epochs')
plt.ylabel('IOU')
plt.title('Attention Unet3+ IOU cuve')
plt.legend(loc='lower right')


plt.subplot(2, 4, 8)
plt.plot(res_att_epochs,res_att_IoU1 , color='blue', lw=1, label = "WT")
plt.plot(res_att_epochs,res_att_IoU2 , color='green', lw=1, label = "TC")
plt.plot(res_att_epochs,res_att_IoU3 , color='red', lw=1, label = "ED")
plt.xlabel('epochs')
plt.ylabel('IOU')
plt.title('Residual attention Unet3+ IOU cuve')
plt.legend(loc='lower right')
plt.show()
