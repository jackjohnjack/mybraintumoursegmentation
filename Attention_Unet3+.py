import torch
import tqdm
import os
from torch import nn
from torch.nn import BatchNorm3d,Conv3d,ConvTranspose3d,MaxPool3d
from torch.nn import ReLU,Sequential,Upsample,Sigmoid
from torch.nn import Module,GroupNorm,BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import BinaryConfusionMatrix
from skimage.transform import resize
import numpy as np
import nibabel as nib
device = "cuda:0"
print("device is:",device)



class AttentionBlock(Module):
    def __init__(self, decoder_input_channels, encode_input_channels, decoder_output_channels):
        super(AttentionBlock, self).__init__()
        self.decoder_g = Sequential(
            Conv3d(decoder_input_channels, decoder_output_channels, kernel_size=1, stride=1, padding=0, bias=True),
            GroupNorm(num_groups=8, num_channels=decoder_output_channels)
        )
        
        self.encoder_x = Sequential(
            Conv3d(encode_input_channels, decoder_output_channels, kernel_size=1, stride=1, padding=0, bias=True),
             GroupNorm(num_groups=8, num_channels=decoder_output_channels)
        )
        
        self.psi = Sequential(
            Conv3d(decoder_output_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            BatchNorm3d(1),
            Sigmoid()
        )
        
        self.relu = ReLU(inplace=True)
        
        self.upsampling = Upsample(scale_factor=2, mode='trilinear')

    def forward(self, decoder_input, encode_input):
        dec_g = self.decoder_g(decoder_input)
        enc_x = self.encoder_x(encode_input)
        psi = self.relu(dec_g + enc_x)
        psi = self.psi(psi)
        #psi = self.upsampling(psi)
        return encode_input * psi
        
        
class FirstBlock(Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        self.conv3dskip = Sequential(
            Conv3d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            GroupNorm(num_groups=8, num_channels=output_channels)
        )
        self.conv3d1 = Conv3d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3d2 = Conv3d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm3d1 = GroupNorm(num_groups=8, num_channels=output_channels)
        self.batchnorm3d2 = GroupNorm(num_groups=8, num_channels=output_channels)
        self.relu = ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv3d1(x)
        x = self.batchnorm3d1(x)
        x = self.relu(x)        
        
        x = self.conv3d2(x)
        return x        
        
        
        

class NewBlock(Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        self.conv3dskip = Sequential(
            Conv3d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            GroupNorm(num_groups=8, num_channels=output_channels)
        )
        self.conv3d1 = Conv3d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3d2 = Conv3d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm3d1 = GroupNorm(num_groups=8, num_channels=input_channels)
        self.batchnorm3d2 = GroupNorm(num_groups=8, num_channels=output_channels)
        self.relu = ReLU(inplace=True)
        
    def forward(self, x):
        x = self.batchnorm3d1(x)
        x = self.relu(x)  
        x = self.conv3d1(x)
      
        x = self.batchnorm3d2(x)
        x = self.relu(x)
        x = self.conv3d2(x)        
        
        
        return x 

class FinalBlock(Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        self.conv3dskip = Sequential(
            Conv3d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm3d(output_channels)
        )
        self.conv3d1 = Conv3d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3d2 = Conv3d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm3d1 = GroupNorm(num_groups=8, num_channels=input_channels)
        self.batchnorm3d2 = BatchNorm3d(output_channels)
        self.relu = ReLU(inplace=True)
        
    def forward(self, x):
        x = self.batchnorm3d1(x)
        x = self.relu(x)  
        x = self.conv3d1(x)
      
        x = self.batchnorm3d2(x)
        x = self.relu(x)
        x = self.conv3d2(x)        
        
        
        return x 

    
class ConvBlock(Module):
    def __init__(self, input_channels, output_channels, padding = 1):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.padding = padding
        
        self.conv3d = Sequential(
            GroupNorm(num_groups=8, num_channels=input_channels),
            ReLU(inplace=True),            
            Conv3d(input_channels, output_channels, kernel_size=3, stride=1, padding=self.padding, bias=False)
        )
        
    def forward(self, x):
        return self.conv3d(x) 

class NewUNet(Module):
    def __init__(self, encoding_channels=(4,32,64,128,256), decoding_channels=(512,256,128,64,32,16,3), retainDim=True, outSize=(128, 128)):
        super().__init__()
        self.pool = MaxPool3d(kernel_size=2,stride=2)
        self.pool4 = MaxPool3d(kernel_size=4,stride=4)
        self.pool8 = MaxPool3d(kernel_size=8,stride=8)

        self.encblock1 = FirstBlock(encoding_channels[0], encoding_channels[1])
        self.encblock2 = NewBlock(encoding_channels[1], encoding_channels[2])
        self.encblock3 = NewBlock(encoding_channels[2], encoding_channels[3])
        self.encblock4 = NewBlock(encoding_channels[3], encoding_channels[4])
        
        self.Bottleneck = NewBlock(encoding_channels[4], decoding_channels[0]) 
        
        self.convtranspose3d4 = ConvTranspose3d(decoding_channels[0], decoding_channels[1], kernel_size=2, stride=2)
        self.convtranspose3d3 = ConvTranspose3d(decoding_channels[1], decoding_channels[2], kernel_size=2, stride=2)
        self.convtranspose3d2 = ConvTranspose3d(decoding_channels[2], decoding_channels[3], kernel_size=2, stride=2)
        self.convtranspose3d1 = ConvTranspose3d(decoding_channels[3], decoding_channels[4], kernel_size=2, stride=2)
        self.convtranspose3doutput = ConvTranspose3d(decoding_channels[4]*5, decoding_channels[5], kernel_size=(2,2,1), stride=(2,2,1))
        
        self.attention4 = AttentionBlock(decoding_channels[4],decoding_channels[4]*4,decoding_channels[4])
        self.attention3 = AttentionBlock(decoding_channels[4],decoding_channels[4]*4,decoding_channels[4])
        self.attention2 = AttentionBlock(decoding_channels[4],decoding_channels[4]*4,decoding_channels[4])
        self.attention1 = AttentionBlock(decoding_channels[4],decoding_channels[4]*4,decoding_channels[4])
        
        self.decblock4 = NewBlock(decoding_channels[4]*5, decoding_channels[4]*5)
        self.decblock3 = NewBlock(decoding_channels[4]*5, decoding_channels[4]*5)
        self.decblock2 = NewBlock(decoding_channels[4]*5, decoding_channels[4]*5)
        self.decblock1 = NewBlock(decoding_channels[4]*5, decoding_channels[4]*5)
        self.output = FinalBlock(decoding_channels[5], decoding_channels[6])
        
        self.upsample2 = Upsample(scale_factor=2, mode='trilinear')
        self.upsample4 = Upsample(scale_factor=4, mode='trilinear')
        self.upsample8 = Upsample(scale_factor=8, mode='trilinear')
        self.upsample16 = Upsample(scale_factor=16, mode='trilinear')
        
        self.Conv3d4 =ConvBlock(decoding_channels[1],decoding_channels[4])
        self.Conv3d3 =ConvBlock(decoding_channels[2],decoding_channels[4])
        self.Conv3d2 =ConvBlock(decoding_channels[3],decoding_channels[4])
        self.batchnorm3d = BatchNorm3d(decoding_channels[5])
        
        self.Conv3d_e1_d4 =ConvBlock(encoding_channels[1],decoding_channels[4])        
        self.Conv3d_e2_d4 =ConvBlock(encoding_channels[2],decoding_channels[4])
        self.Conv3d_e3_d4 =ConvBlock(encoding_channels[3],decoding_channels[4])
        self.Conv3d_e4_d4 =ConvBlock(encoding_channels[4],decoding_channels[4])
        self.Conv3d_neck_d4 =ConvBlock(decoding_channels[0],decoding_channels[4])
        
        self.Conv3d_e1_d3 =ConvBlock(encoding_channels[1],decoding_channels[4])        
        self.Conv3d_e2_d3 =ConvBlock(encoding_channels[2],decoding_channels[4])
        self.Conv3d_e3_d3 =ConvBlock(encoding_channels[3],decoding_channels[4])
        self.Conv3d_d4_d3 =ConvBlock(decoding_channels[4]*5,decoding_channels[4])
        self.Conv3d_neck_d3 =ConvBlock(decoding_channels[0],decoding_channels[4])

        self.Conv3d_e1_d2 =ConvBlock(encoding_channels[1],decoding_channels[4])
        self.Conv3d_e2_d2 =ConvBlock(encoding_channels[2],decoding_channels[4])        
        self.Conv3d_d3_d2 =ConvBlock(decoding_channels[4]*5,decoding_channels[4])
        self.Conv3d_d4_d2 =ConvBlock(decoding_channels[4]*5,decoding_channels[4])
        self.Conv3d_neck_d2 =ConvBlock(decoding_channels[0],decoding_channels[4])

        self.Conv3d_e1_d1 =ConvBlock(encoding_channels[1],decoding_channels[4])  
        self.Conv3d_d2_d1 =ConvBlock(decoding_channels[4]*5,decoding_channels[4])
        self.Conv3d_d3_d1 =ConvBlock(decoding_channels[4]*5,decoding_channels[4])
        self.Conv3d_d4_d1 =ConvBlock(decoding_channels[4]*5,decoding_channels[4])
        self.Conv3d_neck_d1 =ConvBlock(decoding_channels[0],decoding_channels[4])
        
    def forward(self, x):
        decskip1 = self.encblock1(x) # 4 128*128  -> 32 128*128 
        x = self.pool(decskip1)  #32 128*128 -> 32 64*64      
        
        decskip2 = self.encblock2(x) #32 64*64  ->  64  64*64
        x = self.pool(decskip2) # 64  64*64  ->  64  32*32
        
        decskip3 = self.encblock3(x)  #64 32*32  ->  128  32*32
        x = self.pool(decskip3)    #128  32*32   ->   128  16*16     
        
        decskip4 = self.encblock4(x)   #128  16*16  ->  256 16*16
        x = self.pool(decskip4)  #256 16*16  ->   256 8*8
        
        neck = self.Bottleneck(x)  #256  8*8  ->  512  8*8
        
        e1_d4 = self.pool8(decskip1) # 32 128*128   ->  32 16*16  
        e1_d4 = self.Conv3d_e1_d4(e1_d4)  #32 16*16 ->  32 16*16
        
        e2_d4 = self.pool4(decskip2)  # 64  64*64 - > 64  16*16
        e2_d4 = self.Conv3d_e2_d4(e2_d4)  #64  16*16  ->  32  16*16
        
        e3_d4 = self.pool(decskip3)  #128  32*32  ->  128  16*16
        e3_d4 = self.Conv3d_e3_d4(e3_d4)  #128  16*16  ->   32  16*16

        d4_skip = self.Conv3d_e4_d4(decskip4)  # 256 16*16  -> 32 16*16 
  
        neck_d4 = self.upsample2(neck)  #512  8*8  ->  512  16*16
        neck_d4 = self.Conv3d_neck_d4(neck_d4) # 512  16*16  ->  32 16*16 

        d4_skip = torch.cat((d4_skip, e1_d4, e2_d4, e3_d4 ), dim=1)  #  32*5 16*16         
         
        d4_skip = self.attention4(neck_d4, d4_skip) #32 16*16  -> 32 16*16 
        x = torch.cat((neck_d4, d4_skip), dim=1)  #  32*5 16*16 
        decoder_output4 = self.decblock4(x)    # 32*5 16*16  -> 32*5 16*16 
        
        e1_d3 = self.pool4(decskip1)  #32 128*128  -> 32 32*32
        e1_d3 = self.Conv3d_e1_d3(e1_d3)  #32 32*32  ->  32 32*32
        
        e2_d3 = self.pool(decskip2)  #64  64*64  ->  64  32*32
        e2_d3 = self.Conv3d_e2_d3(e2_d3)  #64  32*32  ->  32  32*32
        
        d3_skip = self.Conv3d_e3_d3(decskip3)  #128  32*32 -> 32  32*32
        
        neck_d3 = self.upsample4(neck)   #512  8*8 -> 512  32*32
        neck_d3 = self.Conv3d_neck_d3(neck_d3)  #512  32*32 ->  32  32*32
        
        d4_d3 = self.upsample2(decoder_output4)  #32*5 16*16  -> 32*5 32*32
        d4_d3 = self.Conv3d_d4_d3(d4_d3)  #32*5 32*32  ->  32 32*32
        
        d3_skip = torch.cat((d3_skip, e1_d3, e2_d3, neck_d3), dim=1)  ##  32*5 32*32 
        
        d3_skip = self.attention3(d4_d3, d3_skip)  #32  32*32  -> 32  32*32
        x = torch.cat((d4_d3, d3_skip), dim=1)  ##  32*5 32*32 
        decoder_output3 = self.decblock3(x)   # 32*5 32*32  -> 32*5 32*32 
        
        e1_d2 = self.pool(decskip1)  #32 128*128   ->   32 64*64
        e1_d2 = self.Conv3d_e1_d2(e1_d2)  #32 64*64 ->  32 64*64
        
        d2_skip = self.Conv3d_e2_d2(decskip2) #64  64*64  ->  32  64*64
        
        d3_d2 = self.upsample2(decoder_output3)  #32*5 32*32  -> 32*5 64*64
        d3_d2 = self.Conv3d_d3_d2(d3_d2)  #32*5 64*64  ->  32 64*64
        
        d4_d2 = self.upsample4(decoder_output4)   #32*5 16*16  ->  32*5 64*64
        d4_d2 = self.Conv3d_d4_d2(d4_d2)   #32*5 64*64  ->  32 64*64
        
        neck_d2 = self.upsample8(neck)   #512  8*8  -> 512  64*64
        neck_d2 = self.Conv3d_neck_d2(neck_d2)  #512  64*64  ->  32  64*64
        
        d2_skip = torch.cat((d2_skip, e1_d2, d4_d2, neck_d2), dim=1)  #32*5  64*64
        
        d2_skip = self.attention2(d3_d2, d2_skip)  #32  64*64 
        x = torch.cat((d3_d2, d2_skip), dim=1)  #32*5  64*64
        decoder_output2 = self.decblock2(x)    #32*5  64*64  ->  32*5  64*64  
        
        d1_skip = self.Conv3d_e1_d1(decskip1) #32 128*128  -> 32 128*128
        
        d2_d1 = self.upsample2(decoder_output2)  #32*5  64*64  ->  32*5  128*128
        d2_d1 = self.Conv3d_d2_d1(d2_d1)  #32*5  128*128    ->  32  128*128
        
        d3_d1 = self.upsample4(decoder_output3)  #32*5 32*32  ->  32*5 128*128
        d3_d1 = self.Conv3d_d3_d1(d3_d1)  #32*5 128*128  ->  32 128*128
        
        d4_d1 = self.upsample8(decoder_output4)  #32*5 16*16   ->  32*5 128*128
        d4_d1 = self.Conv3d_d4_d1(d4_d1)  #32*5 128*128  ->  32 128*128
        
        neck_d1 = self.upsample16(neck)   #512 8*8   ->  512 128*128
        neck_d1 = self.Conv3d_neck_d1(neck_d1)  #512 128*128  ->  32*5 128*128
        
        d1_skip = torch.cat((d1_skip, d3_d1, d4_d1, neck_d1), dim=1)  #32*5 128*128
        
        d1_skip = self.attention1(d2_d1, d1_skip)  #32  128*128
        x = torch.cat((d2_d1, d1_skip), dim=1)  #32*5 128*128
        decoder_output1 = self.decblock1(x)     #32*5 128*128    ->  32*5 128*128  
     
        output = self.convtranspose3doutput(decoder_output1) #32*5 128*128  -> 16 256*256       
        output = self.output(output)   #16 256*256    ->  3 256*256

        output = torch.nn.functional.interpolate(output, (240,240,144))
        return output
        
        
class ReadDataset(Dataset):
    def __init__(self,datasetpath, size = [128, 128]): 
        self.datasetpath=datasetpath
        self.size=size
        
        self.image_ext = ['_flair.nii', '_t1.nii', '_t1ce.nii', '_t2.nii']
        self.mask_ext = ['_seg.nii']
        
        self.datasetfolds=os.listdir(self.datasetpath)
        self.datasetfolds.sort()
    def __len__(self):
        return len(self.datasetfolds)    
    
    def __getitem__(self,index):
        datasetfoldpath = os.path.join(self.datasetpath,self.datasetfolds[index])
        datasetprefixpath = os.path.join(datasetfoldpath,self.datasetfolds[index])
        
        images = []
        for file_ext in self.image_ext:
            datasetfilepath = datasetprefixpath + file_ext
            img = nib.load(datasetfilepath)
            img = np.asarray(img.dataobj)
            img = img[:,:,6:150]
            img = img.astype(np.float32)
            img = resize(img, (112, 112, 144), preserve_range=True)
            
			if np.max(img) > np.min(img):
			    img = (img - np.min(img)) / (np.max(img) - np.min(img))

            images.append(img)
        image = np.stack(images)
    
        for file_ext in self.mask_ext:
            datasetfilepath =  datasetprefixpath + file_ext 
            msk = nib.load(datasetfilepath)
            msk = np.asarray(msk.dataobj)
            msk = msk[:,:,6:150]
            msk = msk.astype(np.uint8)
            mask = self.mask_reorganize(msk)

        
        return {
            "image": image,
            "mask":mask
        }

    def mask_reorganize(self,mask):
        
        mask_WT = np.zeros_like(mask)
		mask_TC = np.zeros_like(mask)
		mask_ET = np.zeros_like(mask)
		
		# whole tumour
        mask_WT[mask>0] = 1
        
        # NCR / NET - LABEL 1
        mask_TC[(mask == 1) | (mask == 4)] = 1
        
        # ET - LABEL 4 
        mask_ET[mask == 4] = 1

        
        mask = np.stack([mask_WT, mask_TC, mask_ET])
        mask = mask.astype(np.float32)
        
        return mask
        
        
class dice_loss_fun(nn.Module):
    def __init__(self, eps: float = 1e-9):
        super(dice_loss_fun, self).__init__()
        self.eps = eps
        
    def forward(self, prediction, true_mask):
        
        num = true_mask.size(0)
        probability = torch.sigmoid(prediction)
        probability = probability.reshape(num, -1)
        true_mask = true_mask.reshape(num, -1)
        assert(probability.shape == true_mask.shape)
        
        intersection = 2.0*(probability * true_mask).sum()
        union = probability.sum() + true_mask.sum()
        dice_loss = (intersection + self.eps) / (union  + self.eps)
        return 1.0 - dice_loss
        
        
class bce_dice_loss_fun(nn.Module):
    def __init__(self):
        super(bce_dice_loss_fun, self).__init__()
        self.dice = dice_loss_fun()
        
    def forward(self, prediction, true_mask):
        
        assert(prediction.shape == true_mask.shape)
        dice_loss = self.dice(prediction, true_mask)
        
        num_classes = true_mask.shape[1]
        class_weights = torch.tensor([1.76, 4.23, 9.50], dtype=torch.float32)  # asign weight to each class
        weights = class_weights.view(1, num_classes, 1, 1, 1)  # expand dimension to (1, C, 1, 1) 
        weights = weights.expand_as(true_mask)  # expand the size as the true_mask
        bce = BCEWithLogitsLoss(pos_weight=weights.to(device))
        bce_loss = bce(prediction, true_mask)

        return bce_loss + dice_loss

def calculate_Dice(predictions, true_mask):
    assert(predictions.shape == true_mask.shape)
    predictions = torch.sigmoid(predictions)
    num_classes = predictions.shape[1]
    Dice = []
    for i in range(0, num_classes):
        predict = predictions[:,i,:,:] > 0.5
        true_label = true_mask[:,i,:,:]
        intersection = 2.0 * (predict * true_label).sum()
        total = predict.sum() + true_label.sum()
        intersection = intersection.to("cpu")
        total = total.to("cpu")
        Dice.append([intersection, total])
    return Dice
            



def calculate_IOU(predictions, true_mask):
    assert(predictions.shape == true_mask.shape)
    predictions = torch.sigmoid(predictions)
    num_classes = predictions.shape[1]
    IOU = []
    for i in range(0, num_classes):
        predict = (predictions[:,i,:,:,:] > 0.5).int()
        true_label = true_mask[:,i,:,:,:]
        intersection = (predict * true_label).sum(dim=(0, 1, 2))
        union = predict.sum(dim=(0, 1, 2)) + true_label.sum(dim=(0, 1, 2)) -intersection
        intersection = intersection.to("cpu").numpy()
        union=union.to("cpu").numpy()
        IOU.append([intersection,union])            
    return IOU     
        
def calculate_scores(predictions, true_mask):
    assert(predictions.shape == true_mask.shape)

    confusionmatrix_metric = BinaryConfusionMatrix().to(device)

    confusionmatrix = []
    
    predictions = torch.sigmoid(predictions)
    num_classes = predictions.shape[1]
    for i in range(0, num_classes):
        predict = (predictions[:,i,:,:,:] > 0.5).int()
        true_label = true_mask[:,i,:,:,:]
        confusionmatrix.append(confusionmatrix_metric(predict, true_label).cpu().numpy())
    return confusionmatrix
    
batch_size=1
num_epochs=200

path = "../BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
dataset = ReadDataset(path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

learning_rate=0.0003
weight_decay=0.0000

loss_function= bce_dice_loss_fun()
  
UNET_Model= NewUNet()
UNET_Model.to(device)
optimizer_function= torch.optim.Adam(params=UNET_Model.parameters(),lr=learning_rate, weight_decay=weight_decay)
UNET_Model.train()

IOU_per_epoch=[]
DCS_per_epoch=[]
result_recode = []

bias_index = 0
count = 0
for i in range(bias_index + 1,num_epochs+bias_index + 1):
    #i = 1
    
    IOU_list = []
    Dice_list = []    
    loss_list=[]

    confusionmatrix_list = []
    
    batch_num = 0
    
    for itr, data_batch in tqdm.tqdm(enumerate(dataloader),total=len(dataloader)):
        batch_num + 1
        image = data_batch["image"]
        mask = data_batch["mask"]
        
        image=image.to(device)
        mask=mask.to(device)



        predictions=UNET_Model(image)

        loss=loss_function(predictions,mask)
        loss_list.append(loss.item())
        optimizer_function.zero_grad()
        loss.backward()
        optimizer_function.step()
        
        IOU_list.append(calculate_IOU(predictions,mask))
        Dice_list.append(calculate_Dice(predictions,mask))
        
        confusionmatrix = calculate_scores(predictions,mask)
        confusionmatrix_list.append(confusionmatrix)
    
    IOU_list= np.array(IOU_list)
    Dice_list= np.array(Dice_list)
    
    IOU_list = np.moveaxis(IOU_list, (0, 1, 2, 3), (0, 2, 3, 1))
    Dice_list = np.moveaxis(Dice_list, (0, 1, 2, 3), (0, 2, 3, 1))  

    IOU_box =  np.sum(IOU_list[:,:,:,0], axis=(1))/np.sum(IOU_list[:,:,:,1], axis=(1))
    Dice_box = np.sum(Dice_list[:,:,:,0], axis=(1))/np.sum(Dice_list[:,:,:,1], axis=(1))
    
    box_data_path="./3ClassNet3-res-att-IOU_Dice_data_for_box"
    with open(box_data_path, 'w') as file:
        file.write(f"{IOU_box}\n")
        file.write(f"{Dice_box}\n")
        file.close() 

    IOU_mean = np.mean(IOU_list, axis=(0,1))
    Dice_mean = np.mean(Dice_list, axis=(0,1))
    IOU_mean = IOU_mean[:,0]/IOU_mean[:,1]
    Dice_mean = Dice_mean[:,0]/Dice_mean[:,1]     
        
    loss_mean = np.mean(np.array(loss_list))   
    confusionmatrix_mean = np.mean(np.array(confusionmatrix_list), axis=0)
    TP = confusionmatrix_mean[:,1,1]
    TN = confusionmatrix_mean[:,0,0]
    FP = confusionmatrix_mean[:,0,1]
    FN = confusionmatrix_mean[:,1,0]
    accuracy_mean = (TP+TN)/(TP + TN + FP + FN)    
    precision_mean = TP/(TP + FP)
    recall_mean = TP/(TP + FN)
    F1_mean = 2 * precision_mean * recall_mean/(precision_mean + recall_mean)
    confusionmatrix_mean = confusionmatrix_mean//(batch_size * 155)
    np.set_printoptions(suppress=True)
    result_recode.append({"IOU":IOU_mean,"Dice":Dice_mean,"loss":loss_mean,"accuracy":accuracy_mean,"precision":precision_mean,"recall":recall_mean,"F1":F1_mean,"confusionmatrix":confusionmatrix_mean})
    print("")    
    print(f"----------------------epoch {i}-----------------------------------------")
    print(f"The mean IOU for class WT is: ",IOU_mean[0] )
    print(f"The mean IOU for class TC is: ",IOU_mean[1] )
    print(f"The mean IOU for class ET is: ",IOU_mean[2] )
    print("")    
    print(f"The mean Dice for class WT is: ",Dice_mean[0] )
    print(f"The mean Dice for class TC is: ",Dice_mean[1] )
    print(f"The mean Dice for class ET is: ",Dice_mean[2] )
    print("")
    print(f"At end of epoch {i}, the mean loss is: ",loss_mean)
    print("")
    print(f"The mean accuracy for class WT is: ",accuracy_mean[0])
    print(f"The mean accuracy for class TC is: ",accuracy_mean[1])
    print(f"The mean accuracy for class ET is: ",accuracy_mean[2])
    print("")
    print(f"The mean precision for class WT is: ",precision_mean[0])
    print(f"The mean precision for class TC is: ",precision_mean[1])
    print(f"The mean precision for class ET is: ",precision_mean[2])
    print("")
    print(f"The mean recall for class WT is: ",recall_mean[0])
    print(f"The mean recall for class TC is: ",recall_mean[1])
    print(f"The mean recall for class ET is: ",recall_mean[2])
    print("")
    print(f"The mean f1 for class WT is: ",F1_mean[0])
    print(f"The mean f1 for class TC is: ",F1_mean[1])
    print(f"The mean f1 for class ET is: ",F1_mean[2])
    print("\n")    
    print(f"The mean confusion matrix for class WT is:\n",confusionmatrix_mean[0])
    print(f"The mean confusion matrix for class TC is:\n",confusionmatrix_mean[1])
    print(f"The mean confusion matrix for class ET is:\n",confusionmatrix_mean[2])
        
    if loss_mean < 0.070:
       break
       
       
    
print("Training completed")
model_path="./UNet3-att-epochs"
model_path_pth = model_path + ".pth"
torch.save(UNET_Model,model_path)
torch.save({
    'model_state_dict': UNET_Model.state_dict(),
    'optimizer_state_dict': optimizer_function.state_dict(),
    }, model_path_pth)
        
recode_path="./UNet3-att-result_recode"
with open(recode_path, 'w') as file:
    for label in result_recode:
        file.write(f"{label}\n")
    file.close()

print("model and recode saved")
