import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def show_image():
    
    origin_image_path1 = "./archive2/png/test/BraTS20_Training_003/t1/BraTS20_Training_003_73.png"    
    true_mask_path1 = "./archive2/png/test/BraTS20_Training_003/seg/BraTS20_Training_003_73.png"
    bas_mask_path1 = "./good/Unet3/mask/BraTS20_Training_003/slice_73.png"
    res_mask_path1 = "./good/residual-Unet3/mask/BraTS20_Training_003/slice_73.png"
    att_mask_path1 = "./good/attention-Unet3/mask/BraTS20_Training_003/slice_73.png"
    res_att_mask_path1 = "./good/residual-attention-Unet3/mask/BraTS20_Training_003/slice_73.png" 
    
    origin_image_path2 = "./archive2/png/test/BraTS20_Training_005/t1/BraTS20_Training_005_94.png"    
    true_mask_path2 = "./archive2/png/test/BraTS20_Training_005/seg/BraTS20_Training_005_94.png"
    bas_mask_path2 = "./good/Unet3/mask/BraTS20_Training_005/slice_94.png"
    res_mask_path2 = "./good/residual-Unet3/mask/BraTS20_Training_005/slice_94.png"
    att_mask_path2 = "./good/attention-Unet3/mask/BraTS20_Training_005/slice_94.png"
    res_att_mask_path2 = "./good/residual-attention-Unet3/mask/BraTS20_Training_005/slice_94.png" 
    
    origin_image_path3 = "./archive2/png/test/BraTS20_Training_023/t1/BraTS20_Training_023_92.png"    
    true_mask_path3 = "./archive2/png/test/BraTS20_Training_023/seg/BraTS20_Training_023_92.png"
    bas_mask_path3 = "./good/Unet3/mask/BraTS20_Training_023/slice_92.png"
    res_mask_path3 = "./good/residual-Unet3/mask/BraTS20_Training_023/slice_92.png"
    att_mask_path3 = "./good/attention-Unet3/mask/BraTS20_Training_023/slice_92.png"
    res_att_mask_path3 = "./good/residual-attention-Unet3/mask/BraTS20_Training_023/slice_92.png" 
    
    origin_image_path4 = "./archive2/png/test/BraTS20_Training_037/t1/BraTS20_Training_037_67.png"    
    true_mask_path4 = "./archive2/png/test/BraTS20_Training_037/seg/BraTS20_Training_037_67.png"
    bas_mask_path4 = "./good/Unet3/mask/BraTS20_Training_037/slice_67.png"
    res_mask_path4 = "./good/residual-Unet3/mask/BraTS20_Training_037/slice_67.png"
    att_mask_path4 = "./good/attention-Unet3/mask/BraTS20_Training_037/slice_67.png"
    res_att_mask_path4 = "./good/residual-attention-Unet3/mask/BraTS20_Training_037/slice_67.png" 
    
    origin_image_path5 = "./archive2/png/test/BraTS20_Training_102/t1/BraTS20_Training_102_94.png"    
    true_mask_path5 = "./archive2/png/test/BraTS20_Training_102/seg/BraTS20_Training_102_94.png"
    bas_mask_path5 = "./good/Unet3/mask/BraTS20_Training_102/slice_94.png"
    res_mask_path5 = "./good/residual-Unet3/mask/BraTS20_Training_102/slice_94.png"
    att_mask_path5 = "./good/attention-Unet3/mask/BraTS20_Training_102/slice_94.png"
    res_att_mask_path5 = "./good/residual-attention-Unet3/mask/BraTS20_Training_102/slice_94.png" 
    
    
    # read image
    origin_image1 = Image.open(origin_image_path1)
    origin_image1 = np.array(origin_image1)
    true_mask1 = Image.open(true_mask_path1)
    bas_mask1 = Image.open(bas_mask_path1)
    res_mask1 = Image.open(res_mask_path1)
    att_mask1 = Image.open(att_mask_path1)
    res_att_mask1 = Image.open(res_att_mask_path1)
    
    # read image
    origin_image2 = Image.open(origin_image_path2)
    origin_image2 = np.array(origin_image2)
    true_mask2 = Image.open(true_mask_path2)
    bas_mask2 = Image.open(bas_mask_path2)
    res_mask2 = Image.open(res_mask_path2)
    att_mask2 = Image.open(att_mask_path2)
    res_att_mask2 = Image.open(res_att_mask_path2)
    
    # read image
    origin_image3 = Image.open(origin_image_path3)
    origin_image3 = np.array(origin_image3)
    true_mask3 = Image.open(true_mask_path3)
    bas_mask3 = Image.open(bas_mask_path3)
    res_mask3 = Image.open(res_mask_path3)
    att_mask3 = Image.open(att_mask_path3)
    res_att_mask3 = Image.open(res_att_mask_path3)
    
    # read image
    origin_image4 = Image.open(origin_image_path4)
    origin_image4 = np.array(origin_image4)
    true_mask4 = Image.open(true_mask_path4)
    bas_mask4 = Image.open(bas_mask_path4)
    res_mask4 = Image.open(res_mask_path4)
    att_mask4 = Image.open(att_mask_path4)
    res_att_mask4 = Image.open(res_att_mask_path4)
    
    # read image
    origin_image5 = Image.open(origin_image_path5)
    origin_image5 = np.array(origin_image5)
    true_mask5 = Image.open(true_mask_path5)
    bas_mask5 = Image.open(bas_mask_path5)
    res_mask5 = Image.open(res_mask_path5)
    att_mask5 = Image.open(att_mask_path5)
    res_att_mask5 = Image.open(res_att_mask_path5)
    
    # display image
    plt.figure(figsize=(20, 20))
    
    plt.subplot(5, 5, 1)
    plt.imshow(origin_image1, cmap='gray', vmin=0, vmax=65535)
    plt.imshow(true_mask1, alpha=0.6)
    plt.title('Ground truth', fontsize=20)
    plt.axis('off')  
    
    plt.subplot(5, 5, 2)
    plt.imshow(origin_image1, cmap='gray', vmin=0, vmax=65535)
    plt.imshow(res_att_mask1, alpha=0.6)
    plt.title('Residual attention U-Net3+', fontsize=20)
    plt.axis('off')
    
    plt.subplot(5, 5, 3)
    plt.imshow(origin_image1, cmap='gray', vmin=0, vmax=65535)
    plt.imshow(res_mask1, alpha=0.6)
    plt.title('Residual U-Net3+', fontsize=20)
    plt.axis('off')  
    
    plt.subplot(5, 5, 4)
    plt.imshow(origin_image1, cmap='gray', vmin=0, vmax=65535)
    plt.imshow(att_mask1, alpha=0.6)
    plt.title('Attention U-Net3+', fontsize=20)
    plt.axis('off') 
    
    plt.subplot(5, 5, 5)
    plt.imshow(origin_image1, cmap='gray', vmin=0, vmax=65535)
    plt.imshow(bas_mask1, alpha=0.6)
    plt.title('U-Net3+', fontsize=20)
    plt.axis('off') 
    
    
    
    
    plt.subplot(5, 5, 6)
    plt.imshow(origin_image2, cmap='gray', vmin=0, vmax=65535)
    plt.imshow(true_mask2, alpha=0.6)
    plt.axis('off')  
    
    plt.subplot(5, 5, 7)
    plt.imshow(origin_image2, cmap='gray', vmin=0, vmax=65535)
    plt.imshow(res_att_mask2, alpha=0.6)
    plt.axis('off')
    
    plt.subplot(5, 5, 8)
    plt.imshow(origin_image2, cmap='gray', vmin=0, vmax=65535)
    plt.imshow(res_mask2, alpha=0.6)
    plt.axis('off')  
    
    plt.subplot(5, 5, 9)
    plt.imshow(origin_image2, cmap='gray', vmin=0, vmax=65535)
    plt.imshow(att_mask2, alpha=0.6)
    plt.axis('off') 
    
    plt.subplot(5, 5, 10)
    plt.imshow(origin_image2, cmap='gray', vmin=0, vmax=65535)
    plt.imshow(bas_mask2, alpha=0.6)
    plt.axis('off') 
    
    
    
    plt.subplot(5, 5, 11)
    plt.imshow(origin_image3, cmap='gray', vmin=0, vmax=65535)
    plt.imshow(true_mask3, alpha=0.6)
    plt.axis('off')  
    
    plt.subplot(5, 5, 12)
    plt.imshow(origin_image3, cmap='gray', vmin=0, vmax=65535)
    plt.imshow(res_att_mask3, alpha=0.6)
    plt.axis('off')
    
    plt.subplot(5, 5, 13)
    plt.imshow(origin_image3, cmap='gray', vmin=0, vmax=65535)
    plt.imshow(res_mask3, alpha=0.6)
    plt.axis('off')  
    
    plt.subplot(5, 5, 14)
    plt.imshow(origin_image3, cmap='gray', vmin=0, vmax=65535)
    plt.imshow(att_mask3, alpha=0.6)
    plt.axis('off') 
    
    plt.subplot(5, 5, 15)
    plt.imshow(origin_image3, cmap='gray', vmin=0, vmax=65535)
    plt.imshow(bas_mask3, alpha=0.6)
    plt.axis('off') 
    
    

    
    plt.subplot(5, 5, 16)
    plt.imshow(origin_image4, cmap='gray', vmin=0, vmax=65535)
    plt.imshow(true_mask4, alpha=0.6)
    plt.axis('off')  
    
    plt.subplot(5, 5, 17)
    plt.imshow(origin_image4, cmap='gray', vmin=0, vmax=65535)
    plt.imshow(res_att_mask4, alpha=0.6)
    plt.axis('off')
    
    plt.subplot(5, 5, 18)
    plt.imshow(origin_image4, cmap='gray', vmin=0, vmax=65535)
    plt.imshow(res_mask4, alpha=0.6)
    plt.axis('off')  
    
    plt.subplot(5, 5, 19)
    plt.imshow(origin_image4, cmap='gray', vmin=0, vmax=65535)
    plt.imshow(att_mask4, alpha=0.6)
    plt.axis('off') 
    
    plt.subplot(5, 5, 20)
    plt.imshow(origin_image4, cmap='gray', vmin=0, vmax=65535)
    plt.imshow(bas_mask4, alpha=0.6)
    plt.axis('off') 
    
    
    
    plt.subplot(5, 5, 21)
    plt.imshow(origin_image5, cmap='gray', vmin=0, vmax=65535)
    plt.imshow(true_mask5, alpha=0.6)
    plt.axis('off')  
    
    plt.subplot(5, 5, 22)
    plt.imshow(origin_image5, cmap='gray', vmin=0, vmax=65535)
    plt.imshow(res_att_mask5, alpha=0.6)
    plt.axis('off')
    
    plt.subplot(5, 5, 23)
    plt.imshow(origin_image5, cmap='gray', vmin=0, vmax=65535)
    plt.imshow(res_mask5, alpha=0.6)
    plt.axis('off')  
    
    plt.subplot(5, 5, 24)
    plt.imshow(origin_image5, cmap='gray', vmin=0, vmax=65535)
    plt.imshow(att_mask5, alpha=0.6)
    plt.axis('off') 
    
    plt.subplot(5, 5, 25)
    plt.imshow(origin_image5, cmap='gray', vmin=0, vmax=65535)
    plt.imshow(bas_mask5, alpha=0.6)
    plt.axis('off')
    
    plt.tight_layout()
    
    plt.show()

show_image()

