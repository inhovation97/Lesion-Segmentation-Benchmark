import glob
import os
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import time
import torch

def masked_raw_imgplot(img_path, label_path):
    #   PILImage.open(img_path)
    #   PILImage.open(label_path)

    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    plt.axis('off')

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 양성형 이웃 보간 (2x2 픽셀 참조하여 보간함.)
    img = cv2.resize(img, (256,256), interpolation=cv2.INTER_LINEAR)

    label = cv2.imread(label_path)
    label = cv2.cvtColor(label, cv2.COLOR_RGB2BGR)
    # 양선형 이웃 보간 (2x2 픽셀 참조하여 보간함.)
    label = cv2.resize(label, (256,256), interpolation=cv2.INTER_LINEAR)

    # 마스크 만들기
    masking = label.copy()
    masking[masking == 255] = 1

    masked_img = img * masking
    plt.imshow( np.concatenate([img, label, masked_img], axis = 1) )

def testing_seged_image(self):
    # 랜덤 검수
    import random
    img_path = random.choice(self.valid_path)

    fig = plt.gcf()
    fig.set_size_inches(12, 5)
    plt.axis('off')

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 양성형 이웃 보간 (2x2 픽셀 참조하여 보간함.)
    img = cv2.resize(img, (256,256), interpolation = cv2.INTER_LINEAR)
    
    self.model.eval()
    eval_image = img/255.0
    eval_image = eval_image.astype(np.float32)
    eval_image = eval_image.transpose((2,0,1))
    eval_image = torch.from_numpy(eval_image).unsqueeze(0) # Batch 채널 추가 -> (1, 3, 256, 256)
    eval_image = eval_image.to( device=self.device, dtype = torch.float32 )

    with torch.no_grad():
      output = self.model(eval_image) # (1, 2, 256, 256)
      mask = torch.argmax( output, dim=1 ) # (1, 256, 256)
    mask = mask.squeeze() # (256, 256)
    mask = mask.to('cpu').numpy() # tensor to numpy (반드시 디바이스도 변경)
    mask = np.stack( (mask,)*3, axis=-1 ) # (256,256,3)
    # 양선형 이웃 보간 (2x2 픽셀 참조하여 보간함.)
    mask = cv2.resize(mask, (256,256), interpolation=cv2.INTER_LINEAR)
    # 마스킹을 보여주기 위해 흰색처리
    real_mask = mask.copy()
    real_mask[real_mask == 1] = 255
    # segmentationed image
    masked_img = img * mask

    combined = np.concatenate([img, real_mask, masked_img], axis = 1)
    plt.axis('off')
    plt.imshow(combined)
    plt.show()

def plot_model_prediction(model, DEVICE, valid_images, valid_labels):#, save_dir, epochs):
    
    size = (224, 224)
    
    i = random.randint(0, len(valid_images)-1)
    path_img = valid_images[i]
    path_label = valid_labels[i]
    
    # figure 생성
    fig = plt.gcf()
    fig.set_size_inches(12, 5)
    plt.axis('off')
    
    # eval 전 이미지 전처리
    img = cv2.imread(path_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 양성형 이웃 보간 (2x2 픽셀 참조하여 보간함.)
    img = cv2.resize(img, size, interpolation = cv2.INTER_LINEAR)
    
    # label 이미지 전처리
    label = cv2.imread(path_label)
    # 양성형 이웃 보간 (2x2 픽셀 참조하여 보간함.)
    label = cv2.resize(label, size, interpolation = cv2.INTER_LINEAR)

    model.eval()
    eval_image = img / 255.0
    eval_image = eval_image.astype(np.float32)
    eval_image = eval_image.transpose((2,0,1))
    eval_image = torch.from_numpy(eval_image).unsqueeze(0) # Batch 채널 추가 -> (1, 3, 256, 256)
    eval_image = eval_image.to( device=DEVICE, dtype = torch.float32 )

    # we do not need to calculate gradients
    with torch.no_grad():
        # Prediction
        pred = model(eval_image)

    # put on cpu
#     pred = pred.cpu()

    # pass sigloid 
    # pred = torch.sigmoid(pred)
    
    # dict형태로 데이터가 들어오는 경우가 있음 ######################################################################
    
    if isinstance(pred, dict):
        pred = torch.sigmoid(pred['out'])
        
    else:
        pred = torch.sigmoid(pred)  
    
    mask = pred.clone()
    
    # 0.5를 기준으로 마스크 만들기.
    mask[mask >= 0.5 ] = 1
    mask[mask < 0.5 ] = 0
    mask = mask.squeeze() # (256, 256)
    mask = mask.to(device = 'cpu', dtype = torch.int64).numpy() # tensor to numpy (반드시 디바이스도 변경)
    mask = np.stack( (mask,)*3, axis=-1 ) # (256,256,3)
    
    # 마스킹을 보여주기 위해 흰색처리
    real_mask = mask.copy()
    real_mask[real_mask == 1] = 255
    
    # segmentationed image
    masked_img = img * mask
    
    # 예측 결과 plot
    combined = np.concatenate([img, label, real_mask, masked_img], axis = 1)
    plt.axis('off')
    plt.imshow(combined)
    plt.show()
    
    
    # save
#     pred_path = os.path.join(MASK_PATH, name)
#     cv2.imwrite(pred_path, pred)

def plot_test_image(models :list, DEVICE, test_images, test_labels, data_name):#, save_dir, epochs):
    size = (224,224)
    
    # i = random.randint(0, len(test_images)-1)
    i = 19
    path_img = test_images[i]
    path_label = test_labels[i]
    
    # figure 생성
    fig = plt.gcf()
    fig.set_size_inches(12, 5)
    plt.axis('off')
    
    # eval 전 이미지 전처리
    img = cv2.imread(path_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 양성형 이웃 보간 (2x2 픽셀 참조하여 보간함.)
    img = cv2.resize(img, size, interpolation = cv2.INTER_LINEAR)
    
    # label 이미지 전처리
    label = cv2.imread(path_label)
    # 양성형 이웃 보간 (2x2 픽셀 참조하여 보간함.)
    label = cv2.resize(label, size, interpolation = cv2.INTER_LINEAR)

    combined = np.concatenate([img, label], axis = 1)   # combine img, label
    
    for model in models:
        model.eval()
        eval_image = img / 255.0
        eval_image = eval_image.astype(np.float32)
        eval_image = eval_image.transpose((2,0,1))
        eval_image = torch.from_numpy(eval_image).unsqueeze(0) # Batch 채널 추가 -> (1, 3, 256, 256)
        eval_image = eval_image.to( device=DEVICE, dtype = torch.float32 )

        with torch.no_grad():
            pred = model(eval_image)

        # pass sigloid 
        if isinstance(pred, dict):
            pred = torch.sigmoid(pred['out'])

        else:
            pred = torch.sigmoid(pred)
        mask = pred.clone()

        # 0.5를 기준으로 마스크 만들기.
        mask[mask >= 0.5 ] = 1
        mask[mask < 0.5 ] = 0
        mask = mask.squeeze() # (256, 256)
        mask = mask.to(device = 'cpu', dtype = torch.int64).numpy() # tensor to numpy (반드시 디바이스도 변경)
        mask = np.stack( (mask,)*3, axis=-1 ) # (256,256,3)

        # 마스킹을 보여주기 위해 흰색처리
        real_mask = mask.copy()
        real_mask[real_mask == 1] = 255

        # 예측 결과 plot(re)
        combined = np.concatenate([combined, real_mask], axis = 1)
        
    plt.axis('off')
    
    plt.imshow(combined)
    plt.savefig(f'{data_name}.png', bbox_inches='tight') 
    plt.show() 