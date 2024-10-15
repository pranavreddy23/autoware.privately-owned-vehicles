
#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import torch
from torchvision import transforms
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
from model_components.scene_seg_network import SceneSegNetwork
from data_utils.load_data import LoadData
from data_utils.augmentations import Augmentations
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# Checking devices (GPU vs CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} for inference')

# Load Image as Tensor
def load_image_tensor(image):
    image = image_loader(image)
    image = image.unsqueeze(0)
    return image.to(device)

image_loader = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

# Load Ground Truth as Tensor
def load_gt_tensor(gt):
    gt = gt_loader(gt)
    gt = gt.unsqueeze(0)
    return gt.to(device)

gt_loader = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# Visualize predicted result
def visualize_result(prediction):
    shape = prediction.shape
    _, output = torch.max(prediction, dim=2)

    row = shape[0]
    col = shape[1]
    vis_predict = Image.new(mode="RGB", size=(col, row))
 
    vx = vis_predict.load()

    background_objects_colour = (61, 93, 255)
    foreground_objects_colour = (255, 28, 145)
    road_colour = (0, 255, 220)

    # Extracting predicted classes and assigning to colourmap
    for x in range(row):
        for y in range(col):
            if(output[x,y].item() == 0):
                vx[y,x] = background_objects_colour
            elif(output[x,y].item() == 1):
                 vx[y,x] = foreground_objects_colour
            elif(output[x,y].item() == 2):
                 vx[y,x] = road_colour               
    
    return vis_predict

def main():

    # Instantiate model 
    model = SceneSegNetwork().to(device)

    # epochs, learning rate
    num_epochs = 50
    learning_rate = 0.0001

    # optimizer
    optimizer = optim.AdamW(model.parameters(), learning_rate)
    optimizer.zero_grad()

    # TensorBoard
    writer = SummaryWriter()

    # Root path
    root = '/home/zain/Autoware/AutoSeg/training_data/Scene_Seg/'

    # Model save path
    model_save_path = '/home/zain/Autoware/AutoSeg/Models/exports/SceneSeg/'

    # Data paths
    # ACDC
    acdc_labels_filepath= root + 'ACDC/gt_masks/'
    acdc_images_filepath = root + 'ACDC/images/'

    # BDD100K
    bdd100k_labels_fileapath = root + 'BDD100K/gt_masks/'
    bdd100k_images_fileapath = root + 'BDD100K/images/'

    # IDDAW
    iddaw_labels_fileapath = root + 'IDDAW/gt_masks/'
    iddaw_images_fileapath = root + 'IDDAW/images/'

    # MUSES
    muses_labels_fileapath = root + 'MUSES/gt_masks/'
    muses_images_fileapath = root + 'MUSES/images/'

    # MAPILLARY
    mapillary_labels_fileapath = root + 'Mapillary_Vistas/gt_masks/'
    mapillary_images_fileapath = root + 'Mapillary_Vistas/images/'

    # COMMA10K
    comma10k_labels_fileapath = root + 'comma10k/gt_masks/'
    comma10k_images_fileapath = root + 'comma10k/images/'


    # ACDC - Data Loading
    acdc_Dataset = LoadData(acdc_labels_filepath, acdc_images_filepath, 'ACDC')
    acdc_num_train_samples, acdc_num_val_samples = acdc_Dataset.getItemCount()

    # BDD100K - Data Loading
    bdd100k_Dataset = LoadData(bdd100k_labels_fileapath, bdd100k_images_fileapath, 'BDD100K')
    bdd100k_num_train_samples, bdd100k_num_val_samples = bdd100k_Dataset.getItemCount()

    # IDDAW - Data Loading
    iddaw_Dataset = LoadData(iddaw_labels_fileapath, iddaw_images_fileapath, 'IDDAW')
    iddaw_num_train_samples, iddaw_num_val_samples = iddaw_Dataset.getItemCount()

    # MUSES - Data Loading
    muses_Dataset = LoadData(muses_labels_fileapath, muses_images_fileapath, 'MUSES')
    muses_num_train_samples, muses_num_val_samples = muses_Dataset.getItemCount()

    # Mapillary - Data Loading
    mapillary_Dataset = LoadData(mapillary_labels_fileapath, mapillary_images_fileapath, 'MAPILLARY')
    mapillary_num_train_samples, mapillary_num_val_samples = mapillary_Dataset.getItemCount()

    # comma10k - Data Loading
    comma10k_Dataset = LoadData(comma10k_labels_fileapath, comma10k_images_fileapath, 'COMMA10K')
    comma10k_num_train_samples, comma10k_num_val_samples = comma10k_Dataset.getItemCount()

    # Iterators for datasets
    acdc_count = 0
    bdd100k_count = 0
    iddaw_count = 0
    muses_count = 0
    comma10k_count = 0
    mapillary_count = 0

    data_list = []
    data_list.append('ACDC')
    data_list.append('BDD100K')
    data_list.append('IDDAW')
    data_list.append('MUSES')
    data_list.append('MAPILLARY')
    data_list.append('COMMA10K')
    data_list_count = 0

    # Total number of training samples
    total_train_samples = acdc_num_train_samples + bdd100k_num_train_samples \
    + iddaw_num_train_samples + muses_num_train_samples \
    + mapillary_num_train_samples + comma10k_num_train_samples
    print(total_train_samples, ': total training samples')

    # Total number of validation samples
    total_val_samples = acdc_num_val_samples + bdd100k_num_val_samples \
    + iddaw_num_val_samples + muses_num_val_samples \
    + mapillary_num_val_samples + comma10k_num_val_samples
    print(total_val_samples, ': total validation samples')

    # Epochs
    for epoch in range(0, num_epochs):

        # Loop through data
        for count in range(0, total_train_samples):

            # Print step
            print('Step', count + total_train_samples*epoch)

            # Reset iterators
            if(acdc_count == acdc_num_train_samples):
                acdc_count = 0
                
            if(bdd100k_count == bdd100k_num_train_samples):
                bdd100k_count = 0
            
            if(iddaw_count == iddaw_num_train_samples):
                iddaw_count = 0
            
            if(muses_count == muses_num_train_samples):
                muses_count = 0
            
            if(mapillary_count == mapillary_num_train_samples):
                muses_count = 0

            if(comma10k_count == comma10k_num_train_samples):
                comma10k_count = 0

            if(data_list_count == len(data_list)):
                data_list_count = 0

            # Read images, apply augmentation, run prediction, calculate
            # loss for iterated image from each dataset, and increment
            # dataset iterators

            if(data_list[data_list_count] == 'ACDC'):
                image, gt, class_weights = \
                        acdc_Dataset.getItemTrain(acdc_count)
                acdc_count += 1
            
            if(data_list[data_list_count] == 'BDD100K'):
                image, gt, class_weights = \
                    bdd100k_Dataset.getItemTrain(bdd100k_count)
                bdd100k_count += 1

            if(data_list[data_list_count] == 'IDDAW'):
                image, gt, class_weights = \
                    iddaw_Dataset.getItemTrain(iddaw_count)      
                iddaw_count += 1

            if(data_list[data_list_count] == 'MUSES'):
                image, gt, class_weights = \
                    muses_Dataset.getItemTrain(muses_count)
                muses_count += 1
            
            if(data_list[data_list_count] == 'MAPILLARY'):
                image, gt, class_weights = \
                    mapillary_Dataset.getItemTrain(mapillary_count)
                mapillary_count +=1
            
            if(data_list[data_list_count] == 'COMMA10K'):
                image, gt, class_weights = \
                    comma10k_Dataset.getItemTrain(comma10k_count)
                comma10k_count += 1
            
            # Augmenting Image
            aug_train = Augmentations(image, gt, True)
            image, augmented = aug_train.getAugmentedData()

            # Ground Truth with probabiliites for each class in separate channels
            gt_fused = np.stack((augmented[1], augmented[2], \
                    augmented[3]), axis=2)
                
            # Converting to tensor and loading
            image_tensor = load_image_tensor(image)
            gt_tensor = load_gt_tensor(gt_fused)
            class_weights_tensor = torch.tensor(class_weights).to(device)

            # Run model and calculate loss
            loss = nn.CrossEntropyLoss(weight=class_weights_tensor)
            prediction = model(image_tensor)
            calc_loss = loss(prediction, gt_tensor)
            
            # Gradient accumulation
            calc_loss.backward()

            # Simulating batch size of 3
            # Batch size of 3 gives good results in testing
            if((count+1) % 3 == 0):
                optimizer.step()
                optimizer.zero_grad()

            # Logging loss to Tensor Board every 100 steps
            if((count+1) % 100 == 0):
                print('Logging')
                writer.add_scalar("Loss/train", calc_loss,\
                    (count + total_train_samples*epoch))

            # Logging Image to Tensor Board every 1000 steps
            if((count+1) % 1000 == 0):   
                print('Saving Visualization')
                prediction = prediction.squeeze(0).cpu().detach()
                prediction = prediction.permute(1, 2, 0)
                vis_predict = visualize_result(prediction)
                label = augmented[0]
                fig, axs = plt.subplots(1,3)
                axs[0].imshow(image)
                axs[0].set_title('Image',fontweight ="bold") 
                axs[1].imshow(label)
                axs[1].set_title('Ground Truth',fontweight ="bold") 
                axs[2].imshow(vis_predict)
                axs[2].set_title('Prediction',fontweight ="bold") 
                writer.add_figure('predictions vs. actuals', \
                    fig, global_step=(count + total_train_samples*epoch))

            # Save model and run validation on entire validation 
            # dataset after 4000 steps
            if((count+1) % 4000 == 0):
                
                print('Saving model')
                save_path = model_save_path + 'iter_' + \
                    str(count + total_train_samples*epoch) \
                    + '_epoch_' +  str(epoch) + '_step_' + \
                    str(count) + '.pth'
                torch.save(model.state_dict(), save_path)

                print('Validating')

                # Setting model to evaluation mode
                model = model.eval()
                running_val_loss = 0

                # No gradient calculation
                with torch.no_grad():
                    
                    # ACDC
                    for val_count in range(0, acdc_num_val_samples):
                        image_val, gt_val, _ = \
                            acdc_Dataset.getItemVal(val_count)
                        
                        aug_val = Augmentations(image_val, gt_val, False)
                        image_val, augmented_val = aug_val.getAugmentedData()

                        gt_val_fused = np.stack((augmented_val[1], augmented_val[2], \
                        augmented_val[3]), axis=2)
        
                        image_val_tensor = load_image_tensor(image_val)
                        gt_val_tensor = load_gt_tensor(gt_val_fused)

                        loss_val = nn.CrossEntropyLoss()
                        prediction_val = model(image_val_tensor)
                        val_loss = loss(prediction_val, gt_val_tensor)
                        running_val_loss += val_loss.item()

                    # BDD100K
                    for val_count in range(0, bdd100k_num_val_samples):
                        image_val, gt_val, _ = \
                            bdd100k_Dataset.getItemVal(val_count)
                        
                        aug_val = Augmentations(image_val, gt_val, False)
                        image_val, augmented_val = aug_val.getAugmentedData()

                        gt_val_fused = np.stack((augmented_val[1], augmented_val[2], \
                        augmented_val[3]), axis=2)
        
                        image_val_tensor = load_image_tensor(image_val)
                        gt_val_tensor = load_gt_tensor(gt_val_fused)

                        loss_val = nn.CrossEntropyLoss()
                        prediction_val = model(image_val_tensor)
                        val_loss = loss(prediction_val, gt_val_tensor)
                        running_val_loss += val_loss.item()
                    
                    # MUSES
                    for val_count in range(0, muses_num_val_samples):
                        image_val, gt_val, _ = \
                            muses_Dataset.getItemVal(val_count)
                        
                        aug_val = Augmentations(image_val, gt_val, False)
                        image_val, augmented_val = aug_val.getAugmentedData()

                        gt_val_fused = np.stack((augmented_val[1], augmented_val[2], \
                        augmented_val[3]), axis=2)
        
                        image_val_tensor = load_image_tensor(image_val)
                        gt_val_tensor = load_gt_tensor(gt_val_fused)

                        loss_val = nn.CrossEntropyLoss()
                        prediction_val = model(image_val_tensor)
                        val_loss = loss(prediction_val, gt_val_tensor)
                        running_val_loss += val_loss.item()
                    
                    # IDDAW
                    for val_count in range(0, iddaw_num_val_samples):
                        image_val, gt_val, _ = \
                            iddaw_Dataset.getItemVal(val_count)
                        
                        aug_val = Augmentations(image_val, gt_val, False)
                        image_val, augmented_val = aug_val.getAugmentedData()

                        gt_val_fused = np.stack((augmented_val[1], augmented_val[2], \
                        augmented_val[3]), axis=2)
        
                        image_val_tensor = load_image_tensor(image_val)
                        gt_val_tensor = load_gt_tensor(gt_val_fused)

                        loss_val = nn.CrossEntropyLoss()
                        prediction_val = model(image_val_tensor)
                        val_loss = loss(prediction_val, gt_val_tensor)
                        running_val_loss += val_loss.item()

                    # MAPILLARY
                    for val_count in range(0, mapillary_num_val_samples):
                        image_val, gt_val, _ = \
                            mapillary_Dataset.getItemVal(val_count)
                        
                        aug_val = Augmentations(image_val, gt_val, False)
                        image_val, augmented_val = aug_val.getAugmentedData()

                        gt_val_fused = np.stack((augmented_val[1], augmented_val[2], \
                        augmented_val[3]), axis=2)
        
                        image_val_tensor = load_image_tensor(image_val)
                        gt_val_tensor = load_gt_tensor(gt_val_fused)

                        loss_val = nn.CrossEntropyLoss()
                        prediction_val = model(image_val_tensor)
                        val_loss = loss(prediction_val, gt_val_tensor)
                        running_val_loss += val_loss.item()

                    # COMMA10K
                    for val_count in range(0, comma10k_num_val_samples):
                        image_val, gt_val, _ = \
                            comma10k_Dataset.getItemVal(val_count)
                        
                        aug_val = Augmentations(image_val, gt_val, False)
                        image_val, augmented_val = aug_val.getAugmentedData()

                        gt_val_fused = np.stack((augmented_val[1], augmented_val[2], \
                        augmented_val[3]), axis=2)
        
                        image_val_tensor = load_image_tensor(image_val)
                        gt_val_tensor = load_gt_tensor(gt_val_fused)

                        loss_val = nn.CrossEntropyLoss()
                        prediction_val = model(image_val_tensor)
                        val_loss = loss(prediction_val, gt_val_tensor)
                        running_val_loss += val_loss.item()

                    # Calculating average loss of complete validation set
                    avg_val_loss = running_val_loss/total_val_samples
                    print('Average Validation loss:', avg_val_loss)
                    
                    # Logging average validation loss to TensorBoard
                    writer.add_scalar("Loss/val", avg_val_loss,\
                        (count + total_train_samples*epoch))

                # Resetting model back to training
                model = model.train()

            data_list_count += 1

    writer.flush()
    writer.close()
    print('Finished Training')

if __name__ == '__main__':
    main()
# %%
