
#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import torch
from torchvision import transforms
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
sys.path.append('..')
from model_components.scene_seg_network import SceneSegNetwork
from data_utils.load_data import LoadData
from data_utils.augmentations import Augmentations
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

class trainSceneSeg():
    def __init__(self):

        self.image = 0
        self.image_val = 0
        self.gt = 0
        self.gt_val = 0
        self.class_weights = 0
        self.gt_fused = 0
        self.gt_val_fused = 0
        self.augmented = 0
        self.augmented_val = 0
        self.image_tensor = 0
        self.image_val_tensor = 0
        self.gt_tensor = 0
        self.gt_val_tensor = 0
        self.class_weights_tensor = 0
        self.loss = 0
        self.prediction = 0
        self.calc_loss = 0
        self.prediction_vis = 0

        # Checking devices (GPU vs CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device} for inference')
        
        # Instantiate model 
        self.model = SceneSegNetwork().to(self.device)

        # TensorBoard
        self.writer = SummaryWriter()

        # Learning rate and optimizer
        self.learning_rate = 0.0001
        self.optimizer = optim.AdamW(self.model.parameters(), self.learning_rate)

        # Loaders
        self.image_loader = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

        self.gt_loader = transforms.Compose(
            [transforms.ToTensor()]
        )

    # Logging Training Loss
    def log_loss(self, log_count):
        print('Logging Training Loss', log_count, self.get_loss())
        self.writer.add_scalar("Loss/train",self.get_loss(), (log_count))

    # Logging Validation mIoU Score
    def log_IoU(self, mIoU_full, mIoU_bg, mIoU_fg, mIoU_rd, log_count):
        print('Logging Validation')
        self.writer.add_scalars("Val/IoU_Classes",{
            'mIoU_bg': mIoU_bg,
            'mIoU_fg': mIoU_fg,
            'mIoU_rd': mIoU_rd
        }, (log_count))
        
        self.writer.add_scalar("Val/IoU", mIoU_full, (log_count))
        
    # Assign input variables
    def set_data(self, image, gt, class_weights):
        self.image = image
        self.gt = gt
        self.class_weights = class_weights

    def set_val_data(self, image_val, gt_val):
        self.image_val = image_val
        self.gt_val = gt_val

    # Image agumentations
    def apply_augmentations(self, is_train):

        if(is_train):
            # Augmenting Image
            aug_train = Augmentations(self.image, self.gt, True)
            self.image, self.augmented = aug_train.getAugmentedData()
            
            # Ground Truth with probabiliites for each class in separate channels
            self.gt_fused = np.stack((self.augmented[1], self.augmented[2], \
                        self.augmented[3]), axis=2)
        else:
            # Augmenting Image
            aug_val = Augmentations(self.image_val, self.gt_val, False)
            self.image_val, self.augmented_val = aug_val.getAugmentedData()

            # Ground Truth with probabiliites for each class in separate channels
            self.gt_val_fused = np.stack((self.augmented_val[1], self.augmented_val[2], \
                        self.augmented_val[3]), axis=2)
    
    # Load Data
    def load_data(self, is_train):
        self.load_image_tensor(is_train)
        self.load_gt_tensor(is_train)
        
        if(is_train):
            self.class_weights_tensor = \
            torch.tensor(self.class_weights).to(self.device)

    # Run Model
    def run_model(self):     
        self.loss = nn.CrossEntropyLoss(weight=self.class_weights_tensor)
        self.prediction = self.model(self.image_tensor)
        self.calc_loss = self.loss(self.prediction, self.gt_tensor)

    # Loss Backward Pass
    def loss_backward(self): 
        self.calc_loss.backward()

    # Get loss value
    def get_loss(self):
        return self.calc_loss.item()

    # Run Optimizer
    def run_optimizer(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    # Set train mode
    def set_train_mode(self):
        self.model = self.model.train()

    # Set evaluation mode
    def set_eval_mode(self):
        self.model = self.model.eval()

    # Save predicted visualization
    def save_visualization(self, log_count):
        print('Saving Visualization')
        self.prediction_vis = self.prediction.squeeze(0).cpu().detach()
        self.prediction_vis = self.prediction_vis.permute(1, 2, 0)
                
        vis_predict = self.make_visualization()
        label = self.augmented[0]
        fig, axs = plt.subplots(1,3)
        axs[0].imshow(self.image)
        axs[0].set_title('Image',fontweight ="bold") 
        axs[1].imshow(label)
        axs[1].set_title('Ground Truth',fontweight ="bold") 
        axs[2].imshow(vis_predict)
        axs[2].set_title('Prediction',fontweight ="bold") 
        self.writer.add_figure('predictions vs. actuals', \
        fig, global_step=(log_count))

    # Load Image as Tensor
    def load_image_tensor(self, is_train):

        if(is_train):
            image_tensor = self.image_loader(self.image)
            image_tensor = image_tensor.unsqueeze(0)
            self.image_tensor = image_tensor.to(self.device)
        else:
            image_val_tensor = self.image_loader(self.image_val)
            image_val_tensor = image_val_tensor.unsqueeze(0)
            self.image_val_tensor = image_val_tensor.to(self.device)

    # Load Ground Truth as Tensor
    def load_gt_tensor(self, is_train):

        if(is_train):
            gt_tensor = self.gt_loader(self.gt_fused)
            gt_tensor = gt_tensor.unsqueeze(0)
            self.gt_tensor = gt_tensor.to(self.device)
        else:
            gt_val_tensor = self.gt_loader(self.gt_val_fused)
            gt_val_tensor = gt_val_tensor.unsqueeze(0)
            self.gt_val_tensor = gt_val_tensor.to(self.device)
    
    # Zero Gradient
    def zero_grad(self):
        self.optimizer.zero_grad()

    # Save Model
    def save_model(self, model_save_path):
        print('Saving model')
        torch.save(self.model.state_dict(), model_save_path)
    
    # Calculate IoU score for validation
    def calc_IoU_val(self):
        output_val = self.model(self.image_val_tensor)
        output_val = output_val.squeeze(0).cpu().detach()
        output_val = output_val.permute(1, 2, 0)
        output_val = output_val.numpy()

        for x in range(0, output_val.shape[0]):
            for y in range(0, output_val.shape[1]):
                
                bg_prob = output_val[x,y,0]
                fg_prob = output_val[x,y,1]
                rd_prob = output_val[x,y,2]

                if(bg_prob >= fg_prob and bg_prob >= rd_prob):
                    output_val[x,y,0] = 1
                    output_val[x,y,1] = 0
                    output_val[x,y,2] = 0
                elif(fg_prob >= bg_prob and fg_prob >= rd_prob):
                    output_val[x,y,0] = 0
                    output_val[x,y,1] = 1
                    output_val[x,y,2] = 0
                elif(rd_prob >= bg_prob and rd_prob >= fg_prob):
                    output_val[x,y,0] = 0
                    output_val[x,y,1] = 0
                    output_val[x,y,2] = 1
       
        iou_score_full = self.IoU(output_val, self.gt_val_fused)
        iou_score_bg = self.IoU(output_val[:,:,0], self.gt_val_fused[:,:,0])
        iou_score_fg = self.IoU(output_val[:,:,1], self.gt_val_fused[:,:,1])
        iou_score_rd = self.IoU(output_val[:,:,2], self.gt_val_fused[:,:,2])

        return iou_score_full, iou_score_bg, iou_score_fg, iou_score_rd

    # IoU calculation
    def IoU(self, output, label):
        intersection = np.logical_and(label, output)
        union = np.logical_or(label, output)
        iou_score = np.sum(intersection) / (np.sum(union) + 1)
        return iou_score
    
    def validate(self, image_val, gt_val):

        # Set Data
        self.set_val_data(image_val, gt_val)

        # Augmenting Image
        self.apply_augmentations(is_train=False)

        # Converting to tensor and loading
        self.load_data(is_train=False)

        # Calculate IoU score
        iou_score_full, iou_score_bg, iou_score_fg, iou_score_rd \
            = self.calc_IoU_val()
        
        return iou_score_full, iou_score_bg, iou_score_fg, iou_score_rd


    # Visualize predicted result
    def make_visualization(self):
        shape = self.prediction_vis.shape
        _, output = torch.max(self.prediction_vis, dim=2)

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
    
    def cleanup(self):
        self.writer.flush()
        self.writer.close()
        print('Finished Training')

def main():

    # Root path
    root = '/home/zain/Autoware/AutoSeg/training_data/Scene_Seg/'

    # Model save path
    model_save_root_path = '/home/zain/Autoware/AutoSeg/Models/exports/SceneSeg/'

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
    #bdd100k_Dataset = LoadData(bdd100k_labels_fileapath, bdd100k_images_fileapath, 'BDD100K')
    #bdd100k_num_train_samples, bdd100k_num_val_samples = bdd100k_Dataset.getItemCount()

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

    # Total number of training samples
    total_train_samples = acdc_num_train_samples + \
    + iddaw_num_train_samples + muses_num_train_samples \
    + mapillary_num_train_samples + comma10k_num_train_samples
    print(total_train_samples, ': total training samples')

    # Total number of validation samples
    total_val_samples = acdc_num_val_samples + \
    + iddaw_num_val_samples + muses_num_val_samples \
    + mapillary_num_val_samples + comma10k_num_val_samples
    print(total_val_samples, ': total validation samples')

    # Trainer Class
    trainer = trainSceneSeg()
    trainer.zero_grad()
    
    # Total training epochs
    num_epochs = 75

    # Epochs
    for epoch in range(0, num_epochs):

        # Iterators for datasets
        acdc_count = 0
        bdd100k_count = 0
        iddaw_count = 0
        muses_count = 0
        comma10k_count = 0
        mapillary_count = 0

        is_acdc_complete = False
        is_bdd100k_complete = False
        is_iddaw_complete = False
        is_muses_complete = False
        is_mapillary_complete = False
        is_comma10k_complete = False

        data_list = []
        data_list.append('ACDC')
        #data_list.append('BDD100K')
        data_list.append('IDDAW')
        data_list.append('MUSES')
        data_list.append('MAPILLARY')
        data_list.append('COMMA10K')
        random.shuffle(data_list)
        data_list_count = 0

        # Loop through data
        for count in range(0, total_train_samples):

            log_count = count + total_train_samples*epoch

            # Reset iterators
            if(acdc_count == acdc_num_train_samples and \
               is_acdc_complete == False):
                is_acdc_complete =  True
                data_list.remove("ACDC")
            
            #if(bdd100k_count == bdd100k_num_train_samples and \
            #  is_bddd100k_complete == True):
            #    is_bdd100k_complete = True
            #    data_list.remove("BDD100K")
            
            if(iddaw_count == iddaw_num_train_samples and \
               is_iddaw_complete == False):
                is_iddaw_complete = True
                data_list.remove("IDDAW")
            
            if(muses_count == muses_num_train_samples and \
                is_muses_complete == False):
                is_muses_complete = True
                data_list.remove('MUSES')
            
            if(mapillary_count == mapillary_num_train_samples and \
               is_mapillary_complete == False):
                is_mapillary_complete = True
                data_list.remove('MAPILLARY')

            if(comma10k_count == comma10k_num_train_samples and \
               is_comma10k_complete == False):
                is_comma10k_complete = True
                data_list.remove('COMMA10K')

            if(data_list_count >= len(data_list)):
                data_list_count = 0

            # Read images, apply augmentation, run prediction, calculate
            # loss for iterated image from each dataset, and increment
            # dataset iterators

            if(data_list[data_list_count] == 'ACDC' and \
                is_acdc_complete == False):
                image, gt, class_weights = \
                        acdc_Dataset.getItemTrain(acdc_count)
                acdc_count += 1
            
            #if(data_list[data_list_count] == 'BDD100K' and \
            #   is_bdd100k_complete == False):
            #    image, gt, class_weights = \
            #        bdd100k_Dataset.getItemTrain(bdd100k_count)
            #    bdd100k_count += 1

            if(data_list[data_list_count] == 'IDDAW' and \
               is_iddaw_complete == False):
                image, gt, class_weights = \
                    iddaw_Dataset.getItemTrain(iddaw_count)      
                iddaw_count += 1

            if(data_list[data_list_count] == 'MUSES' and \
               is_muses_complete == False):
                image, gt, class_weights = \
                    muses_Dataset.getItemTrain(muses_count)
                muses_count += 1
            
            if(data_list[data_list_count] == 'MAPILLARY' and \
               is_mapillary_complete == False):
                image, gt, class_weights = \
                    mapillary_Dataset.getItemTrain(mapillary_count)
                mapillary_count +=1
            
            if(data_list[data_list_count] == 'COMMA10K' and \
                is_comma10k_complete == False):
                image, gt, class_weights = \
                    comma10k_Dataset.getItemTrain(comma10k_count)
                comma10k_count += 1
            
            # Assign Data
            trainer.set_data(image, gt, class_weights)
            
            # Augmenting Image
            trainer.apply_augmentations(is_train=True)

            # Converting to tensor and loading
            trainer.load_data(is_train=True)

            # Run model and calculate loss
            trainer.run_model()
            
            # Gradient accumulation
            trainer.loss_backward()

            # Simulating batch size of 3 for optimizer
            if((count+1) % 3 == 0):
                trainer.run_optimizer()

            # Logging loss to Tensor Board every 250 steps
            if((count+1) % 250 == 0):
                trainer.log_loss(log_count)
            
            # Logging Image to Tensor Board every 1000 steps
            if((count+1) % 1000 == 0):  
                trainer.save_visualization(log_count)
            
            # Save model and run validation on entire validation 
            # dataset after 8000 steps
            if((count+1) % 8000 == 0):
                
                # Save Model
                model_save_path = model_save_root_path + 'iter_' + \
                    str(count + total_train_samples*epoch) \
                    + '_epoch_' +  str(epoch) + '_step_' + \
                    str(count) + '.pth'
                
                trainer.save_model(model_save_path)
                
                # Validate
                print('Validating')

                # Setting model to evaluation mode
                trainer.set_eval_mode()

                running_IoU_full = 0
                running_IoU_bg = 0
                running_IoU_fg = 0
                running_IoU_rd = 0

                # No gradient calculation
                with torch.no_grad():

                    # ACDC
                    for val_count in range(80, 90):
                        image_val, gt_val, _ = \
                            acdc_Dataset.getItemVal(val_count)

                        # Run Validation and calculate IoU Score
                        IoU_score_full, IoU_score_bg, IoU_score_fg, IoU_score_rd = \
                            trainer.validate(image_val, gt_val)

                        running_IoU_full += IoU_score_full
                        running_IoU_bg += IoU_score_bg
                        running_IoU_fg += IoU_score_fg
                        running_IoU_rd += IoU_score_rd
                    '''
                    # BDD100K
                   
                    for val_count in range(0, bdd100k_num_val_samples):
                        image_val, gt_val, _ = \
                            bdd100k_Dataset.getItemVal(val_count)
                        
                        # Run Validation and calculate IoU Score
                        IoU_score_full, IoU_score_bg, IoU_score_fg, IoU_score_rd = \
                            trainer.validate(image_val, gt_val)

                        running_IoU_full += IoU_score_full
                        running_IoU_bg += IoU_score_bg
                        running_IoU_fg += IoU_score_fg
                        running_IoU_rd += IoU_score_rd
                    '''
                    # MUSES
                    for val_count in range(0, muses_num_val_samples):
                        image_val, gt_val, _ = \
                            muses_Dataset.getItemVal(val_count)
                        
                        # Run Validation and calculate IoU Score
                        IoU_score_full, IoU_score_bg, IoU_score_fg, IoU_score_rd = \
                            trainer.validate(image_val, gt_val)

                        running_IoU_full += IoU_score_full
                        running_IoU_bg += IoU_score_bg
                        running_IoU_fg += IoU_score_fg
                        running_IoU_rd += IoU_score_rd
                    
                    # IDDAW
                    for val_count in range(0, iddaw_num_val_samples):
                        image_val, gt_val, _ = \
                            iddaw_Dataset.getItemVal(val_count)
                        
                        # Run Validation and calculate IoU Score
                        IoU_score_full, IoU_score_bg, IoU_score_fg, IoU_score_rd = \
                            trainer.validate(image_val, gt_val)

                        running_IoU_full += IoU_score_full
                        running_IoU_bg += IoU_score_bg
                        running_IoU_fg += IoU_score_fg
                        running_IoU_rd += IoU_score_rd

                    # MAPILLARY
                    for val_count in range(0, mapillary_num_val_samples):
                        image_val, gt_val, _ = \
                            mapillary_Dataset.getItemVal(val_count)
                        
                         # Run Validation and calculate IoU Score
                        IoU_score_full, IoU_score_bg, IoU_score_fg, IoU_score_rd = \
                            trainer.validate(image_val, gt_val)

                        running_IoU_full += IoU_score_full
                        running_IoU_bg += IoU_score_bg
                        running_IoU_fg += IoU_score_fg
                        running_IoU_rd += IoU_score_rd

                    # COMMA10K
                    for val_count in range(0, comma10k_num_val_samples):
                        image_val, gt_val, _ = \
                            comma10k_Dataset.getItemVal(val_count)
                        
                        # Run Validation and calculate IoU Score
                        IoU_score_full, IoU_score_bg, IoU_score_fg, IoU_score_rd = \
                            trainer.validate(image_val, gt_val)

                        running_IoU_full += IoU_score_full
                        running_IoU_bg += IoU_score_bg
                        running_IoU_fg += IoU_score_fg
                        running_IoU_rd += IoU_score_rd
                    
                    # Calculating average loss of complete validation set
                    mIoU_full = running_IoU_full/total_val_samples
                    mIoU_bg = running_IoU_bg/total_val_samples
                    mIoU_fg = running_IoU_fg/total_val_samples
                    mIoU_rd = running_IoU_rd/total_val_samples
                    
                    # Logging average validation loss to TensorBoard
                    trainer.log_IoU(mIoU_full, mIoU_bg, mIoU_fg, mIoU_rd, log_count)

                # Resetting model back to training
                trainer.set_train_mode()
                
            data_list_count += 1

    trainer.cleanup()


if __name__ == '__main__':
    main()
# %%
