#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import cv2
import sys
sys.path.append('..')
from inference.scene_seg_infer import SceneSegNetworkInfer
import numpy as np
from PIL import Image

def make_visualization(prediction):
    shape = prediction.shape
  
    row = shape[0]
    col = shape[1]
    vis_predict_object = np.zeros((row, col, 3), dtype = "uint8")

    background_objects_colour = (255, 93, 61)
    foreground_objects_colour = (145, 28, 255)

    # Extracting predicted classes and assigning to colourmap
    for x in range(row):
        for y in range(col):
            if(prediction[x,y].item() == 0):
                vis_predict_object[x,y] = background_objects_colour
            elif(prediction[x,y].item() == 1):
                vis_predict_object[x,y] = foreground_objects_colour
            elif(prediction[x,y].item() == 2):
                vis_predict_object[x,y] = background_objects_colour
               
    return vis_predict_object

def main():  

  # Saved model checkpoint path
  model_checkpoint_path = '/home/zain/Autoware/AutoSeg/Models/exports/SceneSeg/' \
    + 'run_1_batch_decay_Oct18_02-46-35/iter_140215_epoch_4_step_15999.pth'
  
  model = SceneSegNetworkInfer(checkpoint_path=model_checkpoint_path)
    
  # Create a VideoCapture object and read from input file
  # If the input is taken from the camera, pass 0 instead of the video file name.
  video_filepath = '/home/zain/Autoware/AutoSeg/videos/Driving_Scenes_Videos_2.mp4'
  cap = cv2.VideoCapture(video_filepath)

  # Output filepath
  output_filepath_obj = '/home/zain/Autoware/AutoSeg/videos/Driving_Scenes_Visualization_Objects_2.avi'

  
  writer_obj = cv2.VideoWriter(output_filepath_obj,
    cv2.VideoWriter_fourcc(*"MJPG"), 25,(1280,720))

  # Check if video catpure opened successfully
  if (cap.isOpened()== False): 
    print("Error opening video stream or file")
  
  # Transparency factor
  alpha = 0.5
  
  # Read until video is completed
  while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
 
      # Display the resulting frame
      image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      image_pil = Image.fromarray(image)
      image_pil = image_pil.resize((640, 320))
      
      prediction = model.inference(image_pil)
      vis_obj = make_visualization(prediction)
      
      vis_obj = cv2.resize(vis_obj, (1280, 720))
      image_vis_obj = cv2.addWeighted(vis_obj, alpha, frame, 1 - alpha, 0)
      cv2.imshow('Prediction Objects', image_vis_obj)
      cv2.waitKey(10)

      # Writing to video frame
      writer_obj.write(image_vis_obj)
  
  # When everything done, release the video capture and writer objects
  cap.release()
  writer_obj.release()

  # Closes all the frames
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()
# %%