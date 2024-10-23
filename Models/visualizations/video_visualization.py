#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import cv2


def main():  

  # Create a VideoCapture object and read from input file
  # If the input is taken from the camera, pass 0 instead of the video file name.
  video_filepath = '/home/zain/Autoware/AutoSeg/videos/Driving_Scenes_Videos.mp4'
  cap = cv2.VideoCapture(video_filepath)

  # Check if video catpure opened successfully
  if (cap.isOpened()== False): 
    print("Error opening video stream or file")
  
  # Read until video is completed
  while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
  
      # Display the resulting frame
      cv2.imshow('Frame',frame)
  
      # Press Q on keyboard to  exit
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break
  
    # Break the loop
    else: 
      break
  
  # When everything done, release the video capture object
  cap.release()
  
  # Closes all the frames
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()
# %%