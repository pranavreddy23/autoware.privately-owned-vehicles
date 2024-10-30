#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import torch
import time
import numpy as np
from pytorch_model_summary import summary
import sys
sys.path.append('..')
from model_components.scene_seg_network import SceneSegNetwork


def benchmark(model, input_data, dtype='fp32', nwarmup=50, nruns=1000):

    if dtype=='fp16':
        input_data = input_data.half()
        model = model.half()

    print("Warm up ...")

    with torch.no_grad():
        for _ in range(nwarmup):
            _ = model(input_data)

    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []

    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            output = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)

            if i%100==0:
                print('Iteration %d/%d, avg batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))

    print("Input shape:", input_data.size())
    print("Output shape:", output.shape)
    print('Average batch time: %.2f ms'%(np.mean(timings)*1000))

def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} for inference')

    # Instantiating Model and setting to evaluation mode
    model = SceneSegNetwork()
    print(summary(SceneSegNetwork(), torch.zeros((1, 3, 320, 640)), show_input=True))
    model = model.to(device)
    model = model.eval()

    # Fake input data
    input_shape=(1, 3, 320, 640)
    input_data = torch.randn(input_shape)
    input_data = input_data.to(device)

    # Run speed benchmark
    benchmark(model, input_data, 'fp32', 50, 1000)

if __name__ == '__main__':
  main()
# %%