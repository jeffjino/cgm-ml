This code is written with the help of `https://github.com/leoxiaobin/deep-high-resolution-net.pytorch`

List of registered model in azure-ml workspace:

1. [pose_hrnet_w48_384x288](https://ml.azure.com/model/pose_hrnet_w48_384x288:1/details?wsid=/subscriptions/9b5bbfae-d5d1-4aae-a2ca-75159c0c887d/resourceGroups/cgm-ml-prod-we-rg/providers/Microsoft.MachineLearningServices/workspaces/cgm-ml-prod-we-azml&tid=3a27c573-ec1a-4734-9cd3-3208af51794b)
2. [pose_hrnet_w32_384x288](https://ml.azure.com/model/pose_hrnet_w32_384x288:1/details?wsid=/subscriptions/9b5bbfae-d5d1-4aae-a2ca-75159c0c887d/resourceGroups/cgm-ml-prod-we-rg/providers/Microsoft.MachineLearningServices/workspaces/cgm-ml-prod-we-azml&tid=3a27c573-ec1a-4734-9cd3-3208af51794b)

1.Update NUM_CHANNELS and MODEL_FILE in the inference file.

# Steps to use/run the prose prediction on sample scans

1. For pose_hrnet_w48_384x288

- NUM_CHANNELS
  - 48
  - 96
- NUM_CHANNELS:
  - 48
  - 96
  - 192
- NUM_CHANNELS:
  - 48
  - 96
  - 192
  - 384
- TEST
  - MODEL_FILE: 'src/models/HRNET/weights/pytorch/pose_coco/pose_hrnet_w48_384x288.pth'
- MODAL
  - NAME: pose_hrnet_w48_384x288

2. For pose_hrnet_w32_384x288

- NUM_CHANNELS
  - 32
  - 64
- NUM_CHANNELS:
  - 32
  - 64
  - 128
- NUM_CHANNELS:
  - 32
  - 64
  - 128
  - 256
- TEST
  - MODEL_FILE: 'src/models/HRNET/weights/pytorch/pose_coco/pose_hrnet_w32_384x288.pth'
- MODAL
  - NAME: pose_hrnet_w32_384x288

2. Download the registered pretrained pose detection model from the azure ml workspace.
   `python 1_download_model.py`

3. Make sure you have sample scans at `data/anon_rgb_training/scans` this path
4. Run below script to generate csv file with all the results and time taken to execute the pose prediction.
   `python 2_inference_model.py`
