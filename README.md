

## How to start training?

We provide a simple codebase here:


```
git clone https://github.com/anettv05/NTIRE2025_EventDeblur_challenge_Group10
cd NTIRE2025_EventDeblur_challenge
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```
Single GPU training:
```
python ./basicsr/train.py -opt options/train/HighREV/EFNet_HighREV_Deblur.yml
```
Multi-GPU training:
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/HighREV/EFNet_HighREV_Deblur.yml --launcher pytorch
```

''' 
I have actually run the code in kaggle so i will be giving the link of the kaggle for reference.
'''
## How to start testing?
Example:
```
python3 basicsr/test.py -opt options/test/HighREV/EFNet_HighREV_Deblur.yml
```

Calculating flops:
set ``print_flops`` to ``true`` and set your input shapes in ``flops_input_shape`` in the test yml file.
Example:
```
print_flops: true 
flops_input_shape: 
  - [3, 256, 256] # image shape
  - [6, 256, 256] # event shape
```

Be sure to modify the path configurations in yml file.



## Develop your own model
We recommand to used basicsr (already used here, [tutorial](https://github.com/XPixelGroup/BasicSR)) for developing. It is easy to change the models in `./basicsr/models`.



## Use your own codes
The dataset-related code is in `./basicsr/data/npz_image_dataset.py`. If you are not using the code from this repository, please integrate it into your own code for convenience.


## Others
The aim is to obtain a network design / solution that fusing events and images and produce high quality results with the best performance (i.e., PSNR). We suggest using HighREV dataset for training.

For the sake of fairness, please do not train your model with the (HighREV) validation GT images.


The top ranked participants will be awarded and invited to follow the CVPR submission guide for workshops to describe their solution and to submit to the associated NTIRE workshop at CVPR 2025.
   
<!-- ## How to add your model to this baseline?
1. Register your team in the [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1XVa8LIaAURYpPvMf7i-_Yqlzh-JsboG0hvcnp-oI9rs/edit?usp=sharing) and get your team ID.
2. Put your the code of your model in `./models/[Your_Team_ID]_[Your_Model_Name].py`
   - Please add **only one** file in the folder `./models`. **Please do not add other submodules**.
   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02 
3. Put the pretrained model in `./model_zoo/[Your_Team_ID]_[Your_Model_Name].[pth or pt or ckpt]`
   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02  
4. Add your model to the model loader `./test_demo/select_model` as follows:
    ```python
        elif model_id == [Your_Team_ID]:
            # define your model and load the checkpoint
    ```
   - Note: Please set the correct data_range, either 255.0 or 1.0
5. Send us the command to download your code, e.g, 
   - `git clone [Your repository link]`
   - We will do the following steps to add your code and model checkpoint to the repository.
This repository shows how to add noise to synthesize the noisy image. It also shows how you can save an image. -->



## Citations

```
@inproceedings{sun2023event,
  title={Event-based frame interpolation with ad-hoc deblurring},
  author={Sun, Lei and Sakaridis, Christos and Liang, Jingyun and Sun, Peng and Cao, Jiezhang and Zhang, Kai and Jiang, Qi and Wang, Kaiwei and Van Gool, Luc},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18043--18052},
  year={2023}
}

@inproceedings{sun2022event,
  title={Event-Based Fusion for Motion Deblurring with Cross-modal Attention},
  author={Sun, Lei and Sakaridis, Christos and Liang, Jingyun and Jiang, Qi and Yang, Kailun and Sun, Peng and Ye, Yaozu and Wang, Kaiwei and Gool, Luc Van},
  booktitle={European Conference on Computer Vision},
  pages={412--428},
  year={2022},
  organization={Springer}
}
```
