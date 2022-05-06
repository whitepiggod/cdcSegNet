# cdcSegNet

- This is a model for segmenting COVID-19 CT images.

## Requirements

- Python3
- Pytorch version >= 1.2.0.
- Some basic python packages, such as Numpy, Pandas, SimpleITK.

## Data Preparation

- Please put CT images and segmentation masks in the following directory: `./images/`, and organize the data as follows:
  ``` 
     ├── train
        ├── image
           ├── 1.jpg, 2.jpg, xxxx
        ├── mask
           ├── 1.png, 2.png, xxxx
     ├── test
        ├── image
           ├── case01
               ├── 1.jpg, 2.jpg, xxxx
        ├── mask
           ├── case01
               ├── 1.png, 2.png, xxxx
  ```

## Training & Testing

- Train the cdcSegNet:

  `python train.py`
  
  Weight values are saved in `./weight`

- Test the cdcSegNet:

  `python test.py`

  The results will be saved to `./Results`.

- Evaluate the segmentation maps:

  You can evaluate the segmentation maps using the tool in `./utils/evaluation.py`.

  `record_loss.txt` recorded various data in the experiment

## Acknowledgement

A collection of COVID-19 imaging-based AI research papers and datasets: https://github.com/HzFu/COVID19_imaging_AI_paper_list

<p align="center">
    <img src="images/paper_list.png" width="80%"/> <br />
</p>



