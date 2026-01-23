# Vehicle Damage Detection and Cost Estimation

This repository contains the prototype implementation for a Final Year Project (FYP)
focused on detecting vehicle damages from images and estimating repair costs using
machine learning and computer vision techniques.

## Project Structure

- data/
  - raw/: Original datasets (unchanged)
  - interim/: Cleaned and preprocessed data
  - processed/: Final datasets ready for model training
- notebooks/: Jupyter notebooks for data exploration and preprocessing
- src/: Source code for preprocessing and utilities
- models/: Trained models and checkpoints
- reports/: Figures, results, and documentation

## Status
Project setup and dataset preparation phase.


## Dataset Inspection – CarDD_COCO

The CarDD_COCO dataset was inspected prior to preprocessing to understand its structure and label distribution.

- **Dataset type:** Object Detection (COCO format)
- **Image splits:** Training, Validation, and Test sets
- **Number of damage classes:** 6

### Damage Classes
- dent  
- scratch  
- crack  
- glass shatter  
- lamp broken  
- tire flat  

### Training Annotation Distribution
- scratch: 2560  
- dent: 1806  
- crack: 651  
- lamp broken: 494  
- glass shatter: 475  
- tire flat: 225  

**Observation:**  
The dataset shows class imbalance, with *scratch* and *dent* being dominant classes, while *tire flat* is underrepresented. This will be considered during preprocessing and model training.



## Dataset Inspection – archive (1) (Binary Classification)

- Task: Damage vs Whole (No Damage)
- Classes: 00-damage, 01-whole

### Image distribution
Train:
- 00-damage: 920
- 01-whole : 920
Total train: 1840

Validation:
- 00-damage: 230
- 01-whole : 230
Total validation: 460

Overall total: 2300
Note: Dataset is perfectly balanced across both classes.

## Dataset Inspection – archive (2) (Damage Severity Classification)

- Task: Multi-class classification (damage severity)
- Classes: 01-minor, 02-moderate, 03-severe

### Image distribution

Train:
- Minor    : 452
- Moderate : 463
- Severe   : 468
Total train: 1383

Validation:
- Minor    : 82
- Moderate : 75
- Severe   : 91
Total validation: 248

Overall total: 1631 images
Note: Dataset shows near-balanced class distribution.



### Note: archive (3) was found to be identical to archive (1) in terms of class structure and image distribution, and was therefore excluded from further preprocessing and experiments to avoid data duplication.



### Dataset Inspection – archive (4)

archive (4) contains VIA JSON annotations with polygon-based damage regions.
However, the annotations do not include region-level class labels
(i.e., `region_attributes` are missing).

As a result, while the dataset provides damage localization information,
it cannot be directly used for damage type classification or cost estimation.
The dataset is therefore excluded from core training experiments and retained
only for reference and qualitative analysis.



