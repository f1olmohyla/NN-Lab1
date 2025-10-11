**Repository for lab1 in NN's course**

Theme: Detection and classification of military aircrafts

Dataset: https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset?resource=download

Files: 
 - download_aircraft_dataset.py - loads and unzips the archive with original dataset
 - dataset_info.py - prints shallow EDA of the dataset
 - dataset_preprocessing.py - implements preprocessing pipeline for aircraft detection dataset

## Preprocessing Methods

**Image Normalization (ImageNormalizer):** Standardizes image dimensions to 640x640 and applies ImageNet normalization to ensure consistent input format for neural networks across the wide range of original image sizes (121×119 to 8508×7360).

**Data Augmentation (DataAugmenter):** Applies horizontal flips, rotations, brightness and contrast adjustments to artificially increase dataset diversity and improve model generalization, especially critical given the severe class imbalance (F16: 1934 samples vs WZ9: 15 samples).

**Class Balancing (ClassBalancer):** Implements weighted sampling and minority class oversampling to address the significant class distribution imbalance across 95 aircraft types, preventing the model from being biased toward the most frequent classes.

**Bounding Box Validation (BoundingBoxValidator):** Ensures bounding box coordinates are valid, properly clipped to image boundaries, and filters out boxes with insufficient area to maintain data quality during object detection training.

**Dataset Preprocessing Pipeline (DatasetPreprocessor):** Combines all preprocessing steps into a unified workflow that scales bounding boxes proportionally with image resizing, maintaining spatial relationships between aircraft objects and their annotations.

