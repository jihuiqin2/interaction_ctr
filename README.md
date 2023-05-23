# File

1. model
2. layer
3. utils
4. datasets
5. data
6. config
7. checkpoints

# Requirements

1. python3.6
2. pytorch1.4

# Dataset

1. criteo
   https://criteostorage.blob.core.windows.net/criteo-research-datasets/kaggle-display-advertising-challenge-dataset.tar.gz
2. avazu
   https://www.kaggle.com/c/avazu-ctr-prediction

# Usage

- Convert files from other formats to csv. demo/txt_to_csv.py or data_to_csv.
- You need to run the file demo/csv_to_h5_avazu.py or demo/csv_to_h5_criteo.py to preprocess the data.
- Then you can run the file demo/common_h5_demo.py to train the model.
- You can also change other parameters in config/model_config/.