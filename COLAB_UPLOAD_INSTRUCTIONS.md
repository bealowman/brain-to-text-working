# Google Colab Upload Instructions

This document lists all the files you need to upload to Google Drive for the Colab notebook to work.

## Required Files to Upload

Upload these files to a folder in your Google Drive (e.g., `MyDrive/brain-to-text/`):

### Python Files (from `model_training/` directory)
1. `train_model.py`
2. `rnn_trainer.py`
3. `rnn_model.py`
4. `dataset.py`
5. `data_augmentations.py`
6. `evaluate_model.py`
7. `evaluate_model_helpers.py`

### Configuration Files
8. `rnn_args.yaml` OR `rnn_diphone_args.yaml` (or both if you want to switch between them)

### Data Files
9. `t15_copyTaskData_description.csv` (from `data/` directory)
10. Your HDF5 data files:
    - Upload the entire `hdf5_data_final/` directory structure
    - Or ensure the session directories (e.g., `t15.2023.08.11/`, `t15.2023.08.13/`, etc.) are accessible
    - Each session directory should contain `data_train.hdf5`, `data_val.hdf5`, and optionally `data_test.hdf5`

## Recommended Folder Structure on Google Drive

```
MyDrive/
└── brain-to-text/
    ├── train_model.py
    ├── rnn_trainer.py
    ├── rnn_model.py
    ├── dataset.py
    ├── data_augmentations.py
    ├── evaluate_model.py
    ├── evaluate_model_helpers.py
    ├── rnn_args.yaml (or rnn_diphone_args.yaml)
    ├── data/
    │   ├── t15_copyTaskData_description.csv
    │   └── hdf5_data_final/
    │       ├── t15.2023.08.11/
    │       │   ├── data_train.hdf5
    │       │   ├── data_val.hdf5
    │       │   └── data_test.hdf5
    │       ├── t15.2023.08.13/
    │       │   └── ...
    │       └── ... (other session directories)
    └── trained_models/  (will be created automatically during training)
        └── ...
```

## Notes

- The notebook will automatically create a `trained_models/` directory in your Drive folder to save checkpoints
- Make sure the session names in your YAML config file match the actual session directory names in your data folder
- If you're using a different folder name than `brain-to-text`, update the `DRIVE_PATH` variable in the notebook's second cell


