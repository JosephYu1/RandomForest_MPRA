# author:       Joseph Yu
# file:         dataloader.py
# description:  The file is responsible for loading DragoNN-MPRA data, 
#               which is from Sharpr-MPRA, and extracts NumPy arrays from the hdf5 files.

import h5py
import numpy as np

path = "../data/"

filenames = {"test_file_name": str(path + "test.hdf5"), 
             "train_file_name": str(path + "train.hdf5"), 
             "validate_file_name": str(path + "valid.hdf5")}


def load_file():
    # Read from testing data set
    with h5py.File(testFilename, "r") as f1:
        # print("Keys: %s" % f1.keys())
            
        a_group_key = list(f1.keys())[0]
                
        # print(type(f1[a_group_key]))
        
        data = list(f1[a_group_key])
        
        ds_obj = f1[a_group_key]
        # ds_arr = f1[a_group_key][()]
            
        test_X = f1['X']['sequence'][()]
        test_Y = f1['Y']['output'][()]
    
    # Read from training data set
    with h5py.File(trainFilename, "r") as f2:
        # print("Keys: %s" % f2.keys())
                        
        a_group_key = list(f2.keys())[0]
                            
        # print(type(f2[a_group_key]))
        
        data = list(f2[a_group_key])
        
        ds_obj = f2[a_group_key]
        # ds_arr = f1[a_group_key][()]
            
        training_X = f2['X']['sequence'][()]
        training_Y = f2['Y']['output'][()]

    # Read from validation data set
    with h5py.File(validFilename, "r") as f3:
        # print("Keys: %s" % f3.keys())
                        
        a_group_key = list(f3.keys())[0]
                            
        # print(type(f3[a_group_key]))
        
        data = list(f3[a_group_key])
        
        ds_obj = f3[a_group_key]
        # ds_arr = f1[a_group_key][()]
            
        validation_X = f3['X']['sequence'][()]
        validation_Y = f3['Y']['output'][()]

    result = {"test_X": test_X, "test_Y": test_Y, 
              "training_X": training_X, "training_Y": training_Y, 
              "validation_X": validation_X, "validation_Y": validation_Y}

    return result 









