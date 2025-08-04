


import numpy as np
from efficientGP.data.load_DOE_module import load_DOE_csv
from efficientGP.transforms.mesh.utils import retrieve_data
from efficientGP.data.load_design_module import load_design_csv





def load_design_dataset(    
    DOE_csv_path = 'data/Pump_data/DOE_data.csv',
    design_csv_path = 'data/Pump_data/design_variable.csv'
):

    metadata_keys, metadata_values = load_DOE_csv(DOE_csv_path)


    torque_values = retrieve_data(metadata_values, metadata_keys, 'Torque [N m]')
    absolute_torque_values = np.abs(torque_values)  # Ensure torque values are positive

    eff_values = retrieve_data(metadata_values, metadata_keys, 'Eff [%]')        

    design_keys, design_variable = load_design_csv(design_csv_path)

    design_variable = design_variable[:, 1:]

    return design_variable, absolute_torque_values, eff_values



def load_front_train_val_data(
    num_train = 900,
    num_test = 100
):
    design_variable, torque_values, eff_values = load_design_dataset()

    train_design_variable = design_variable[:num_train]
    test_design_variable = design_variable[-num_test:]

    train_torque_values = torque_values[:num_train]
    test_torque_values = torque_values[-num_test:]

    train_eff_values = eff_values[:num_train]
    test_eff_values = eff_values[-num_test:]
    
    return train_design_variable, test_design_variable, train_torque_values, test_torque_values, train_eff_values, test_eff_values


if __name__ == "__main__":
    design_variable, torque_values, eff_values = load_design_dataset()
    train_design_variable, test_design_variable, train_torque_values,\
          test_torque_values, train_eff_values, test_eff_values \
        = load_front_train_val_data(
        num_train=900,
        num_test=100  
        )
    print('done')

