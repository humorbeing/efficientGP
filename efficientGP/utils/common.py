import os

# Dynamic Path (DP) Module
def DP(relative_path, current_path, num_subdirs=1):
    current_file_path = os.path.dirname(current_path)    
    relative_folder = os.path.join(current_file_path, *['..'] * num_subdirs)
    return os.path.join(relative_folder, relative_path)

if __name__ == "__main__":
    from efficientGP.data.load_design_module import load_design_csv


    target_file = 'data/Pump_data/design_variable.csv'
    from efficientGP.utils.common import DP
    target_file = DP(target_file, __file__, 2)

    
    design_keys, design_variable = load_design_csv(target_file)
    print(design_keys)
    print(design_variable[0])
