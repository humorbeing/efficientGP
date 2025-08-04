

import joblib


def save_gp(gp, model_filename):    
    joblib.dump(gp, model_filename)


def load_gp(model_filename):
    gp = joblib.load(model_filename)
    return gp

if __name__ == "__main__":
    # Example usage
    model_filename = 'model_weights/gp_test.pkl'

    from efficientGP.models.gp_design.gp_model_from_config_module import get_gp_model
    config_path = 'configs/from_gp_dnn_paper.yaml'
    gp = get_gp_model(config_path)
    
    # Save the GP model
    save_gp(gp, model_filename) 

    # Load the GP model
    loaded_gp = load_gp(model_filename)
    
    print("GP model saved and loaded successfully.")
    print(loaded_gp)

    print('done')