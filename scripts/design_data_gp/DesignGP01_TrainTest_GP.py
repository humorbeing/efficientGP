from omegaconf import OmegaConf

exp_config_path = 'configs/design_gp_experiments/front_sample_100.yaml'
exp_config = OmegaConf.load(exp_config_path)

from efficientGP.data.design_data.load_data_module import load_front_train_val_data

if exp_config.is_random_sample:
    pass
else:
    num_train = exp_config.num_train
    sample_type = 'front_sample'
    x_train, x_val, y_torque_train, y_torque_val,\
        y_eff_train, y_eff_val = load_front_train_val_data(
        num_train=num_train)


from efficientGP.models.gp_design.gp_model_from_config_module import get_gp_model

config_path = 'configs/from_gp_dnn_paper.yaml'
gp_torque = get_gp_model(config_path)
gp_torque.fit(x_train, y_torque_train)


config_path = 'configs/from_gp_dnn_paper.yaml'
gp_eff = get_gp_model(config_path)

gp_eff.fit(x_train, y_eff_train)

is_save_gp_model = True
# is_save_gp_model = False

if is_save_gp_model:
    from efficientGP.models.gp_design.utils import save_gp
    from efficientGP.models.gp_design.utils import load_gp
    # Save models
    saved_models_path = 'model_weights'

    model_filename = f"{saved_models_path}/gp_torque_{sample_type}_{num_train}.pkl"
    save_gp(gp_torque, model_filename)
    gp_torque = load_gp(model_filename)

    model_filename = f"{saved_models_path}/gp_efficiency_{sample_type}_{num_train}.pkl"
    save_gp(gp_eff, model_filename)
    gp_eff = load_gp(model_filename)



from efficientGP.metric.error_percentage_module import error_percentage


# Evaluate the models

y_torque_pred, std_torque = gp_torque.predict(x_val, return_std=True)
error_percentage_torque = error_percentage(y_torque_val, y_torque_pred)


y_eff_pred, std_eff = gp_eff.predict(x_val, return_std=True)
error_percentage_eff = error_percentage(y_eff_val, y_eff_pred)

print(f'Torque Error: {error_percentage_torque:.2f}')
print(f'Efficiency Error: {error_percentage_eff:.2f}')
print('done')