from omegaconf import OmegaConf
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel


def get_gp_model(
    config_path = 'configs/from_gp_dnn_paper.yaml'
):
    conf = OmegaConf.load(config_path)

    # GP configs
    length_scale = conf.length_scale                                # Length scale for the GP kernel
    nu = conf.nu                                        # Smoothness parameter for the Matern kernel
    noise_level = conf.noise_level                              # Noise level in the GP model
    noise_level_bounds = (conf.noise_level_bounds_lower, conf.noise_level_bounds_upper)              # Bounds for the noise level
    alpha = conf.alpha                                    # Additive noise for stability in GP
    gp_optimizer = conf.gp_optimizer                # Optimizer for training GP hyperparameters
    n_restarts_optimizer = conf.n_restarts_optimizer                        # Number of restarts for the optimizer

    kernel = 1.0 * Matern(length_scale=length_scale, nu=nu) + WhiteKernel(noise_level=noise_level, noise_level_bounds=noise_level_bounds)

    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=alpha,
        optimizer=gp_optimizer,
        n_restarts_optimizer=n_restarts_optimizer,
        normalize_y=True)

    return gp


if __name__ == "__main__":
    config_path = 'configs/from_gp_dnn_paper.yaml'
    gp_model = get_gp_model(config_path)

    print('done')