from omegaconf import OmegaConf

config_path = '__CODE_SOUP/from_VAD/configs/test.yaml'
conf = OmegaConf.load(config_path)
print(OmegaConf.to_yaml(conf))

print('done')