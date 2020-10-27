import yaml

data_config_path = "./config_files/data_config/crS_rs113-unimodal_default.yaml"

with open(data_config_path, 'r') as dconfig:
    data_config = yaml.load(dconfig)

print(data_config.keys())
print(data_config.values())