import yaml

# Test config loading
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("Data directory:", config['data_paths']['raw_data_dir'])
print("Config loaded successfully!")