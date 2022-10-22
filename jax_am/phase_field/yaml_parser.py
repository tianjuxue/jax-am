import yaml


def pf_parse(yaml_filepath):     
    with open(yaml_filepath) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
        print(f"YAML parameters:")
        # TODO: These are just default parameters
        print(yaml.dump(args, default_flow_style=False))
        print(f"These are default parameters")
    return args


 
