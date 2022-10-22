import json


def cfd_parse(json_filepath):
    with open(json_filepath) as f:
        args = json.load(f)
    json_formatted_str = json.dumps(args, indent=4)
    print(json_formatted_str)
    return args
