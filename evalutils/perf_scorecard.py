import yaml
from collections import OrderedDict

CATEGORIES = ("Data", "Model", "Per Centre Metrics", "Global Metrics")


class PerformanceScorecard():

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.scorecard_dict = OrderedDict()
        for category in CATEGORIES:
            self.scorecard_dict[category] = {}

    def add_info(self, info_name, info, category):
        self.scorecard_dict[category][info_name] = info

    def write_to_file(self):
        with open(f"{self.output_dir}/scorecard.yaml", 'w') as yaml_file:
            yaml.dump(dict(self.scorecard_dict), yaml_file, default_flow_style=False, sort_keys=False)