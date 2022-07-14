# Copyright (c) OpenMMLab. All rights reserved.
import os
from pathlib import Path

import requests
import yaml

MMRAZOR_ROOT = Path(__file__).absolute().parents[1]


class TestMetafiles:

    def get_metafiles(self, code_path):
        """
        Function: get the metafile of all configs from model-index.yml
        """
        metafile = os.path.join(code_path, 'model-index.yml')
        with open(metafile, 'r') as f:
            meta = yaml.safe_load(f)
        return meta['Import']

    def test_metafiles(self):
        metafiles = self.get_metafiles(MMRAZOR_ROOT)
        for mf in metafiles:
            metafile = os.path.abspath(os.path.join(MMRAZOR_ROOT, mf))
            with open(metafile, 'r') as f:
                meta = yaml.safe_load(f)
            for model in meta['Models']:
                # 1. weights url check
                r = requests.head(model['Weights'], timeout=4)
                assert r.status_code != 404, \
                    f"can't connect url {model['Weights']} in " \
                    f'metafile {metafile}'

                # 2. config check
                dir_path = os.path.abspath(os.path.join(metafile, '../'))
                # list all files which are in the same directory of
                # current metafile
                config_files = os.listdir(dir_path)

                if isinstance(model['Config'], list):
                    # TODO: 3. log error
                    continue

                assert (model['Config'].split('/')[-1] in config_files), \
                    f"config error in {metafile} model {model['Name']}"

                # 4. name check
                # erase '.py'
                correct_name = model['Config'].split('/')[-1][:-3]
                assert model['Name'] == correct_name, \
                    f'name error in {metafile}, correct name should ' \
                    f'be {correct_name}'
