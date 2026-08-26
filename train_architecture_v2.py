import os
import json
import argparse
from utils import utils
from agents.AgentV2 import AgentV2


if __name__ == "__main__":
    print('------------------------------------------')
    print('-- Train Anonymization Model (PriCheXy V2) --')
    print('------------------------------------------' + '\n')

    parser = argparse.ArgumentParser('Train Anonymization Model V2')
    parser.add_argument('--config_path', default='./config_files/')
    parser.add_argument('--config', default='config_anonymization_v2_attention.json')
    args = parser.parse_args()
    print('Arguments:\n' + '--config_path: ' + args.config_path +
          '\n--config: ' + args.config + '\n')

    # Normalize the config path: the baseline concatenates config_path +
    # config verbatim, so a missing trailing slash silently corrupts the
    # path. Add it if absent.
    config_path = args.config_path if args.config_path.endswith(os.sep) \
        else args.config_path + os.sep

    with open(config_path + args.config, 'r') as f:
        config = json.loads(f.read())

    # Fail-closed like the baseline runner: refuse to reuse an existing
    # experiment directory.
    os.mkdir('./archive/' + config['experiment_description'])
    SAVINGS_PATH = './archive/' + config['experiment_description'] + '/'

    # NOTE: the baseline runner calls utils.make_zip over the whole repo,
    # which on this workspace produces a multi-GB archive (.venv, datasets,
    # research_runs are all under ./) for every experiment launch. The V2
    # provenance manifest inside AgentV2 already records SHA-256 of every
    # relevant source file plus the full config, so we persist just those.
    import shutil
    shutil.copy(config_path + args.config, SAVINGS_PATH + args.config)

    # Call agent and run experiment
    experiment = AgentV2(config)
    experiment.run()
