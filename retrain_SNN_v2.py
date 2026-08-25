import os
import json
import argparse
from agents.AgentSiameseNetworkV2 import AgentSiameseNetworkV2


if __name__ == "__main__":
    print('---------------------------------------------')
    print('-- Patient Verification (PriCheXy-Net V2) --')
    print('---------------------------------------------' + '\n')

    parser = argparse.ArgumentParser('Retrain SNN against V2 anonymizer')
    parser.add_argument('--config_path', default='./config_files/')
    parser.add_argument('--config', default='config_retrainSNN_v2.json')
    args = parser.parse_args()
    print('Arguments:\n' + '--config_path: ' + args.config_path +
          '\n--config: ' + args.config + '\n')

    # Normalize config path (baseline concatenates verbatim).
    config_path = args.config_path if args.config_path.endswith(os.sep) \
        else args.config_path + os.sep

    with open(config_path + args.config, 'r') as f:
        config = json.loads(f.read())

    # Fail-closed like the baseline runner.
    os.mkdir('./archive/' + config['experiment_description'])
    SAVINGS_PATH = './archive/' + config['experiment_description'] + '/'

    experiment = AgentSiameseNetworkV2(config)
    experiment.run()
