import copy
import hashlib
import json
import os
import subprocess
import time
from statistics import mean
from sklearn import metrics

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from utils import utils
from utils.EarlyStopping import EarlyStopping
from utils.GaussianSmoothing import GaussianSmoothing

from networks.UNet_AttentionPriCheXyNet import UNetAtt
from networks.SiameseNetwork import SiameseNetwork


_FROZEN_MU = 0.01


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class AgentSiameseNetworkV2:
    """Re-train / validate / evaluate the patient verification model (SNN)
    against images anonymized by a V2 attention-gated generator.

    Faithfulness contract with agents.AgentSiameseNetwork:
      * Identical seeding, SNN architecture, loss, optimizer, early stopping,
        batch handling, preprocessing order (deform raw 1-channel [0,1] image,
        then expand to 3 channels and ImageNet-normalize) and metric pipeline.
      * Differences:
          (a) perturbation_type='flow_field_att' loads the generator as a
              UNetAtt with STRICT full-state loading (every key present --
              training-time and eval-time graphs must match bit-for-bit),
          (b) the TEST split is OPT-IN via config['allow_test'] (default
              False): development stays inside the TRAIN/VAL firewall of
              METHOD_RESTART_BRANCH_LOCK.md. When False, no test loader is
              ever constructed and testing_evaluation() refuses to run.
    """

    def __init__(self, config):
        self.config = config
        self._validate_config(config)

        self.SAVINGS_PATH = './archive/' + config['experiment_description'] + '/'
        self.IMAGE_PATH = config['image_path']

        utils.seed_all(42)

        self.perturbation_type = config['perturbation_type']
        self.perturbation_model_file = config['perturbation_model_file']
        self.mu = float(config['mu'])
        self.allow_test = bool(config.get('allow_test', False))

        # dp_pix is not part of the V2 scope; keep keys optional like the
        # baseline requires them, but only enforce them for dp_pix.
        if self.perturbation_type == 'dp_pix':
            raise ValueError("dp_pix is not supported by the V2 agent.")
        self.b = config.get('b')
        self.m = config.get('m')
        self.eps = config.get('eps')

        self.image_size = int(config['image_size'])
        self.batch_size = int(config['batch_size'])
        self.learning_rate = float(config['learning_rate'])
        self.max_epochs = int(config['max_epochs'])
        self.early_stopping = int(config['early_stopping'])

        self.num_workers = 16
        self.pin_memory = True

        if self.perturbation_type in ('flow_field_att', 'flow_field'):
            # NOTE: the baseline hardcodes the identity grid at 256x256
            # regardless of config; mirror that exactly and refuse any other
            # size so the operator cannot silently diverge from the baseline.
            if self.image_size != 256:
                raise ValueError('flow_field evaluation pins image_size=256 '
                                 '(baseline parity), got %s.' % self.image_size)
            d = torch.linspace(-1, 1, 256)
            mesh_x, mesh_y = torch.meshgrid((d, d), indexing='ij')
            grid_identity = torch.stack((mesh_y, mesh_x), 2)
            self.grid_identity = grid_identity.unsqueeze(0).permute(0, 3, 1, 2).cuda()
            self.gauss_filter = GaussianSmoothing(channels=2, kernel_size=9, sigma=2).cuda()
            self.n_channels = 1

            if not os.path.exists(self.perturbation_model_file):
                raise FileNotFoundError('Perturbation model not found: %s'
                                        % self.perturbation_model_file)
            checkpoint = torch.load(self.perturbation_model_file, map_location='cpu')
            if self.perturbation_type == 'flow_field_att':
                self.perturbation_net = UNetAtt(1, 2, 32).cuda()
                # STRICT load: an eval-time graph that differs from the trained
                # one invalidates every number this agent could produce.
                self.perturbation_net.load_state_dict(checkpoint, strict=True)
            else:
                from networks.UNet_PriCheXyNet import UNet
                self.perturbation_net = UNet(1, 2, 32).cuda()
                self.perturbation_net.load_state_dict(checkpoint, strict=True)
            self.perturbation_net.eval()
            # The anonymizer is a FROZEN measurement instrument here, never a
            # trainable module: only self.net (the attacker) is optimized.
            # Without this, its 118 parameter tensors keep requires_grad=True,
            # so every attacker backward also propagates through the whole
            # U-Net and accumulates .grad buffers that are never zeroed (the
            # attacker's optimizer only owns self.net) and never stepped --
            # pure wasted compute and memory. Mirrors the P0 harness contract
            # in generator_guard.assert_generator_frozen_state.
            for p in self.perturbation_net.parameters():
                p.requires_grad_(False)
        else:
            raise ValueError("AgentSiameseNetworkV2 supports 'flow_field_att' "
                             "or 'flow_field', got '%s'." % self.perturbation_type)

        self.start_epoch = 0
        self.es = EarlyStopping(patience=self.early_stopping)
        self.best_loss = 100000
        self.loss_dict = {'training': [], 'validation': []}

        self.net = SiameseNetwork().cuda()
        self.best_net = copy.deepcopy(self.net)

        self.loss = nn.BCEWithLogitsLoss().cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.training_loader = utils.get_data_loader(
            phase='training', experimental_step='retrainSNN',
            image_size=self.image_size, n_channels=self.n_channels,
            batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            b=self.b, m=self.m, eps=self.eps, image_path=self.IMAGE_PATH)
        self.validation_loader = utils.get_data_loader(
            phase='validation', experimental_step='retrainSNN',
            image_size=self.image_size, n_channels=self.n_channels,
            batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            b=self.b, m=self.m, eps=self.eps, image_path=self.IMAGE_PATH)

        # ImageNet transform used by the SNN branch of the baseline
        # (utils.train_snn / validate_snn / test_snn): resize is NOT applied
        # there -- inputs stay at 256 -- only normalization. Keep identical.
        self.snn_transform = normalize

        self._write_provenance_manifest()

    @staticmethod
    def _validate_config(config):
        required = ['experiment_description', 'image_path', 'perturbation_type',
                    'perturbation_model_file', 'mu', 'image_size', 'batch_size',
                    'learning_rate', 'max_epochs', 'early_stopping']
        missing = [k for k in required if k not in config]
        if missing:
            raise ValueError('Missing required config keys: %s' % missing)
        if abs(float(config['mu']) - _FROZEN_MU) > 1e-12:
            raise ValueError('mu=%s violates frozen invariant mu=%s.'
                             % (config['mu'], _FROZEN_MU))
        if config['perturbation_type'] not in ('flow_field_att', 'flow_field'):
            raise ValueError("perturbation_type must be 'flow_field_att' or "
                             "'flow_field'.")
        if not os.path.exists(config['image_path']):
            raise ValueError('image_path does not exist: %s' % config['image_path'])

    def _write_provenance_manifest(self):
        def git_head():
            try:
                return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                               cwd='./').decode().strip()
            except Exception:
                return None

        manifest = {
            'agent': 'AgentSiameseNetworkV2',
            'git_head': git_head(),
            'torch': torch.__version__,
            'config': json.loads(json.dumps(self.config)),
            'perturbation_model_sha256': _sha256_file(self.perturbation_model_file),
            'test_split_accessed': False,
            'allow_test': self.allow_test,
            'image_pairs_sha256': {
                p: (_sha256_file('./image_pairs/' + p)
                    if os.path.exists('./image_pairs/' + p) else None)
                for p in ['image_pairs_training_10000.txt',
                          'image_pairs_validation_2000.txt',
                          'image_pairs_testing_5000.txt']},
        }
        with open(self.SAVINGS_PATH + 'v2_snn_provenance_manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        print('[SNN-V2] Provenance manifest written.')

    # ------------------------------------------------------------------ #
    # Anonymization operator (mirrors utils.train_snn flow_field branch)  #
    # ------------------------------------------------------------------ #

    def _anonymize_pair(self, inputs1, inputs2=None):
        out1 = self._anonymize_single(inputs1)
        if inputs2 is None:
            return out1, None
        return out1, self._anonymize_single(inputs2)

    def _anonymize_single(self, x):
        # no_grad: the attacker only needs gradients w.r.t. its OWN parameters,
        # and the anonymized image is an input to it, not a differentiable
        # function of anything being optimized. Building the generator's graph
        # here would cost a full U-Net backward per batch for nothing.
        with torch.no_grad():
            grid = self.perturbation_net(x)
            grid = self.grid_identity - self.mu * grid
            grid = self.gauss_filter(grid)
            grid = grid.permute(0, 2, 3, 1)
            return torch.nn.functional.grid_sample(x, grid,
                                                   padding_mode='border',
                                                   align_corners=True)

    # ------------------------------------------------------------------ #
    # Epoch loops (mirror utils.train_snn / validate_snn step-for-step)   #
    # ------------------------------------------------------------------ #

    def _run_epoch(self, loader, training, epoch):
        self.net.train(mode=training)
        self.perturbation_net.eval()
        running_loss = []
        print(('Training----->' if training else 'Validating----->'))

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for i, batch in enumerate(loader):
                inputs1, inputs2, labels = batch
                inputs1 = inputs1.cuda()
                inputs2 = inputs2.cuda()
                labels = labels.cuda()

                inputs1, inputs2 = self._anonymize_pair(inputs1, inputs2)
                inputs1 = self.snn_transform(inputs1.expand(-1, 3, -1, -1))
                inputs2 = self.snn_transform(inputs2.expand(-1, 3, -1, -1))

                outputs = self.net(inputs1, inputs2).squeeze()
                labels_t = labels.type_as(outputs)
                loss = self.loss(outputs, labels_t)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                running_loss.append(loss.item())
                print('Epoch [%d/%d], Iteration [%d/%d], Loss: %.4f'
                      % (epoch + 1, self.max_epochs, i + 1, len(loader), loss.item()))

        return mean(running_loss) if running_loss else float('inf')

    def training_validation(self):
        for epoch in range(self.start_epoch, self.max_epochs):
            start_time = time.time()

            training_loss = self._run_epoch(self.training_loader, True, epoch)
            validation_loss = self._run_epoch(self.validation_loader, False, epoch)

            self.loss_dict['training'].append(training_loss)
            self.loss_dict['validation'].append(validation_loss)

            print('Time elapsed for epoch ' + str(epoch + 1) + ': ' +
                  str(round((time.time() - start_time) / 60, 2)) + ' minutes')

            if validation_loss < self.best_loss:
                self.best_loss = validation_loss
                self.best_net = copy.deepcopy(self.net)

            torch.save(self.best_net.state_dict(),
                       self.SAVINGS_PATH + self.config['experiment_description']
                       + '_best_network.pth')

            try:
                utils.save_loss_curves_snn(self.loss_dict, self.SAVINGS_PATH,
                                           self.config['experiment_description'])
                utils.plot_loss_curves_snn(self.loss_dict, self.SAVINGS_PATH,
                                           self.config['experiment_description'])
            except Exception as e:
                print('[SNN-V2] loss-curve helpers skipped: %r' % e)

            if self.es.step(validation_loss):
                break

        print('Finished Training!')

    # ------------------------------------------------------------------ #
    # Evaluation                                                          #
    # ------------------------------------------------------------------ #

    def evaluate_on_loader(self, loader, tag):
        """Run the realistic attack scenario (real image vs anonymized image)
        on the given loader using the best-validation SNN. Mirrors
        utils.test_snn's flow_field branch exactly."""
        net = self.best_net
        net.eval()
        self.perturbation_net.eval()
        y_true = None
        y_pred = None

        print('Evaluating (%s)----->' % tag)
        with torch.no_grad():
            for i, batch in enumerate(loader):
                inputs1, inputs2, labels = batch

                y_true = labels if y_true is None else torch.cat((y_true, labels), 0)

                inputs1 = inputs1.cuda()
                inputs2 = inputs2.cuda()

                inputs1 = self._anonymize_single(inputs1)
                inputs1 = self.snn_transform(inputs1.expand(-1, 3, -1, -1))
                inputs2 = self.snn_transform(inputs2.expand(-1, 3, -1, -1))

                outputs = torch.sigmoid(net(inputs1, inputs2))
                y_pred = outputs.cpu() if y_pred is None else torch.cat((y_pred, outputs.cpu()), 0)

        return y_true.numpy(), y_pred.squeeze().numpy()

    def report_metrics(self, y_true, y_pred, tag):
        fp_rates, tp_rates, thresholds = metrics.roc_curve(y_true, y_pred)
        auc = metrics.roc_auc_score(y_true, y_pred)
        y_pred_thresh = utils.apply_threshold(y_pred, 0.5)
        accuracy, f1_score, precision, recall, report, confusion_matrix = \
            utils.get_evaluation_metrics(y_true, y_pred_thresh)

        results = {
            'tag': tag,
            'auc': auc,
            'accuracy': accuracy,
            'f1_score': f1_score,
            'precision': precision,
            'recall': recall,
        }
        with open(self.SAVINGS_PATH + '%s_metrics.json' % tag, 'w') as f:
            json.dump(results, f, indent=2)

        # Full text artifacts, mirroring the baseline file layout.
        utils.save_labels_predictions(y_true, y_pred, y_pred_thresh,
                                      self.SAVINGS_PATH,
                                      self.config['experiment_description'] + '_' + tag)
        utils.save_results_to_file(auc, accuracy, f1_score, precision, recall,
                                   report, confusion_matrix, self.SAVINGS_PATH,
                                   self.config['experiment_description'] + '_' + tag)
        utils.save_roc_metrics_to_file(fp_rates, tp_rates, thresholds,
                                       self.SAVINGS_PATH,
                                       self.config['experiment_description'] + '_' + tag)
        utils.plot_roc_curve(fp_rates, tp_rates, self.SAVINGS_PATH,
                             self.config['experiment_description'] + '_' + tag)

        auc_mean, ci_lo, ci_hi = utils.bootstrap(
            1000, y_true, y_pred, self.SAVINGS_PATH,
            self.config['experiment_description'] + '_' + tag)

        print('[%s] EVALUATION METRICS:' % tag)
        print('AUC: ' + str(auc))
        print('Accuracy: ' + str(accuracy))
        print('BOOTSTRAP AUC Mean: %s CI [%s, %s]' % (auc_mean, ci_lo, ci_hi))
        return results

    def run(self):
        self.training_validation()

        # VAL evaluation always runs (development firewall compliant).
        y_true, y_pred = self.evaluate_on_loader(self.validation_loader, 'val')
        self.report_metrics(y_true, y_pred, 'val')

        # TEST evaluation ONLY on explicit opt-in.
        if self.allow_test:
            print('[SNN-V2] allow_test=True: touching the TEST split. Make sure '
                  'this is a final confirmatory run, not development.')
            test_loader = utils.get_data_loader(
                phase='testing', experimental_step='retrainSNN',
                image_size=self.image_size, n_channels=self.n_channels,
                batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=self.pin_memory,
                b=self.b, m=self.m, eps=self.eps, image_path=self.IMAGE_PATH)
            yt, yp = self.evaluate_on_loader(test_loader, 'test')
            self.report_metrics(yt, yp, 'test')
        else:
            print('[SNN-V2] TEST split NOT accessed (allow_test=False). '
                  'VAL AUC is the reported development metric.')
