import os
import time
import math
import copy
from sklearn import metrics

import torch
import torch.nn as nn
import torch.optim as optim

from utils import utils
from utils.EarlyStopping import EarlyStopping
from utils.GaussianSmoothing import GaussianSmoothing

from networks.UNet_PriCheXyNet import UNet
from networks.UNet_PrivacyNet import Unet2D_encoder
from networks.SiameseNetwork import SiameseNetwork


class AgentSiameseNetwork:
    def __init__(self, config):
        """This is the agent that provides code for re-training, validating and testing the patient verification model.

        :param config: dict
            A dictionary that stores the hyper-parameter configuration and some other important variables.
        """

        self.config = config

        # Set path used to save experiment-related files and results
        self.SAVINGS_PATH = './archive/' + self.config['experiment_description'] + '/'
        self.IMAGE_PATH = self.config['image_path']

        # Reproducibility
        utils.seed_all(self.config.get('seed', 42))

        # Set all the important variables
        self.perturbation_type = self.config['perturbation_type']
        self.perturbation_model_file = self.config['perturbation_model_file']
        self.mu = self.config['mu']
        self.stochastic_lambda = self.config.get('stochastic_lambda', 0.0)
        self.transform_mode = utils.resolve_transform_mode(self.config.get('transform_mode', 'legacy'))
        utils.record_transform_mode_provenance(self.transform_mode, self.SAVINGS_PATH, self.config)

        self.b = self.config['b']
        self.m = self.config['m']
        self.eps = self.config['eps']

        self.image_size = self.config['image_size']
        self.batch_size = self.config['batch_size']
        self.learning_rate = self.config['learning_rate']
        self.max_epochs = self.config['max_epochs']
        self.early_stopping = self.config['early_stopping']

        self.num_workers = 16
        self.pin_memory = True

        # STEP 2B.2: real-run diagnostics wiring. All keys are optional so legacy
        # configs (plain retrain_SNN.py) remain bit-for-bit unchanged: diagnostics are
        # only persisted when a path is supplied by the protocol runner.
        from adaptive_reid import diagnostics as arm_diag
        self.training_diagnostics_path = self.config.get('training_diagnostics_path')
        self.evaluate_test_after_training = self.config.get('evaluate_test_after_training', True)
        self.run_start_timestamp = arm_diag.utcnow_iso()

        # Define the identity grid and the 
        if self.perturbation_type == 'flow_field':
            d = torch.linspace(-1, 1, 256)
            mesh_x, mesh_y = torch.meshgrid((d, d), indexing='ij')
            grid_identity = torch.stack((mesh_y, mesh_x), 2)
            self.grid_identity = grid_identity.unsqueeze(0).permute(0, 3, 1, 2).cuda()
            self.gauss_filter = GaussianSmoothing(channels=2, kernel_size=9, sigma=2).cuda()
        elif self.perturbation_type in ['privacy_net', 'dp_pix', 'none']:
            self.grid_identity = None
            self.gauss_filter = None
        else:
            raise Exception('Invalid argument: ' + self.perturbation_type)

        self.start_epoch = 0
        self.es = EarlyStopping(patience=self.early_stopping)
        self.best_loss = 100000
        self.loss_dict = {'training': [],
                          'validation': []}

        # Initialize the perturbation network
        if self.perturbation_type == 'none':
            self.n_channels = 3
            self.perturbation_net = None
        else:
            self.n_channels = 1
            if self.perturbation_type == 'flow_field':
                self.perturbation_net = utils.load_flow_generator(self.perturbation_model_file)
            elif self.perturbation_type == 'privacy_net':
                self.perturbation_net = Unet2D_encoder(self.n_channels, self.n_channels, 16).cuda()
                self.perturbation_net.load_state_dict(torch.load(self.perturbation_model_file, weights_only=False))
            elif self.perturbation_type == 'dp_pix':
                self.perturbation_net = None
            else:
                raise Exception('Invalid argument: ' + self.perturbation_type)
            

        # Define the siamese neural network architecture
        self.net = SiameseNetwork().cuda()
        self.best_net = copy.deepcopy(self.net)

        # Choose loss function
        self.loss = nn.BCEWithLogitsLoss().cuda()

        # Set the optimizer function
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

        # Initialize data loaders
        self.training_loader = utils.get_data_loader(phase='training', experimental_step='retrainSNN', 
                                                     image_size=self.image_size, n_channels=self.n_channels, 
                                                     batch_size=self.batch_size, shuffle=True, 
                                                     num_workers=self.num_workers, pin_memory=self.pin_memory, 
                                                     b=self.b, m=self.m, eps=self.eps, image_path=self.IMAGE_PATH)
        self.validation_loader = utils.get_data_loader(phase='validation', experimental_step='retrainSNN',
                                                       image_size=self.image_size, n_channels=self.n_channels, 
                                                       batch_size=self.batch_size, shuffle=False, 
                                                       num_workers=self.num_workers, pin_memory=self.pin_memory, 
                                                       b=self.b, m=self.m, eps=self.eps, image_path=self.IMAGE_PATH)
        self.test_loader = None
        if self.evaluate_test_after_training:
            self.test_loader = utils.get_data_loader(phase='testing', experimental_step='retrainSNN',
                                                     image_size=self.image_size, n_channels=self.n_channels,
                                                     batch_size=self.batch_size, shuffle=False,
                                                     num_workers=self.num_workers, pin_memory=self.pin_memory,
                                                     b=self.b, m=self.m, eps=self.eps, image_path=self.IMAGE_PATH)

    def training_validation(self):
        """Training and validation loop.

        STEP 2B.2: for protocol-run real attacker attempts this method additionally
        persists ``training_diagnostics.json`` (see ``adaptive_reid.diagnostics``) with
        per-epoch training/validation loss, validation AUC + accuracy, best-epoch
        accounting, termination reason, NaN/Inf state, and parameter-hash state. All
        values come from this actual run; nothing is fabricated. When no
        ``training_diagnostics_path`` is configured (legacy ``retrain_SNN.py``), the
        historical behaviour is preserved bit-for-bit.
        """
        from adaptive_reid import diagnostics as arm_diag
        from adaptive_reid import weights as arm_weights
        from adaptive_reid import constants as arm_const

        initial_parameter_hash = arm_weights.parameters_hash(self.net)

        training_losses = []
        validation_losses = []
        validation_aucs = []
        validation_accs = []
        any_nan_inf = False
        best_validation_auc = -1.0
        best_validation_auc_epoch = -1
        best_validation_loss_epoch = -1
        termination_reason = arm_const.TERMINATION_EPOCH_CAP

        for epoch in range(self.start_epoch, self.max_epochs):
            start_time = time.time()

            training_loss = utils.train_snn(self.perturbation_type, self.net, self.perturbation_net, self.grid_identity,
                                            self.gauss_filter, self.mu, self.training_loader, self.loss, self.optimizer,
                                            epoch, self.max_epochs, stochastic_lambda=self.stochastic_lambda,
                                            transform_mode=self.transform_mode)
            val_result = utils.validate_snn(self.perturbation_type, self.net, self.perturbation_net,
                                            self.grid_identity, self.gauss_filter, self.mu, self.validation_loader,
                                            self.loss, epoch, self.max_epochs,
                                            stochastic_lambda=self.stochastic_lambda,
                                            transform_mode=self.transform_mode, return_metrics=True)
            validation_loss, val_metrics = val_result
            validation_auc = val_metrics['auc']
            validation_acc = val_metrics['accuracy']

            training_losses.append(float(training_loss))
            validation_losses.append(float(validation_loss))
            validation_aucs.append(float(validation_auc))
            validation_accs.append(float(validation_acc))

            if not (math.isfinite(float(training_loss))
                    and math.isfinite(float(validation_loss))
                    and math.isfinite(float(validation_auc))
                    and math.isfinite(float(validation_acc))):
                any_nan_inf = True

            self.loss_dict['training'].append(training_loss)
            self.loss_dict['validation'].append(validation_loss)

            end_time = time.time()
            print('Time elapsed for epoch ' + str(epoch + 1) + ': ' + str(
                round((end_time - start_time) / 60, 2)) + ' minutes')

            if validation_loss < self.best_loss:
                self.best_loss = validation_loss
                self.best_net = copy.deepcopy(self.net)
                best_validation_loss_epoch = epoch

            if validation_auc > best_validation_auc:
                best_validation_auc = validation_auc
                best_validation_auc_epoch = epoch

            torch.save(self.best_net.state_dict(), self.SAVINGS_PATH + self.config[
                'experiment_description'] + '_best_network.pth')

            utils.save_loss_curves_snn(self.loss_dict, self.SAVINGS_PATH, self.config['experiment_description'])
            utils.plot_loss_curves_snn(self.loss_dict, self.SAVINGS_PATH, self.config['experiment_description'])

            if self.es.step(validation_loss):
                termination_reason = arm_const.TERMINATION_EARLY_STOPPING
                break

        print('Finished Training!')

        if self.training_diagnostics_path:
            self._persist_training_diagnostics(
                arm_diag=arm_diag, arm_weights=arm_weights,
                initial_parameter_hash=initial_parameter_hash,
                training_losses=training_losses,
                validation_losses=validation_losses,
                validation_aucs=validation_aucs,
                validation_accs=validation_accs,
                any_nan_inf=any_nan_inf,
                best_validation_auc=best_validation_auc,
                best_validation_auc_epoch=best_validation_auc_epoch,
                best_validation_loss_epoch=best_validation_loss_epoch,
                termination_reason=termination_reason)

    def _persist_training_diagnostics(self, *, arm_diag, arm_weights, initial_parameter_hash,
                                      training_losses, validation_losses, validation_aucs,
                                      validation_accs, any_nan_inf, best_validation_auc,
                                      best_validation_auc_epoch, best_validation_loss_epoch,
                                      termination_reason):
        """        Build + write the real training-diagnostics record for this actual run.

        Only called when the protocol runner supplies ``training_diagnostics_path``.
        All execution-health values (parameter hashes, checkpoint loadability,
        termination reason, NaN/Inf) are computed from this real run.
        """
        checkpoint_path = self.SAVINGS_PATH + self.config['experiment_description'] + '_best_network.pth'
        checkpoint_exists = os.path.exists(checkpoint_path)
        checkpoint_loadable = arm_weights.checkpoint_loadable(checkpoint_path)
        final_parameter_hash = arm_weights.parameters_hash(self.net)
        weights_changed = arm_weights.weights_changed(initial_parameter_hash, final_parameter_hash)

        epochs_completed = len(training_losses)
        # Canonical checkpoint selection is LOWEST VALIDATION LOSS; the best loss and
        # its epoch are exactly what drove self.best_net, not a recomputation.
        best_validation_loss = float(self.best_loss) if validation_losses else float('inf')
        best_validation_loss_epoch = (best_validation_loss_epoch
                                      if validation_losses and best_validation_loss_epoch >= 0
                                      else -1)
        best_validation_auc = (best_validation_auc
                               if validation_aucs and best_validation_auc_epoch >= 0 else 0.0)
        best_validation_auc_epoch = best_validation_auc_epoch if validation_aucs else -1

        record = arm_diag.build_training_diagnostics(
            attacker_seed=int(self.config.get('seed', 0)),
            transform_mode=self.transform_mode,
            mu=float(self.mu),
            stochastic_lambda=float(self.stochastic_lambda),
            generator_checkpoint_path=str(self.perturbation_model_file),
            generator_checkpoint_hash=str(self.config.get('generator_checkpoint_hash', '')),
            pair_train_path=str(self.config.get('pair_train_path', '')),
            pair_validation_path=str(self.config.get('pair_validation_path', '')),
            pair_train_hash=str(self.config.get('pair_train_hash', '')),
            pair_validation_hash=str(self.config.get('pair_validation_hash', '')),
            epochs_completed=epochs_completed,
            termination_reason=termination_reason,
            training_loss_per_epoch=training_losses,
            validation_loss_per_epoch=validation_losses,
            validation_auc_per_epoch=validation_aucs,
            validation_accuracy_per_epoch=validation_accs,
            best_validation_loss=best_validation_loss,
            best_validation_loss_epoch=best_validation_loss_epoch,
            best_validation_auc=best_validation_auc,
            best_validation_auc_epoch=best_validation_auc_epoch,
            any_nan_inf=any_nan_inf,
            checkpoint_exists=checkpoint_exists,
            checkpoint_loadable=checkpoint_loadable,
            weights_changed_from_initialization=weights_changed,
            initial_parameter_hash=initial_parameter_hash,
            final_parameter_hash=final_parameter_hash,
            protocol_documents=dict(self.config.get('protocol_documents', {}) or {}),
            frozen_artifacts=dict(self.config.get('frozen_artifacts', {}) or {}),
            run_start_timestamp=str(self.run_start_timestamp),
            run_end_timestamp=arm_diag.utcnow_iso(),
        )
        arm_diag.persist_real_training_diagnostics(self.training_diagnostics_path, record)
    
    def testing_evaluation(self):
        # Testing phase
        y_true, y_pred = utils.test_snn(self.perturbation_type, self.best_net, self.perturbation_net, 
                                        self.grid_identity, self.gauss_filter, self.mu, self.test_loader,
                                        stochastic_lambda=self.stochastic_lambda, transform_mode=self.transform_mode)

        y_true, y_pred = [y_true.numpy(), y_pred.numpy()]

        # Compute the evaluation metrics
        fp_rates, tp_rates, thresholds = metrics.roc_curve(y_true, y_pred)
        auc = metrics.roc_auc_score(y_true, y_pred)
        y_pred_thresh = utils.apply_threshold(y_pred, 0.5)
        accuracy, f1_score, precision, recall, report, confusion_matrix = utils.get_evaluation_metrics(y_true,
                                                                                                       y_pred_thresh)
        auc_mean, confidence_lower, confidence_upper = utils.bootstrap(1000,
                                                                       y_true,
                                                                       y_pred,
                                                                       self.SAVINGS_PATH,
                                                                       self.config['experiment_description'])

        # Plot ROC curve
        utils.plot_roc_curve(fp_rates, tp_rates, self.SAVINGS_PATH, self.config['experiment_description'])

        # Save all the results to files
        utils.save_labels_predictions(y_true, y_pred, y_pred_thresh, self.SAVINGS_PATH,
                                      self.config['experiment_description'])

        utils.save_results_to_file(auc, accuracy, f1_score, precision, recall, report, confusion_matrix,
                                   self.SAVINGS_PATH, self.config['experiment_description'])

        utils.save_roc_metrics_to_file(fp_rates, tp_rates, thresholds, self.SAVINGS_PATH,
                                       self.config['experiment_description'])

        # Print the evaluation metrics
        print('EVALUATION METRICS:')
        print('AUC: ' + str(auc))
        print('Accuracy: ' + str(accuracy))
        print('F1-Score: ' + str(f1_score))
        print('Precision: ' + str(precision))
        print('Recall: ' + str(recall))
        print('Report: ' + str(report))
        print('Confusion matrix: ' + str(confusion_matrix))

        print('BOOTSTRAPPING: ')
        print('AUC Mean: ' + str(auc_mean))
        print('Confidence interval for the AUC score: ' + str(confidence_lower) + ' - ' + str(confidence_upper))
    
    def run(self):
        # Call training/validation and testing loop successively.
        self.training_validation()
        # STEP 2B.2 (protocol): the frozen pipeline evaluates the TEST split only in
        # Stage E (after representative selection is frozen). The runner therefore sets
        # `evaluate_test_after_training: False` so a protocol attacker attempt trains +
        # validates but never touches the test split here. Legacy callers keep the
        # historical behaviour (default True).
        if self.evaluate_test_after_training:
            self.testing_evaluation()
