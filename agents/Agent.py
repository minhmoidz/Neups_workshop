import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils import utils
from utils.ACLoss import ACLoss
from utils.VerificationLoss import VerificationLoss
from utils.GaussianSmoothing import GaussianSmoothing

from networks.UNet_PriCheXyNet import UNet
from networks.UNet_PrivacyNet import Unet2D_encoder
from networks.SiameseNetwork import SiameseNetwork


class Agent:
    def __init__(self, config):
        """This is the agent that provides code for training and validating the anonymization model.

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
        self.ac_loss_weight = self.config['ac_loss_weight']
        self.ver_loss_weight = self.config['ver_loss_weight']

        self.generator_type = self.config['generator_type']
        self.mu = self.config['mu']
        # Read before the networks and losses are built below, since they depend on these settings
        # Default 1 = faithful to the original paper's schedule (one generator update per batch)
        self.accumulation_steps = self.config.get('accumulation_steps', 1)
        self.use_budget_map = self.config.get('use_budget_map', False)
        self.stochastic_lambda = self.config.get('stochastic_lambda', 0.0)
        self.ac_pos_weight = self.config.get('ac_pos_weight', None)
        self.feature_loss_weight = self.config.get('feature_loss_weight', 0.0)

        self.image_size = self.config['image_size']
        self.batch_size = self.config['batch_size']
        self.learning_rate = self.config['learning_rate']
        self.max_epochs = self.config['max_epochs']

        self.num_workers = 8
        self.pin_memory = True

        self.show_every_n_epochs = self.config['show_every_n_epochs']
        self.show_every_n_iterations = self.config['show_every_n_iterations']

        self.writer = SummaryWriter(self.SAVINGS_PATH + 'runs/')

        if self.generator_type == 'flow_field':
            # Define the identity grid
            d = torch.linspace(-1, 1, self.image_size)
            mesh_x, mesh_y = torch.meshgrid((d, d), indexing='ij')
            grid_identity = torch.stack((mesh_y, mesh_x), 2)
            self.grid_identity = grid_identity.unsqueeze(0).permute(0, 3, 1, 2).cuda()
            # Define the Gauss filter which is used for smoothing the resulting flow field
            self.gauss_filter = GaussianSmoothing(channels=2, kernel_size=9, sigma=2).cuda()
            # Define the flow field generator
            # With a budget map the U-Net predicts a third channel next to the 2-channel flow field
            self.generator = UNet(1, 3 if self.use_budget_map else 2, 32).cuda()
            state = torch.load('./networks/pretrained_generator_prichexy_net.pth', weights_only=False)
            if self.use_budget_map:
                # Copy the pre-trained 2-channel output layer and zero the budget channel, so that the budget
                # map starts out uniform and training begins from the exact uniform-mu baseline.
                for key, size in (('conv.weight', 3), ('conv.bias', 3)):
                    padded = torch.zeros(size, *state[key].shape[1:], dtype=state[key].dtype)
                    padded[:2] = state[key]
                    state[key] = padded
            self.generator.load_state_dict(state)
        elif self.generator_type == 'privacy_net':
            # Set identity grid and gauss filter to None
            self.grid_identity = None
            self.gauss_filter = None
            # Define PrivacyNet encoder
            self.generator = Unet2D_encoder(1, 1, 16).cuda()
            self.generator.load_state_dict(torch.load('./networks/pretrained_generator_privacy_net.pth', weights_only=False))
        else:
            raise Exception('Invalid argument: ' + self.generator_type)

        self.start_epoch = 0
        self.lowest_ac_loss = 10000
        self.lowest_ver_loss = 10000
        self.lowest_total_loss = 10000
        self.loss_dict = {
            'training': {
                'ac_loss': [],
                'ver_loss': [],
                'log_likelihood_ver_loss': [],
                'total_loss': []
            },
            'validation': {
                'ac_loss': [],
                'ver_loss': [],
                'log_likelihood_ver_loss': [],
                'total_loss': []
            }
        }

        # Define the auxiliary classifier (self.ac_model is already on GPU)
        self.ac_model = torch.load('./networks/pretrained_classifier.pth', weights_only=False)['model']
        self.ac_loss = ACLoss(ac_model=self.ac_model, pos_weight=self.ac_pos_weight,
                              feature_loss_weight=self.feature_loss_weight).cuda()

        # Define the adversary ensemble and the verification loss.
        # A single, continuously fine-tuned adversary lets the generator overfit one parameter trajectory, whereas the
        # evaluation attacker is a fresh SNN trained from scratch. An ensemble plus periodic restarts closes that gap.
        self.ver_ensemble_size = self.config.get('ver_ensemble_size', 3)
        self.ver_restart_every = self.config.get('ver_restart_every', 25)
        self.ver_warmup_iters = self.config.get('ver_warmup_iters', 200)
        self.ver_active_per_step = self.config.get('ver_active_per_step', 1)

        verification_models = []
        for k in range(self.ver_ensemble_size):
            model = SiameseNetwork().cuda()
            if k == 0:
                # Keep one adversary initialized from the pre-trained verification model (paper behaviour)
                model.load_state_dict(torch.load('./networks/pretrained_verification_model.pth', weights_only=False))
            verification_models.append(model)
        self.verification_loss = VerificationLoss(verification_models=verification_models, reduction='none').cuda()

        # Loss functions for the auxiliary classifier and the verification model
        self.criterion_ac = nn.BCELoss().cuda()
        self.criterion_ver = nn.BCEWithLogitsLoss().cuda()

        # Set the optimizer functions
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        self.optimizers_ver = [optim.Adam(m.parameters(), lr=self.learning_rate)
                               for m in self.verification_loss.verification_models]
        self.optimizer_ac = optim.SGD(filter(lambda p: p.requires_grad, self.ac_loss.ac_model.parameters()),
                                      lr=self.learning_rate, momentum=0.9, weight_decay=1e-4)

        # Initialize data loaders
        self.training_loader = utils.get_data_loader(phase='training', experimental_step='anonymization', 
                                                     image_size=self.image_size, n_channels=1, 
                                                     batch_size=self.batch_size, shuffle=True, 
                                                     num_workers=self.num_workers, pin_memory=self.pin_memory, 
                                                     image_path=self.IMAGE_PATH)
        self.validation_loader = utils.get_data_loader(phase='validation', experimental_step='anonymization', 
                                                       image_size=self.image_size, n_channels=1, 
                                                       batch_size=self.batch_size, shuffle=False, 
                                                       num_workers=self.num_workers, pin_memory=self.pin_memory,
                                                       image_path=self.IMAGE_PATH)

    def training_validation(self):
        # Training and validation loop
        for epoch in range(self.start_epoch, self.max_epochs):
            start_time = time.time()

            # Periodically re-initialize one adversary from scratch (round-robin) so that the generator has to defeat
            # newly trained attackers, which is exactly the threat model used at evaluation time.
            if self.ver_restart_every > 0 and epoch > 0 and epoch % self.ver_restart_every == 0:
                self.restart_adversary(epoch)

            # Train the anonymization model
            train_losses = utils.train(self.generator, self.training_loader, self.gauss_filter, self.grid_identity, 
                                       self.mu, self.ac_loss, self.verification_loss, self.ac_loss_weight, 
                                       self.ver_loss_weight, self.optimizer_g, self.optimizer_ac,
                                       self.optimizers_ver, self.criterion_ac, self.criterion_ver, epoch,
                                       self.max_epochs, self.show_every_n_epochs, self.show_every_n_iterations,
                                       self.SAVINGS_PATH, accumulation_steps=self.accumulation_steps,
                                       ver_active_per_step=self.ver_active_per_step,
                                       stochastic_lambda=self.stochastic_lambda)

            # Validate the anonymization model
            val_losses = utils.validate(self.generator, self.validation_loader, self.gauss_filter, self.grid_identity, 
                                        self.mu, self.ac_loss, self.verification_loss, self.ac_loss_weight, 
                                        self.ver_loss_weight, epoch, self.max_epochs, self.show_every_n_epochs, 
                                        self.show_every_n_iterations, self.SAVINGS_PATH,
                                        stochastic_lambda=self.stochastic_lambda)

            end_time = time.time()
            print('Time elapsed for epoch ' + str(epoch + 1) + ': ' + str(
                round((end_time - start_time) / 60, 2)) + ' minutes')

            # Append losses to dict
            utils.append_losses_to_dict(self.loss_dict, 'training', train_losses)
            utils.append_losses_to_dict(self.loss_dict, 'validation', val_losses)

            # Plot loss curves
            utils.show_loss_curves(self.loss_dict, pre_train=False, save_fig=True, show_fig=False, path=self.SAVINGS_PATH)

            # Save loss dict
            utils.save_loss_dict(self.loss_dict, self.SAVINGS_PATH)

            # Save latest network components
            torch.save(self.generator.state_dict(), self.SAVINGS_PATH + 'generator_latest.pth')
            torch.save(self.ac_loss.ac_model.state_dict(), self.SAVINGS_PATH + 'ac_model_trained_latest.pth')
            for k, ver_model in enumerate(self.verification_loss.verification_models):
                torch.save(ver_model.state_dict(), self.SAVINGS_PATH + 'ver_model%d_trained_latest.pth' % k)

            # Save flow field generator that produces the lowest ac_loss
            if val_losses[0] < self.lowest_ac_loss:
                self.lowest_ac_loss = val_losses[0]
                torch.save(self.generator.state_dict(), self.SAVINGS_PATH + 'generator_lowest_ac_loss.pth')
                print('Current generator with lowest ac_loss: epoch ' + str(epoch))

            # Save flow field generator that produces the lowest ver_loss
            if val_losses[1] < self.lowest_ver_loss:
                self.lowest_ver_loss = val_losses[1]
                torch.save(self.generator.state_dict(), self.SAVINGS_PATH + 'generator_lowest_ver_loss.pth')
                print('Current generator with lowest ver_loss: epoch ' + str(epoch))

            # Save flow field generator that produces the lowest total_loss
            if val_losses[3] < self.lowest_total_loss:
                self.lowest_total_loss = val_losses[3]
                torch.save(self.generator.state_dict(), self.SAVINGS_PATH + 'generator_lowest_total_loss.pth')
                print('Current generator with lowest total_loss: epoch ' + str(epoch))
                torch.save(self.ac_loss.ac_model.state_dict(), self.SAVINGS_PATH + 'ac_model_trained_lowest_total_loss.pth')
                for k, ver_model in enumerate(self.verification_loss.verification_models):
                    torch.save(ver_model.state_dict(),
                               self.SAVINGS_PATH + 'ver_model%d_trained_lowest_total_loss.pth' % k)

        print('Finished Training!')

    def restart_adversary(self, epoch):
        """Re-initialize one adversary of the ensemble from scratch (round-robin) and put it into warm-up, during
        which it is trained but excluded from the generator loss.

        :param epoch: int
            The current epoch. Determines which ensemble member is restarted.
        """

        k = (epoch // self.ver_restart_every - 1) % self.ver_ensemble_size
        self.verification_loss.verification_models[k] = SiameseNetwork().cuda()
        self.optimizers_ver[k] = optim.Adam(self.verification_loss.verification_models[k].parameters(),
                                            lr=self.learning_rate)
        self.verification_loss.warmup_remaining[k] = self.ver_warmup_iters
        print('Restarted adversary %d at epoch %d (warm-up: %d iterations)' % (k, epoch, self.ver_warmup_iters))

    def run(self):
        # Call training/validation loop
        self.training_validation()
