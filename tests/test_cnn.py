import os
import sys
import tempfile
import unittest

import torch
import torch.nn as nn
import torch.optim as optim


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
UTILS_DIR = os.path.join(ROOT_DIR, 'utils')

for path in (MODELS_DIR, UTILS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import cnn


class CnnTests(unittest.TestCase):
    def test_forward_returns_flat_batch_predictions(self):
        model = cnn.Net(256)
        model.conv1 = nn.Identity()
        model.conv2 = nn.Identity()
        model.conv3 = nn.Identity()
        model.conv4 = nn.Identity()
        model.conv5 = nn.Identity()
        model.fc1 = nn.Identity()
        model.fc2 = nn.Identity()

        outputs = model(torch.randn(4, 1, 1, 1))

        self.assertEqual(tuple(outputs.shape), (4,))

    def test_checkpoint_round_trip_restores_state(self):
        model = nn.Linear(2, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'ckpt.pth.tar')
            original_weight = model.weight.detach().clone()

            cnn.save_checkpoint(model, optimizer, scheduler, 1.5, 3, checkpoint_path)

            with torch.no_grad():
                model.weight.add_(10.0)

            best_loss, start_epoch = cnn.load_checkpoint(
                model,
                optimizer,
                scheduler=scheduler,
                checkpoint_path=checkpoint_path,
                map_location='cpu',
            )

        self.assertAlmostEqual(best_loss, 1.5)
        self.assertEqual(start_epoch, 4)
        self.assertTrue(torch.allclose(model.weight, original_weight))

    def test_load_checkpoint_overwrites_current_weights_with_saved_best_state(self):
        saved_model = nn.Linear(2, 1)
        current_model = nn.Linear(2, 1)
        saved_optimizer = optim.Adam(saved_model.parameters(), lr=0.1)
        current_optimizer = optim.Adam(current_model.parameters(), lr=0.1)
        saved_scheduler = optim.lr_scheduler.StepLR(saved_optimizer, step_size=1)
        current_scheduler = optim.lr_scheduler.StepLR(current_optimizer, step_size=1)

        with torch.no_grad():
            saved_model.weight.fill_(0.25)
            saved_model.bias.fill_(0.10)
            current_model.weight.fill_(0.75)
            current_model.bias.fill_(0.50)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'ckpt.pth.tar')
            cnn.save_checkpoint(
                saved_model,
                saved_optimizer,
                saved_scheduler,
                2.0,
                1,
                checkpoint_path,
            )
            cnn.load_checkpoint(
                current_model,
                current_optimizer,
                scheduler=current_scheduler,
                checkpoint_path=checkpoint_path,
                map_location='cpu',
            )

        self.assertTrue(torch.allclose(current_model.weight, saved_model.weight))
        self.assertTrue(torch.allclose(current_model.bias, saved_model.bias))


if __name__ == '__main__':
    unittest.main()
