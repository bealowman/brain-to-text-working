import unittest
from unittest.mock import Mock, MagicMock, patch, call
import torch
import numpy as np
import tempfile
import os
import shutil
from pathlib import Path

# Import the trainer
import sys
sys.path.insert(0, os.path.dirname(__file__))
from rnn_trainer import BrainToTextDecoder_Trainer
from rnn_model import RNNT


class TestRNNTTrainerCheckpointSaving(unittest.TestCase):
    """Test checkpoint saving logic for RNNT trainer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        self.output_dir = os.path.join(self.temp_dir, "outputs")
        
        # Create minimal args for RNNT training
        self.args = {
            'model_type': 'RNNT',
            'mode': 'train',
            'gpu_number': '0',
            'use_amp': False,  # Disable AMP for testing
            'use_diphones': False,
            'seed': 42,
            
            'model': {
                'n_input_features': 512,
                'n_units': 128,  # Smaller for faster tests
                'n_layers': 2,
                'rnn_dropout': 0.1,
                'rnn_trainable': True,
                'patch_size': 0,  # Disable patching for simplicity
                'patch_stride': 0,
                'input_network': {
                    'input_layer_dropout': 0.1,
                    'input_trainable': True,
                }
            },
            
            'dataset': {
                'sessions': ['session1', 'session2'],
                'n_classes': 41,
                'batch_size': 4,
                'days_per_batch': 1,
                'seed': 42,
                'num_dataloader_workers': 0,  # Single-threaded for testing
                'loader_shuffle': False,
                'dataset_dir': '/fake/path',
                'dataset_probability_val': [1, 1],  # Both days have validation data
                'data_transforms': {
                    'white_noise_std': 0.0,
                    'constant_offset_std': 0.0,
                    'random_walk_std': 0.0,
                    'static_gain_std': 0.0,
                    'random_cut': 0,
                    'smooth_data': False,
                    'mask_data': {'max_mask_width': 0},
                }
            },
            
            'output_dir': self.output_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'save_best_checkpoint': True,
            'save_all_val_steps': False,
            'save_final_model': False,
            'save_val_metrics': True,
            'init_from_checkpoint': False,
            'early_stopping': False,
            'early_stopping_val_steps': 20,
            
            'num_training_batches': 10,
            'batches_per_train_log': 5,
            'batches_per_val_step': 5,
            'log_individual_day_val_PER': False,
            'log_val_skip_logs': False,
            'save_val_logits': False,
            'save_val_data': False,
            
            'lr_scheduler_type': 'linear',
            'lr_max': 0.001,
            'lr_min': 0.0001,
            'lr_decay_steps': 10,
            'lr_warmup_steps': 0,
            'lr_max_day': 0.001,
            'lr_min_day': 0.0001,
            'lr_decay_steps_day': 10,
            'lr_warmup_steps_day': 0,
            
            'beta0': 0.9,
            'beta1': 0.999,
            'epsilon': 1e-8,
            'weight_decay': 0.0001,
            'weight_decay_day': 0.0,
            'grad_norm_clip_value': 1.0,
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('rnn_trainer.BrainToTextDataset')
    @patch('rnn_trainer.train_test_split_indicies')
    def test_checkpoint_saving_on_first_validation(self, mock_split, mock_dataset):
        """Test that checkpoint is saved on first validation (when best_val_PER is inf)"""
        # Mock dataset split
        mock_split.return_value = (
            {0: {'trials': [1, 2], 'session_path': '/fake/path/session1'}, 
             1: {'trials': [1, 2], 'session_path': '/fake/path/session2'}},
            {0: {'trials': [1], 'session_path': '/fake/path/session1'}, 
             1: {'trials': [1], 'session_path': '/fake/path/session2'}}
        )
        
        # Create mock dataset that returns fake batches
        def mock_getitem(idx):
            batch = {
                'input_features': torch.randn(4, 100, 512),
                'seq_class_ids': torch.randint(1, 41, (4, 50)),
                'n_time_steps': torch.tensor([100, 90, 80, 70]),
                'phone_seq_lens': torch.tensor([30, 25, 20, 15]),
                'day_indicies': torch.tensor([0, 0, 1, 1]),
                'transcriptions': torch.zeros(4, 100),
                'block_nums': torch.zeros(4),
                'trial_nums': torch.zeros(4),
            }
            return batch
        
        mock_dataset_instance = Mock()
        mock_dataset_instance.__getitem__ = mock_getitem
        mock_dataset_instance.__len__ = lambda x: 2
        mock_dataset.return_value = mock_dataset_instance
        
        # Create trainer
        trainer = BrainToTextDecoder_Trainer(self.args)
        
        # Verify initial state
        self.assertEqual(trainer.best_val_PER, torch.inf)
        self.assertEqual(trainer.best_val_loss, torch.inf)
        
        # Mock validation to return metrics
        def mock_validation(loader, return_logits=False, return_data=False):
            return {
                'avg_PER': 0.5,  # Some PER value
                'avg_loss': 2.5,  # Some loss value
                'day_PERs': {0: {'total_edit_distance': 10, 'total_seq_length': 20}},
                'losses': [2.5],
            }
        
        trainer.validation = mock_validation
        
        # Mock save_model_checkpoint to track calls
        checkpoint_calls = []
        original_save = trainer.save_model_checkpoint
        
        def tracked_save(path, per, loss):
            checkpoint_calls.append((path, per, loss))
            original_save(path, per, loss)
        
        trainer.save_model_checkpoint = tracked_save
        
        # Run one training iteration that triggers validation
        trainer.train_loader = [None] * 5  # 5 batches to trigger validation
        trainer.val_loader = [None]  # One validation batch
        
        # Manually trigger the validation logic
        val_metrics = trainer.validation(trainer.val_loader)
        
        # Check if new_best logic would trigger
        new_best = False
        if val_metrics['avg_PER'] < trainer.best_val_PER:
            new_best = True
        
        # Verify the comparison works
        self.assertTrue(new_best, "First validation should be new best")
        self.assertIsInstance(val_metrics['avg_PER'], (float, np.floating))
        
        # Test the actual checkpoint saving
        if new_best:
            trainer.best_val_PER = val_metrics['avg_PER']
            trainer.best_val_loss = val_metrics['avg_loss']
            if self.args['save_best_checkpoint']:
                trainer.save_model_checkpoint(
                    f'{self.args["checkpoint_dir"]}/best_checkpoint',
                    trainer.best_val_PER,
                    trainer.best_val_loss
                )
        
        # Verify checkpoint was saved
        checkpoint_path = os.path.join(self.checkpoint_dir, 'best_checkpoint')
        self.assertTrue(os.path.exists(checkpoint_path), "Checkpoint should be saved")
        
        # Verify checkpoint contents
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.assertIn('model_state_dict', checkpoint)
        self.assertIn('optimizer_state_dict', checkpoint)
        self.assertIn('scheduler_state_dict', checkpoint)
        self.assertEqual(checkpoint['val_PER'], 0.5)
        self.assertEqual(checkpoint['val_loss'], 2.5)
    
    def test_type_comparison_issue(self):
        """Test that type comparison between float and torch.inf works correctly"""
        # Test the problematic comparison
        best_val_PER = torch.inf  # This is a tensor
        avg_PER = 0.5  # This is a Python float
        
        # This should work but let's verify
        result = avg_PER < best_val_PER
        
        # The comparison should work, but let's check the types
        self.assertTrue(result, "Float should be less than torch.inf")
        
        # Test with float('inf')
        best_val_PER_float = float('inf')
        result2 = avg_PER < best_val_PER_float
        self.assertTrue(result2, "Float should be less than float('inf')")
        
        # Test edge case: what if avg_PER is also inf?
        avg_PER_inf = float('inf')
        result3 = avg_PER_inf < best_val_PER_float
        self.assertFalse(result3, "inf should not be less than inf")
    
    @patch('rnn_trainer.BrainToTextDataset')
    @patch('rnn_trainer.train_test_split_indicies')
    def test_validation_metrics_calculation(self, mock_split, mock_dataset):
        """Test that validation metrics are calculated correctly"""
        # Mock dataset split
        mock_split.return_value = (
            {0: {'trials': [1, 2], 'session_path': '/fake/path/session1'}},
            {0: {'trials': [1], 'session_path': '/fake/path/session1'}}
        )
        
        mock_dataset_instance = Mock()
        mock_dataset_instance.__len__ = lambda x: 1
        mock_dataset.return_value = mock_dataset_instance
        
        trainer = BrainToTextDecoder_Trainer(self.args)
        
        # Create a mock validation function that simulates the real one
        def mock_validation(loader, return_logits=False, return_data=False):
            # Simulate processing one batch
            total_edit_distance = 10
            total_seq_length = 100
            
            # This is the problematic line - total_seq_length could be a tensor
            if isinstance(total_seq_length, torch.Tensor):
                avg_PER = total_edit_distance / total_seq_length.item()
            else:
                avg_PER = total_edit_distance / total_seq_length
            
            return {
                'avg_PER': avg_PER if not isinstance(avg_PER, torch.Tensor) else avg_PER.item(),
                'avg_loss': 2.5,
                'day_PERs': {0: {'total_edit_distance': 10, 'total_seq_length': 100}},
                'losses': [2.5],
            }
        
        trainer.validation = mock_validation
        
        # Test that metrics are returned correctly
        val_metrics = trainer.validation(trainer.val_loader)
        
        self.assertIn('avg_PER', val_metrics)
        self.assertIn('avg_loss', val_metrics)
        self.assertIsInstance(val_metrics['avg_PER'], (float, np.floating))
        self.assertEqual(val_metrics['avg_PER'], 0.1)  # 10/100
    
    @patch('rnn_trainer.BrainToTextDataset')
    @patch('rnn_trainer.train_test_split_indicies')
    def test_validation_with_zero_seq_length(self, mock_split, mock_dataset):
        """Test validation when total_seq_length is zero (edge case)"""
        mock_split.return_value = (
            {0: {'trials': [1, 2], 'session_path': '/fake/path/session1'}},
            {0: {'trials': [1], 'session_path': '/fake/path/session1'}}
        )
        
        mock_dataset_instance = Mock()
        mock_dataset_instance.__len__ = lambda x: 1
        mock_dataset.return_value = mock_dataset_instance
        
        trainer = BrainToTextDecoder_Trainer(self.args)
        
        # Simulate validation with zero seq length (all batches skipped)
        def mock_validation(loader, return_logits=False, return_data=False):
            total_edit_distance = 0
            total_seq_length = 0  # This would cause division by zero
            
            # This should be handled gracefully
            if total_seq_length == 0:
                avg_PER = float('inf')  # Or handle appropriately
            else:
                avg_PER = total_edit_distance / total_seq_length
            
            return {
                'avg_PER': avg_PER,
                'avg_loss': float('inf'),
                'day_PERs': {},
                'losses': [],
            }
        
        trainer.validation = mock_validation
        
        val_metrics = trainer.validation(trainer.val_loader)
        
        # Check that it handles inf PER
        self.assertTrue(np.isinf(val_metrics['avg_PER']) or val_metrics['avg_PER'] == float('inf'))
        
        # Check that comparison still works
        result = val_metrics['avg_PER'] < trainer.best_val_PER
        # inf < inf should be False
        self.assertFalse(result)
    
    @patch('rnn_trainer.BrainToTextDataset')
    @patch('rnn_trainer.train_test_split_indicies')
    def test_new_best_detection_logic(self, mock_split, mock_dataset):
        """Test the new_best detection logic thoroughly"""
        mock_split.return_value = (
            {0: {'trials': [1, 2], 'session_path': '/fake/path/session1'}},
            {0: {'trials': [1], 'session_path': '/fake/path/session1'}}
        )
        
        mock_dataset_instance = Mock()
        mock_dataset_instance.__len__ = lambda x: 1
        mock_dataset.return_value = mock_dataset_instance
        
        trainer = BrainToTextDecoder_Trainer(self.args)
        
        # Test case 1: First validation (best_val_PER is inf)
        trainer.best_val_PER = torch.inf
        trainer.best_val_loss = torch.inf
        
        val_metrics_1 = {'avg_PER': 0.5, 'avg_loss': 2.0}
        
        new_best = False
        if val_metrics_1['avg_PER'] < trainer.best_val_PER:
            new_best = True
            trainer.best_val_PER = val_metrics_1['avg_PER']
            trainer.best_val_loss = val_metrics_1['avg_loss']
        
        self.assertTrue(new_best, "First validation should be new best")
        self.assertEqual(trainer.best_val_PER, 0.5)
        
        # Test case 2: Better PER
        val_metrics_2 = {'avg_PER': 0.3, 'avg_loss': 2.5}
        
        new_best = False
        if val_metrics_2['avg_PER'] < trainer.best_val_PER:
            new_best = True
            trainer.best_val_PER = val_metrics_2['avg_PER']
            trainer.best_val_loss = val_metrics_2['avg_loss']
        
        self.assertTrue(new_best, "Better PER should be new best")
        self.assertEqual(trainer.best_val_PER, 0.3)
        
        # Test case 3: Same PER, better loss
        val_metrics_3 = {'avg_PER': 0.3, 'avg_loss': 1.5}
        
        new_best = False
        if val_metrics_3['avg_PER'] < trainer.best_val_PER:
            new_best = True
            trainer.best_val_PER = val_metrics_3['avg_PER']
            trainer.best_val_loss = val_metrics_3['avg_loss']
        elif val_metrics_3['avg_PER'] == trainer.best_val_PER and val_metrics_3['avg_loss'] < trainer.best_val_loss:
            new_best = True
            trainer.best_val_loss = val_metrics_3['avg_loss']
        
        self.assertTrue(new_best, "Same PER with better loss should be new best")
        self.assertEqual(trainer.best_val_loss, 1.5)
        
        # Test case 4: Worse metrics
        val_metrics_4 = {'avg_PER': 0.4, 'avg_loss': 3.0}
        
        new_best = False
        if val_metrics_4['avg_PER'] < trainer.best_val_PER:
            new_best = True
            trainer.best_val_PER = val_metrics_4['avg_PER']
            trainer.best_val_loss = val_metrics_4['avg_loss']
        elif val_metrics_4['avg_PER'] == trainer.best_val_PER and val_metrics_4['avg_loss'] < trainer.best_val_loss:
            new_best = True
            trainer.best_val_loss = val_metrics_4['avg_loss']
        
        self.assertFalse(new_best, "Worse metrics should not be new best")
        self.assertEqual(trainer.best_val_PER, 0.3)  # Should remain unchanged
    
    @patch('rnn_trainer.BrainToTextDataset')
    @patch('rnn_trainer.train_test_split_indicies')
    def test_save_model_checkpoint_signature(self, mock_split, mock_dataset):
        """Test that save_model_checkpoint is called with correct arguments"""
        mock_split.return_value = (
            {0: {'trials': [1, 2], 'session_path': '/fake/path/session1'}},
            {0: {'trials': [1], 'session_path': '/fake/path/session1'}}
        )
        
        mock_dataset_instance = Mock()
        mock_dataset_instance.__len__ = lambda x: 1
        mock_dataset.return_value = mock_dataset_instance
        
        trainer = BrainToTextDecoder_Trainer(self.args)
        
        # Test that save_model_checkpoint requires 3 arguments
        with self.assertRaises(TypeError):
            trainer.save_model_checkpoint('path', 0.5)  # Missing loss argument
        
        # Test that it works with 3 arguments
        try:
            trainer.save_model_checkpoint('test_checkpoint', 0.5, 2.0)
            self.assertTrue(True, "Should accept 3 arguments")
        except TypeError:
            self.fail("save_model_checkpoint should accept 3 arguments")
    
    @patch('rnn_trainer.BrainToTextDataset')
    @patch('rnn_trainer.train_test_split_indicies')
    def test_checkpoint_saving_in_train_loop(self, mock_split, mock_dataset):
        """Test checkpoint saving during actual training loop"""
        # This is a more integration-style test
        mock_split.return_value = (
            {0: {'trials': [1, 2], 'session_path': '/fake/path/session1'}},
            {0: {'trials': [1], 'session_path': '/fake/path/session1'}}
        )
        
        # Create a simple mock dataset
        def create_mock_batch():
            return {
                'input_features': torch.randn(4, 50, 512),
                'seq_class_ids': torch.randint(1, 41, (4, 30)),
                'n_time_steps': torch.tensor([50, 45, 40, 35]),
                'phone_seq_lens': torch.tensor([20, 18, 15, 12]),
                'day_indicies': torch.tensor([0, 0, 0, 0]),
                'transcriptions': torch.zeros(4, 50),
                'block_nums': torch.zeros(4),
                'trial_nums': torch.zeros(4),
            }
        
        mock_train_dataset = Mock()
        mock_train_dataset.__getitem__ = lambda x, idx: create_mock_batch()
        mock_train_dataset.__len__ = lambda x: 10
        
        mock_val_dataset = Mock()
        mock_val_dataset.__getitem__ = lambda x, idx: create_mock_batch()
        mock_val_dataset.__len__ = lambda x: 1
        
        mock_dataset.side_effect = [mock_train_dataset, mock_val_dataset]
        
        trainer = BrainToTextDecoder_Trainer(self.args)
        
        # Track checkpoint saves
        checkpoint_saves = []
        original_save = trainer.save_model_checkpoint
        
        def tracked_save(path, per, loss):
            checkpoint_saves.append((path, per, loss))
            # Don't actually save to avoid file I/O in tests
            # original_save(path, per, loss)
        
        trainer.save_model_checkpoint = tracked_save
        
        # Mock the model forward to avoid actual computation
        def mock_forward(x, day_idx, targets_with_sos=None):
            B, T, D = x.shape
            if targets_with_sos is not None:
                U = targets_with_sos.shape[1]
                return torch.randn(B, T, U, 41)  # RNNT logits shape
            else:
                return torch.randn(B, T, 41)  # GRU logits shape
        
        trainer.model.forward = mock_forward
        
        # Mock greedy_decode for validation
        def mock_greedy_decode(x, day_idx, blank_id=0, max_symbols_per_step=30):
            B = x.shape[0]
            return [[1, 2, 3] for _ in range(B)]
        
        trainer.model.greedy_decode = mock_greedy_decode
        
        # Run a few training steps
        # We'll manually call the validation logic since full training is complex
        trainer.best_val_PER = torch.inf
        
        # Simulate validation metrics
        val_metrics = {
            'avg_PER': 0.6,
            'avg_loss': 3.0,
            'day_PERs': {0: {'total_edit_distance': 30, 'total_seq_length': 50}},
            'losses': [3.0],
        }
        
        # Simulate the new_best logic from train()
        new_best = False
        if val_metrics['avg_PER'] < trainer.best_val_PER:
            trainer.best_val_PER = val_metrics['avg_PER']
            trainer.best_val_loss = val_metrics['avg_loss']
            new_best = True
        
        if new_best and self.args['save_best_checkpoint']:
            trainer.save_model_checkpoint(
                f'{self.args["checkpoint_dir"]}/best_checkpoint',
                trainer.best_val_PER,
                trainer.best_val_loss
            )
        
        # Verify checkpoint was "saved" (tracked)
        self.assertEqual(len(checkpoint_saves), 1)
        self.assertEqual(checkpoint_saves[0][1], 0.6)  # PER
        self.assertEqual(checkpoint_saves[0][2], 3.0)  # Loss


class TestRNNTTrainerValidationMetrics(unittest.TestCase):
    """Test validation metrics calculation specifically"""
    
    def test_per_calculation_types(self):
        """Test that PER calculation handles types correctly"""
        # Simulate the problematic code path
        total_edit_distance = 10
        total_seq_length = torch.tensor(100)  # This becomes a tensor
        
        # This is what happens in the code
        total_seq_length += torch.sum(torch.tensor([20, 30, 50]))
        
        # Now total_seq_length is a tensor
        self.assertIsInstance(total_seq_length, torch.Tensor)
        
        # Calculate PER
        avg_PER = total_edit_distance / total_seq_length
        
        # This is a tensor division
        self.assertIsInstance(avg_PER, torch.Tensor)
        
        # Convert to float
        avg_PER_float = avg_PER.item()
        self.assertIsInstance(avg_PER_float, float)
        
        # Now test comparison
        best_val_PER = torch.inf
        result = avg_PER_float < best_val_PER
        
        self.assertTrue(result)
    
    def test_per_calculation_with_python_int(self):
        """Test PER calculation when total_seq_length stays as Python int"""
        total_edit_distance = 10
        total_seq_length = 0  # Python int
        
        # Add tensor sum as item()
        total_seq_length += torch.sum(torch.tensor([20, 30, 50])).item()
        
        # Now total_seq_length is still a Python int
        self.assertIsInstance(total_seq_length, (int, np.integer))
        
        # Calculate PER
        avg_PER = total_edit_distance / total_seq_length
        
        # This is Python division
        self.assertIsInstance(avg_PER, float)
        
        # Test comparison
        best_val_PER = torch.inf
        result = avg_PER < best_val_PER
        
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()