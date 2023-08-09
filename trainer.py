import torch
from torch import nn, tensor, Tensor
from torch.utils.data import Dataset, DataLoader
import accelerate
import jsonargparse
import os
from datamodule import DataModule
from typing import Union, Dict, Any


class Trainer:
	"""
	Args:
		log_dir: directory to store all running utilities (logs, checkpoints, etc.)	

	"""
	def __init__(self, log_dir: str, model: nn.Module, datamodule: DataModule) -> None:
		self.log_dir = log_dir
		self.model = model
		self.datamodule = datamodule
	
	def fit(self):
		pass

	def train_step(self, batch: Any, batch_idx: int) -> Union[Tensor, Dict[str, Any]]:
		"""
		Args: 
		batch: a batch of training data
		batch_idx: batch index 
		Return:
			A tensor of Loss or a dictionary containing a 'loss' key and other keys
		"""
		raise NotImplementedError
		
	def val_step(self, batch: Any, batch_idx: int) -> Any:
		"""
		Args: 
		batch: a batch of validation data
		batch_idx: batch index 
		Return:
			Anything (metrics, results) you want to return during validation
		"""
		raise NotImplementedError

	def test_step(self, batch: Any, batch_idx: int) -> Any:
		"""
		Args: 
		batch: a batch of test data
		batch_idx: batch index 
		Return:
			Anything (metrics, results) you want to return during testing
		"""
		raise NotImplementedError


def configure_parser():
	parser = jsonargparse.ArgumentParser()
	parser.add_argument('-c', '--config', action=jsonargparse.ActionConfigFile)
	parser.add_class_arguments(accelerate.Accelerator, 'accelerator')
	parser.add_argument('--trainer', type=Trainer)


	return parser

def cli_main():
	parser = configure_parser()
	args = parser.parse_args()
	acc = accelerate.Accelerator(args.accelerator)

if __name__ == '__main__':
	cli_main()
