import torch
from torch import nn, tensor, Tensor
import accelerate
import jsonargparse
import os

class Trainer:
	"""
	Args:
		log_dir: directory to store all running utilities (logs, checkpoints, etc.)	


	"""
	def __init__(self, log_dir: str, ) -> None:
		self.log_dir = log_dir
		pass

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
