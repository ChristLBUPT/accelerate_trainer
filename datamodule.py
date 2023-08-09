from torch.utils.data import Dataset, DataLoader

class DataModule:
	"""
	
	Args:
		train_dataset: the pytorch's `Dataset` you are using for training
		val_dataset: the pytorch's `Dataset` you are using for validation
		test_dataset: the pytorch's `Dataset` you are using for testing
	"""
	def __init__(self, train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset):
		self.train_dataset = train_dataset
		self.val_dataset = val_dataset
		self.test_dataset = test_dataset
	
	def train_dataloader(self) -> DataLoader:
		"""
		returns a dataloader of train dataset
		"""
		raise NotImplementedError


	def train_dataloader(self) -> DataLoader:
		"""
		returns a dataloader of train dataset
		"""
		raise NotImplementedError

	def train_dataloader(self) -> DataLoader:
		"""
		returns a dataloader of train dataset
		"""
		raise NotImplementedError