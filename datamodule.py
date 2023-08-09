from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

class DataModule:
    """

    Args:
    train_dataset: the pytorch's `Dataset` you are using for training
    val_dataset: the pytorch's `Dataset` you are using for validation
    test_dataset: the pytorch's `Dataset` you are using for testing
    customize_dataloaders: whether or not to use user-defined dataloaders,
        if set to `True`, you should subclass the `DataModule` class 
        and implement `train_dataloader`, `val_dataloader`, `test_dataloader` methods
    dataloader_kwargs: kwargs to be passed to the dataloader, format is `<split_name(train, val, test)>_<kwarg_name>`,
        for example: `train_batch_size`, `train_num_workers`, `val_batch_size` 
    """
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset = None,
        test_dataset: Dataset = None,
        customize_dataloaders: bool = False,
        **dataloader_kwargs,
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.customize_dataloaders = customize_dataloaders
        split_names = ["train", "val", "test"]
        kwargs_for_dataloaders = [
            "batch_size",
            "num_workers",
            "pin_memory",
            "drop_last",
            "timeout",
            "sampler",
            "pin_memory_device",
            "prefetch_factor"
        ]
        seperated_dataloader_kwargs = defaultdict(dict) 
        # seperated dataloader kwargs, 
        # {'train': {'batch_size': xx, 'num_workers': xx}, 'val': {'batch_size': xx, 'num_workers': xx}, 'test': {'batch_size': xx, 'num_workers': xx}} 
        for split_name in split_names: 
            for dataloader_kwarg in kwargs_for_dataloaders:
                if f"{split_name}_{dataloader_kwarg}" in dataloader_kwargs:
                    seperated_dataloader_kwargs[split_name][dataloader_kwarg] = dataloader_kwargs[f"{split_name}_{dataloader_kwarg}"]
        self.seperated_dataloader_kwargs = seperated_dataloader_kwargs


    def train_dataloader(self) -> DataLoader:
        """
        returns a dataloader of train dataset
        """
        if self.customize_dataloaders:
            raise NotImplementedError
        else:
            return DataLoader(self.train_dataset, **self.seperated_dataloader_kwargs['train'])



    def val_dataloader(self) -> DataLoader:
        """
        returns a dataloader of validation dataset
        """
        if self.customize_dataloaders:
            raise NotImplementedError
        else:
            return DataLoader(self.val_dataset, **self.seperated_dataloader_kwargs['val'])

    def test_dataloader(self) -> DataLoader:
        """
        returns a dataloader of test dataset
        """
        if self.customize_dataloaders:
            raise NotImplementedError
        else:
            return DataLoader(self.test_dataset, **self.seperated_dataloader_kwargs['test'])