from data.unaligned_dataset import UnalignedDataset
import torch


def create_data_loader(opt, phase, batch_size, shuffle, num_workers):
    dataset = UnalignedDataset(opt, phase)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
