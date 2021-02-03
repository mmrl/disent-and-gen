import numpy as np
import torch
from torch.utils.data import Dataset


class Triplets(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        # self.n_samples = n_samples if n_samples is not None else len(dataset)

    @property
    def latents(self):
        return self.dataset.latent_values

    @property
    def imgs(self):
        return self.dataset.imgs

    def __len__(self):
        return len(self.dataset)
        # return self.n_samples

    def __getitem__(self, idx):
        img, z = self.imgs[idx], self.latents[idx]

        latent_size = z.shape[0]
        dim = np.random.choice(range(latent_size))

        # idx = self.latents.new_ones(len(self.latents), dtype=torch.bool)
        idx = np.ones(len(self.latents), dtype=np.bool)

        for i in range(latent_size):
            if i != dim:
                idx &= self.latents[:, i] == z[i]
            else:
                idx &= self.latents[:, i] != z[i]

        choices = idx.nonzero()[0]
        target_idx = np.random.choice(choices)

        idx = self.latents[:, dim] == self.latents[target_idx, dim]
        choices = idx.nonzero()[0]

        swap_idx = np.random.choice(choices)

        action = np.zeros(latent_size, dtype=np.float32)
        action[dim] = 1

        img = self.dataset.transform(img)
        swap = self.dataset.transform(self.imgs[swap_idx])
        target = self.dataset.transform(self.imgs[target_idx])

        input_imgs = torch.stack([img, swap], dim=0).contiguous()

        return (input_imgs, action), target
