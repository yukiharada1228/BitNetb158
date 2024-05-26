import torch

def calculate_mean_std(dataloader, device):
    h, w = 0, 0
    chsum = None
    for batch_idx, (inputs, _) in enumerate(dataloader):
        inputs = inputs.to(device)
        if batch_idx == 0:
            h, w = inputs.size(2), inputs.size(3)
            chsum = inputs.sum(dim=(0, 2, 3), keepdim=True)
        else:
            chsum += inputs.sum(dim=(0, 2, 3), keepdim=True)
    mean = chsum / len(dataloader.dataset) / h / w
    chsum = None
    for batch_idx, (inputs, _) in enumerate(dataloader):
        inputs = inputs.to(device)
        if batch_idx == 0:
            chsum = (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
        else:
            chsum += (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
    std = torch.sqrt(chsum / (len(dataloader.dataset) * h * w - 1))
    mean = mean.view(-1).cpu().numpy()
    std = std.view(-1).cpu().numpy()
    return mean, std
