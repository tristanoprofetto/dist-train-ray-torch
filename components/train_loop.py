import argparse
import torch
import torch.nn.functional as F
from ray import train
from ray.air import session


def train_model(model, train_loader, optimizer, epoch, device):
    """Executes training iteration for a given epoch
    
    Args:
        model (torch.nn.Module): PyTorch model being trained
        train_loader (torch.utils.data.DataLoader): training data loader
        optimizer (torch.optim.Optimizer): PyTorch optimizer
        epoch (int):  current training iteration
    
    Returns:
        None
    """
    model.train()
    ddp_loss = torch.zeros(2).to()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        # generate predictions for the given batch
        preds = model(x)
        # compute the loss with respect to the target variable
        loss = F.nll_loss(preds, y, reduction='sum')
        train.torch.backward(loss)
        # update model parameters
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(x)
    # print and record metrics
    print(f'Epoch: {epoch} \tTrain Loss: {ddp_loss[0] / ddp_loss[1]}')
    session.report(
        metrics={'epoch': epoch, 'train_loss': ddp_loss[0].tolist() / ddp_loss[1].tolist()},
        checkpoint=train.torch.TorchCheckpoint.from_state_dict(model.state_dict())
    )


def evaluate_model(model, test_loader):
    """
    Executes evaluation iteration for a given epoch
    Args:
        model (torch.nn.Module): PyTorch model being trained
        test_loader (torch.utils.data.DataLoader): test data loader
    
    Returns:
        None
    """
    model.eval()
    ddp_loss = torch.zeros(2).to()
    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x)
            loss = F.nll_loss(preds, y, reduction='sum')
            ddp_loss[0] += loss.item()
            ddp_loss[1] += len(x)
    print(f'Test Loss: {ddp_loss[0] / ddp_loss[1]}')
    session.report(
        metrics={'test_loss': ddp_loss[0].tolist() / ddp_loss[1].tolist()},
        checkpoint=train.torch.TorchCheckpoint.from_state_dict(model.state_dict())
    )


def get_args():
    """
    Get addtional command-line arguments for training job
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    return parser.parse_args()
    

if __name__ == "__main__":
    from torch.optim import SGD
    from dataset import get_data_loaders
    from model import DigitClassifier
    # get command line arguments
    args = get_args()
    # get data loaders
    train_loader, test_loader = get_data_loaders(batch_size=args.batch_size)
    # instantiate model and optimizer
    model = DigitClassifier()
    optimizer = SGD(model.parameters(), lr=args.learning_rate)
    # move model and data to GPU
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    for epoch in range(1, args.epochs +1):
        train_model(model, train_loader, optimizer, epoch, device)
        evaluate_model(model, test_loader)



    
