from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from calibration.ece import ECELoss


class WRNTrainer:
    def __init__(self, model, optim, train_data, val_data, device):
        self.model = model
        self.optim = optim
        self.train_data = train_data
        self.val_data = val_data
        self.device = device

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        total_acc = 0
        total_ece = 0

        ECE = ECELoss()

        for x, y in tqdm(self.train_data, desc="Training"):
            x, y = x.to(self.device), y["y"].to(self.device)

            logits = self.model(x)
            loss = F.cross_entropy(logits, y)
            acc = (logits.argmax(1) == y).float().mean()
            ece = ECE(logits, y)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            total_loss += loss.item()
            total_acc += acc.item()
            total_ece += ece.item()

        avg_loss = total_loss / len(self.train_data)
        avg_acc = total_acc / len(self.train_data)
        avg_ece = total_ece / len(self.train_data)
        return avg_loss, avg_acc, avg_ece

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        total_acc = 0
        total_ece = 0

        ECE = ECELoss()

        with torch.no_grad():
            for x, y in tqdm(self.val_data, desc="Evaluating"):
                x, y = x.to(self.device), y["y"].to(self.device)
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
                acc = (logits.argmax(1) == y).float().mean()
                ece = ECE(logits, y)

                total_loss += loss.item()
                total_acc += acc.item()
                total_ece += ece.item()

        avg_loss = total_loss / len(self.val_data)
        avg_acc = total_acc / len(self.val_data)
        avg_ece = total_ece / len(self.val_data)
        return avg_loss, avg_acc, avg_ece

    def run_loop(self, num_epochs, eval_every_epoch: int):
        for epoch in tqdm(range(num_epochs)):
            self.train_one_epoch()

            if epoch % eval_every_epoch == 0:
                self.evaluate()
