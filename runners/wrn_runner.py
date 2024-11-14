from typing import Optional
import torch.nn.functional as F
from calibration.ece import ECELoss
from guided_diffusion import dist_util


class WRNTrainer:
    def __init__(self, model, optim, scheduler, train_data, val_data):
        self.model = model
        self.optim = optim
        self.train_data = train_data
        self.val_data = val_data
        self.scheduler = scheduler
        self.device = dist_util.dev()

    def _run_epoch(self, data_loader, train=True):
        total_loss = 0
        total_acc = 0
        total_ece = 0

        ECE = ECELoss()

        for x, y in data_loader:
            x, y = x.to(self.device), y["y"].to(self.device)
            logits = self.model(x)
            loss = F.cross_entropy(logits, y)
            acc = (logits.argmax(1) == y).float().mean()
            ece = ECE(logits, y)

            if train:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            total_loss += loss.item()
            total_acc += acc.item()
            total_ece += ece.item()

        avg_loss = total_loss / len(data_loader)
        avg_acc = total_acc / len(data_loader)
        avg_ece = total_ece / len(data_loader)

        return avg_loss, avg_acc, avg_ece

    def run_loop(self, num_epochs, eval_freq: int, save_callback: Optional[callable] = None):
        for epoch in range(num_epochs):
            self.model.train()
            train_loss, train_acc, train_ece = self._run_epoch(self.train_data, train=True)

            self.scheduler.step()

            val_print = None
            if epoch % eval_freq == 0:
                self.model.eval()
                val_loss, val_acc, val_ece = self._run_epoch(self.val_data, train=False)
                val_print = f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, ECE: {val_ece:.4f}"

                # Call checkpoint callback if provided and validation improves
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    if save_callback is not None:
                        metrics = {
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "train_acc": train_acc,
                            "train_ece": train_ece,
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                            "val_ece": val_ece,
                            "learning_rate": self.optim.param_groups[0]["lr"],
                        }
                        save_callback(self.model, self.optim, self.scheduler, metrics)

            print(
                f"Epoch {epoch + 1}/{num_epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, ECE: {train_ece:.4f}, lr: {self.optim.param_groups[0]['lr']}"
                + (f", {val_print}" if val_print else "")
            )
