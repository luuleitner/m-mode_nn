import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import utils.logging_config as logconf
logger = logconf.get_logger("MAIN")
from utils import wandb_utils


class ClassifierTrainer():
    def __init__(self, model=None, Xy_train=None, Xy_val=None, config=None):

        self._labeltype = config.model.label    # defines the label_logic to prediction behaviour

        # Push Model to Device (GPU)
        self._model = model.to(config.training.device)

        # Watch model with wandb
        wandb_utils.watch_model(model)

        # Define Loss and Optimizer
        self._criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.training.lr)


        ##  Start Training
        #   Epoch Training
        for epoch in range(config.training.epochs):
            # Start Forward Pass
            self._model.train()

            # Set initial coditions
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            all_preds = []
            all_labels = []

            #   Batch Training
            for batch_idx, (X, y) in enumerate(tqdm(Xy_train, desc=f"Epoch {epoch + 1}/{config.training.epochs} - Training", leave=False)):

                if self._labeltype == 'sequence label_logic':
                    # If Prediction is on sequence labels then the majority vote over all timesteps in the sequence needs to be computed
                    y = y.squeeze(-1)
                    y = torch.mode(y, dim=1).values.long()  # returns the majority mode (mode) with int64 (long)
                else:
                    logger.error('Wrong label_logic type.')
                    break

                X, y = X.to(config.training.device), y.to(config.training.device)

                #   Prediction, Loss and Backward Pass
                optimizer.zero_grad()
                preds = self._model(X)
                loss = self._criterion(preds, y)
                loss.backward()
                optimizer.step()

                #   Metric
                pred_labels = torch.argmax(preds, dim=1)
                correct = (pred_labels == y).sum().item()
                total = y.size(0)
                epoch_loss += loss.item()
                epoch_correct += correct
                epoch_total += total

                #   Send Predictions and Labels to Wandb
                # print(f"Predictions {pred_labels}")
                # print(f'LABEL: {y}')
                all_preds.append(pred_labels.detach().cpu())
                all_labels.append(y.detach().cpu())

            avg_loss = epoch_loss / len(Xy_train)
            avg_acc = epoch_correct / epoch_total

            # Combine predictions and labels for logging
            all_preds_tensor = torch.cat(all_preds)
            all_labels_tensor = torch.cat(all_labels)

            # Validation
            avg_val_loss, avg_val_acc = self._validate(Xy_val, config)

            logger.info(f"Epoch {epoch + 1}/{config.training.epochs} | "
                        f"Train, Val Loss: {avg_loss:.4f}, {avg_val_loss:.4f} | "
                        f"Train, Val Accuracy: {avg_acc * 100:.2f}%, {avg_val_acc * 100:.2f}%")

            wandb_utils.log_training_metrics(
                avg_loss, avg_acc, avg_val_loss, avg_val_acc,
                epoch,
                all_preds_tensor,
                all_labels_tensor
            )

        wandb_utils.finish()


    def _validate(self, Xy_val, config):
        self._model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(tqdm(Xy_val, desc="Validating", leave=False)):
                if self._labeltype == 'sequence label_logic':
                    y = y.squeeze(-1)
                    y = torch.mode(y, dim=1).values.long()

                X, y = X.to(config.training.device), y.to(config.training.device)

                preds = self._model(X)
                loss = self._criterion(preds, y)

                pred_labels = torch.argmax(preds, dim=1)
                correct = (pred_labels == y).sum().item()
                total = y.size(0)

                val_loss += loss.item()
                val_correct += correct
                val_total += total

        avg_loss = val_loss / len(Xy_val)
        avg_acc = val_correct / val_total

        return avg_loss, avg_acc


    def _save_checkpoint(self, epoch, model, optimizer, val_loss, filename):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))
        logger.info(f"Checkpoint saved: {filename}")

    @property
    def model(self):
        return self._model