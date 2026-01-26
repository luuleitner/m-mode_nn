import os
os.environ["WANDB_API_KEY"] = "c1842818a3c8db66c3f75d77b4640749e64cdc0d"
import wandb


import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

training_loss_over_epochs = []
validation_loss_over_epochs = []


def initialize_wandb(config_dict):
    wandb.init(project=config_dict['wandb']['project'], name=config_dict['wandb']['name'], config=config_dict)
    wandb.run.notes = config_dict.get("notes", "")
    wandb.run.tags = config_dict.get("tags", [])
    return wandb.config

def log_image(key, tensor, caption=""):
    wandb.log({key: wandb.Image(tensor.numpy(), caption=caption)})

def log_histogram(key, tensor):
    wandb.log({key: wandb.Histogram(tensor.detach().numpy())})


def watch_model(model):
    wandb.watch(model, log="all", log_freq=1)

def _to_float(x):
    return x.item() if isinstance(x, torch.Tensor) else float(x)


#######################################
# ------ Logging AE

def log_training_metrics_AE(avg_loss, avg_val_loss, epoch, inputs, recons, latents, current_lr,
                            model=None, n_samples=8, log_freq=5):
    wandb.log({
        "AE avg training loss": _to_float(avg_loss),
        "AE avg validation loss": _to_float(avg_val_loss),
        "AE epoch": epoch + 1,
        'learning_rate': current_lr
    })

    if inputs is not None and recons is not None:
        mse, rmse, pearson = _reconstruction_metrics(inputs, recons)
        wandb.log({
            "AE_reconstruction_MSE": mse,
            "AE_reconstruction_RMSE": rmse,
            "AE_reconstruction_pearson": pearson,
        })

    # Update loss curves
    training_loss_over_epochs.append(_to_float(avg_loss))
    validation_loss_over_epochs.append(_to_float(avg_val_loss))

    # Loss curve plot
    fig, ax = plt.subplots()
    ax.plot(range(1, len(training_loss_over_epochs) + 1), training_loss_over_epochs, label="Training Loss")
    ax.plot(range(1, len(validation_loss_over_epochs) + 1), validation_loss_over_epochs, label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Autoencoder Training vs Validation Loss")
    ax.legend()
    plt.tight_layout()
    wandb.log({"AE_loss_curve": wandb.Image(fig)})
    plt.close(fig)

    # Log histogram of latent values
    if latents is not None:
        wandb.log({"AE_latent_histogram": wandb.Histogram(latents.detach().cpu().numpy())})

    # Log kernel histograms for first conv layer
    if model is not None:
        for name, param in model.named_parameters():
            if "weight" in name and "conv" in name:
                wandb.log({f"{name}_hist": wandb.Histogram(param.detach().cpu().numpy())})


def _reconstruction_metrics(input_data, recon_data):
    inputs_np = input_data.cpu().numpy()
    recons_np = recon_data.cpu().numpy()
    mse = np.mean((inputs_np - recons_np) ** 2)
    rmse = np.sqrt(mse)
    pearson = np.mean([np.corrcoef(inputs_np[i], recons_np[i])[0, 1] for i in range(inputs_np.shape[0])])
    return mse, rmse, pearson



def log_training_metrics_classifier(avg_loss, avg_acc, avg_val_loss, avg_val_acc, epoch, pred_labels, y):
    # Ensure tensors are detached and on CPU
    preds = pred_labels.detach().cpu().numpy()
    labels = y.detach().cpu().numpy()

    wandb.log({
        "avg training loss": _to_float(avg_loss),
        "avg training accuracy": _to_float(avg_acc),
        "avg validation loss": _to_float(avg_val_loss),
        "avg validation accuracy": _to_float(avg_val_acc),
        "epoch": epoch + 1
    })


    # Append to loss history
    training_loss_over_epochs.append(_to_float(avg_loss))
    validation_loss_over_epochs.append(_to_float(avg_val_loss))

    # Plot training and validation loss
    fig, ax = plt.subplots()
    ax.plot(range(1, len(training_loss_over_epochs) + 1), training_loss_over_epochs, label="Training Loss")
    ax.plot(range(1, len(validation_loss_over_epochs) + 1), validation_loss_over_epochs, label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    plt.tight_layout()
    wandb.log({"loss_curve": wandb.Image(fig)})
    plt.close(fig)

    # Log confusion matrix image
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    plt.close(fig)

    # Log interactive confusion matrix
    class_names = ["0", "1", "2", "3"]
    wandb_cm = wandb.plot.confusion_matrix(
        probs=None,
        y_true=labels,
        preds=preds,
        class_names=class_names
    )
    wandb.log({"interactive_confusion_matrix": wandb_cm})

    # Count bar plot (True Labels)
    true_label_counts = pd.Series(labels).value_counts().sort_index()
    true_table = wandb.Table(
        data=[[str(i), int(true_label_counts.get(i, 0))] for i in range(4)],
        columns=["Class", "Count"]
    )
    wandb.log({
        "True Label Distribution": wandb.plot.bar(
            true_table,
            "Class",
            "Count",
            title="True Label Distribution"
        )
    })

    # Count bar plot (Predicted Labels)
    pred_label_counts = pd.Series(preds).value_counts().sort_index()
    pred_table = wandb.Table(
        data=[[str(i), int(pred_label_counts.get(i, 0))] for i in range(4)],
        columns=["Class", "Count"]
    )
    wandb.log({
        "Predicted Label Distribution": wandb.plot.bar(
            pred_table,
            "Class",
            "Count",
            title="Predicted Label Distribution"
        )
    })

    # Heatmap of label_logic/prediction pairs (acts like a scatter but with density)
    joint_df = pd.DataFrame({
        "True Label": labels,
        "Predicted Label": preds
    })
    joint_counts = joint_df.groupby(["True Label", "Predicted Label"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(joint_counts, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Label vs Prediction Grid")
    plt.tight_layout()
    wandb.log({"label_prediction_grid": wandb.Image(fig)})
    plt.close(fig)



def finish():
    global training_loss_over_epochs, validation_loss_over_epochs
    training_loss_over_epochs.clear()
    validation_loss_over_epochs.clear()
    wandb.finish()

