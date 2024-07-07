import os
import matplotlib.pyplot as plt

# Function to plot loss, validation and accuracy charts, and IoU and F1 Score charts
def plot_charts(train_losses, train_accuracies, val_losses, val_accuracies, ious, f1s, val_interval):
    epochs = len(train_losses)
    val_epochs = range(val_interval, epochs + 1, val_interval)

    plt.figure(figsize=(12, 12))

    # Plot Training and Validation Loss
    plt.subplot(2, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(val_epochs, val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Plot Training and Validation Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(val_epochs, val_accuracies, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot IoU and F1 Score
    plt.subplot(2, 2, 3)
    plt.plot(val_epochs, ious, label='IoU')
    plt.plot(val_epochs, f1s, label='F1')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('Validation IoU and F1 Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Function to save plots
def save_plots(train_losses, train_accuracies, val_losses, val_accuracies, ious, f1s, val_interval, epoch=None, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    epochs = len(train_losses)
    val_epochs = range(val_interval, epochs + 1, val_interval)

    plt.figure(figsize=(12, 12))

    # Plot Training and Validation Loss
    plt.subplot(2, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(val_epochs, val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Plot Training and Validation Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(val_epochs, val_accuracies, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot IoU and F1 Score
    plt.subplot(2, 2, 3)
    plt.plot(val_epochs, ious, label='IoU')
    plt.plot(val_epochs, f1s, label='F1')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('Validation IoU and F1 Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if epoch is not None:
        plt.savefig(os.path.join(output_dir, f'loss_and_scores_epoch_{epoch + 1}.png'))
    else:
        plt.savefig(os.path.join(output_dir, 'loss_and_scores_final.png'))
    plt.close()
