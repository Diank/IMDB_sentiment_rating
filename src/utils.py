import matplotlib.pyplot as plt

def train_val_plots(train_losses, val_losses, train_accuracies, val_accuracies, train_maes, val_maes):
    
    plt.figure(figsize=(18, 5))  
    
    # LOSS
    plt.subplot(1, 3, 1)  
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # ACCURACY
    plt.subplot(1, 3, 2) 
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    # MAE
    plt.subplot(1, 3, 3)  
    plt.plot(train_maes, label='Train MAE')
    plt.plot(val_maes, label='Val MAE')
    plt.legend()
    plt.title('MAE over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')

    plt.tight_layout()  
    plt.show()
