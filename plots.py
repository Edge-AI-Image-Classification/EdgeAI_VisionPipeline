import matplotlib.pyplot as plt
import json

def plotmetrics(logs_path="plottinglogs.json"):
    # 1. Load JSON logs
    with open(logs_path, "r") as f:
        logs = json.load(f)

    epochs       = logs["epochs"]
    train_loss   = logs["train_loss"]
    train_acc    = logs["train_acc"]
    val_loss     = logs["val_loss"]
    val_acc      = logs["val_acc"]

    #plotting loss over n epochs
    plt.figure()
    plt.plot(epochs, train_loss, label='Training Loss', color='blue')
    plt.plot(epochs, val_loss, label='Validation Loss', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True)
    plt.show()

    #plotting accuracy over n epochs
    plt.figure()
    plt.plot(epochs, train_acc, label='Training Accuracy', color='blue')
    plt.plot(epochs, val_acc, label='Validation Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Over Epochs')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plotmetrics("plottinglogs.json")