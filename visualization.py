import os
import pandas as pd
import matplotlib.pyplot as plt


RESULTS = "./results/res_net_18"
def plot_history(history: dict, output_dir: str, prefix: str = ""):
    """Plots training/validation metrics from history dict and saves figures.

    Expected keys: "train_loss", "train_acc", "val_loss", "val_acc" etc.
    """
    os.makedirs(output_dir, exist_ok=True)

    if "train_loss" in history and "val_loss" in history:
        plt.figure()
        plt.plot(history["train_loss"], label="train loss")
        plt.plot(history["val_loss"], label="val loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{prefix}loss.png"))
        plt.close()

    if "train_acc" in history and "val_acc" in history:
        plt.figure()
        plt.plot(history["train_acc"], label="train acc")
        plt.plot(history["val_acc"], label="val acc")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{prefix}acc.png"))
        plt.close()


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(RESULTS, "history.csv"))
    history = df.to_dict(orient='list')
    plot_history(history=history, output_dir=RESULTS)