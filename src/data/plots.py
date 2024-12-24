import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve
import seaborn as sns

def plot_and_save_all_metrics(y_true, y_pred, output_dir, task_number, decode_labels):
    # Convert string labels to numeric (binary)
    unique_labels = list(set(y_true))
    y_true_bin = [unique_labels.index(label) for label in y_true]
    y_pred_bin = [unique_labels.index(label) for label in y_pred]

    # Create figure with subplots
    plt.figure(figsize=(50, 15))  # Increase width for wider graphs
    plt.rcParams.update({'font.size': 52})  # Increase font size

    # ROC curve
    plt.subplot(1, 3, 1)
    fpr, tpr, _ = roc_curve(y_true_bin, y_pred_bin, pos_label=1)
    roc_auc = roc_auc_score(y_true_bin, y_pred_bin)
    plt.plot(fpr, tpr, color="darkorange", linewidth=4, label=f"ROC curve (area = {roc_auc:.2f})")  # Thicker line
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--", linewidth=3)  # Diagonal line also thicker
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")

    # Confusion matrix
    plt.subplot(1, 3, 2)
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=decode_labels, yticklabels=decode_labels, cbar=False, linewidths=2)  # Thicker lines inside table
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Normalized Confusion Matrix")

    # Precision-Recall curve
    plt.subplot(1, 3, 3)
    precision, recall, _ = precision_recall_curve(y_true_bin, y_pred_bin, pos_label=1)
    plt.plot(recall, precision, color="b", linewidth=4, label="Precision-Recall curve")  # Thicker line
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")  # Legend in lower left corner

    # Save the image
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"all_metrics_task{task_number}.png"))
    plt.close()
