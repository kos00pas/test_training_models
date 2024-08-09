import os
# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve, precision_score, recall_score, f1_score
import pickle
import h5py


# Load models and their histories into lists
model_files = [
    'trained_model_20240807_125239.keras', 'trained_model_20240807_125954.keras'
]
models = [tf.keras.models.load_model(model_file) for model_file in model_files]

# Load histories
history_files = [
    'history_20240807_125239.pkl', 'history_20240807_125954.pkl'
]
histories = [pickle.load(open(history_file, 'rb')) for history_file in history_files]
import numpy as np

# Load your custom dataset
# Load custom dataset
file_path = '../FINISHED_V6/ours_3/ours_3_test_dataset.h5'
with h5py.File(file_path, 'r') as f:
        mfcc_keys = list(f['mfcc'].keys())
        x_test = np.array([f['mfcc'][key][:] for key in mfcc_keys])
        y_test = np.array([int(key) for key in mfcc_keys])  # Assuming the labels are derived from the keys

# Preprocess the dataset
x_test = x_test / np.max(x_test)  # Normalize the images
x_test = x_test.reshape((-1, 40, 32, 1))  # Reshape to include channel dimension if required
y_test = tf.keras.utils.to_categorical(y_test, len(set(y_test)))  # Adjust number of classes as needed

# Directory to save plots
save_dir = 'model_comparison_plots'
os.makedirs(save_dir, exist_ok=True)
print("ok")

def save_plot(fig, plot_name, model_name):
    fig.savefig(os.path.join(save_dir, f"{model_name}_{plot_name}.png"))
    plt.close(fig)


def plot_training_history(history, model_name):
    fig = plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title(f'{model_name} Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title(f'{model_name} Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    save_plot(fig, 'training_history', model_name)


def plot_confusion_matrix(model, x_test, y_test, model_name):
    y_pred = model.predict(x_test)
    y_pred_classes = y_pred.argmax(axis=-1)
    y_true_classes = y_test.argmax(axis=-1)
    cm = confusion_matrix(y_true_classes, y_pred_classes)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title(f'{model_name} Confusion Matrix')

    save_plot(fig, 'confusion_matrix', model_name)


def plot_roc_curve(model, x_test, y_test, model_name):
    y_pred = model.predict(x_test).ravel()
    y_test_flat = y_test.ravel()
    fpr, tpr, _ = roc_curve(y_test_flat, y_pred)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    save_plot(fig, 'roc_curve', model_name)


def plot_precision_recall_curve(model, x_test, y_test, model_name):
    y_pred = model.predict(x_test).ravel()
    y_test_flat = y_test.ravel()
    precision, recall, _ = precision_recall_curve(y_test_flat, y_pred)

    fig = plt.figure()
    plt.plot(recall, precision, lw=2, color='b', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Precision-Recall Curve')
    plt.legend(loc="upper right")

    save_plot(fig, 'precision_recall_curve', model_name)


def plot_lr_vs_loss(history, model_name):
    if 'lr' in history:
        fig = plt.figure()
        plt.plot(history['lr'], history['loss'])
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title(f'{model_name} Learning Rate vs. Loss')

        save_plot(fig, 'lr_vs_loss', model_name)


def plot_metrics_over_epochs(history, x_test, y_test, model, model_name):
    precision, recall, f1 = [], [], []
    for epoch in range(len(history['loss'])):
        y_pred = (model.predict(x_test) > 0.5).astype("int32")
        precision.append(precision_score(y_test.argmax(axis=-1), y_pred.argmax(axis=-1), average='micro'))
        recall.append(recall_score(y_test.argmax(axis=-1), y_pred.argmax(axis=-1), average='micro'))
        f1.append(f1_score(y_test.argmax(axis=-1), y_pred.argmax(axis=-1), average='micro'))

    fig = plt.figure(figsize=(12, 4))

    # Precision
    plt.subplot(1, 3, 1)
    plt.plot(range(len(history['loss'])), precision, label='Precision')
    plt.title(f'{model_name} Precision over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')

    # Recall
    plt.subplot(1, 3, 2)
    plt.plot(range(len(history['loss'])), recall, label='Recall')
    plt.title(f'{model_name} Recall over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')

    # F1 Score
    plt.subplot(1, 3, 3)
    plt.plot(range(len(history['loss'])), f1, label='F1 Score')
    plt.title(f'{model_name} F1 Score over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')

    save_plot(fig, 'metrics_over_epochs', model_name)


def print_model_summary(model, model_name):
    with open(os.path.join(save_dir, f"{model_name}_summary.txt"), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(f'Total Parameters in {model_name}: {model.count_params()}')


# Process each model in the list
for i, (model, history) in enumerate(zip(models, histories)):
    model_name = f'Model_{i + 1}'
    print(f"Evaluating {model_name}...")

    plot_training_history(history, model_name)
    plot_confusion_matrix(model, x_test, y_test, model_name)
    plot_roc_curve(model, x_test, y_test, model_name)
    plot_precision_recall_curve(model, x_test, y_test, model_name)
    plot_lr_vs_loss(history, model_name)
    plot_metrics_over_epochs(history, x_test, y_test, model, model_name)
    print_model_summary(model, model_name)
    print(f"Finished evaluating {model_name}.")

