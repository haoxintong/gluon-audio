# MIT License
# Copyright (c) 2019 haoxintong
"""Visualization tools for gluonar"""
import seaborn as sns
import librosa as rosa
import librosa.display
import matplotlib.pyplot as plt

__all__ = ["plot_accuracy", "plot_roc", "view_spec"]


def plot_roc(tpr, fpr, x_name="FPR", y_name="TPR"):
    sns.set(style="darkgrid")
    plt.figure(figsize=(8, 8))
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title("ROC")

    sns.lineplot(x=x_name, y=y_name, data={x_name: fpr, y_name: tpr})
    plt.show()


def plot_accuracy(accs, threholds):
    sns.set(style="darkgrid")
    plt.figure(figsize=(8, 8))
    plt.xlabel("threshold")
    plt.ylabel("accuracy")
    plt.title("Accuracy")

    sns.lineplot(x="threshold", y="accuracy", data={"accuracy": accs, "threshold": threholds})
    plt.show()


def view_spec(spec_img, sample_rate=16000):
    spec_img = rosa.power_to_db(spec_img)
    plt.figure()
    rosa.display.specshow(spec_img, sr=sample_rate, x_axis='time', y_axis='mel')
    plt.title('Spectrogram')
    plt.show()
