import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    r2_score,
    confusion_matrix,
    accuracy_score
)


def plot_histograms(
    verilog,
    python,
    architecture,
    filename
):
    """
    Plota os histogramas das saídas Verilog e Python.
    """

    fig, ax = plt.subplots(
        2, 1,
        figsize=(8, 6),
        sharex=True
    )

    fig.suptitle(
        f"Comparação entre saídas da rede neural em Python e Verilog\n"
        f"{architecture}",
        fontsize=16
    )

    ax[0].hist(
        verilog,
        bins=50,
        edgecolor='black'
    )
    ax[0].set_title("Verilog")
    ax[0].set_ylabel("Frequência")

    ax[1].hist(
        python,
        bins=50,
        edgecolor='black'
    )
    ax[1].set_title("Python")
    ax[1].set_ylabel("Frequência")
    ax[1].set_xlabel("Valor")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_r2(
    verilog,
    python,
    architecture,
    filename
):
    """
    Plota Python vs Verilog, a linha ideal y=x e calcula R².
    """

    r2 = r2_score(python, verilog)

    plt.figure(figsize=(6, 6))

    plt.scatter(
        python,
        verilog,
        edgecolor='k'
    )

    # Linha ideal y = x
    xmin = min(min(python), min(verilog))
    xmax = max(max(python), max(verilog))

    plt.plot(
        [xmin, xmax],
        [xmin, xmax],
        'r--'
    )

    plt.xlabel("Python")
    plt.ylabel("Verilog")

    plt.title(
        f"Python vs Verilog (R² = {r2:.6f})\n"
        f"{architecture}"
    )

    plt.grid(True, alpha=0.3)

    plt.savefig(filename, dpi=300)
    plt.show()

    print(f"R² = {r2:.6f}")

    return r2


def plot_confusion_matrix(
    verilog,
    python,
    ground_truth,
    architecture,
    filename
):
    """
    Calcula e plota as matrizes de confusão
    para Verilog e Python.
    """

    # Converte probabilidades em classes
    verilog_pred = (verilog >= 0.5).astype(int)
    python_pred = (python >= 0.5).astype(int)

    # Matrizes de confusão
    cm_verilog = confusion_matrix(
        ground_truth,
        verilog_pred
    )

    cm_python = confusion_matrix(
        ground_truth,
        python_pred
    )

    # Acurácia
    acc_verilog = accuracy_score(
        ground_truth,
        verilog_pred
    )

    acc_python = accuracy_score(
        ground_truth,
        python_pred
    )

    print(f"Acurácia Verilog: {acc_verilog:.4f}")
    print(f"Acurácia Python : {acc_python:.4f}")

    # Plot
    fig, ax = plt.subplots(
        1, 2,
        figsize=(12, 5)
    )

    fig.suptitle(
        f"Comparação entre matriz de confusão da rede neural "
        f"em Python e Verilog\n{architecture}",
        fontsize=16
    )

    sns.heatmap(
        cm_verilog,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax[0]
    )

    ax[0].set_title(
        f"Verilog\nAccuracy = {acc_verilog:.4f}"
    )
    ax[0].set_xlabel("Predito")
    ax[0].set_ylabel("Positive")

    sns.heatmap(
        cm_python,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax[1]
    )

    ax[1].set_title(
        f"Python\nAccuracy = {acc_python:.4f}"
    )
    ax[1].set_xlabel("Predito")
    ax[1].set_ylabel("Positive")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

    return acc_verilog, acc_python