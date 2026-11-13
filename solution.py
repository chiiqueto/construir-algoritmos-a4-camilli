import numpy as np
import pandas as pd


def solution(arquivo):
    """
    Regressão linear múltipla via equação normal:
        beta = (X^T X)^(-1) X^T y

    Recebe o caminho do CSV e retorna [beta_0, beta_1, beta_2], onde:
        Preco = beta_0 + beta_1 * Area + beta_2 * Quartos
    """
    df = pd.read_csv(arquivo)

    area = df["area_util_m2"].to_list()
    quartos = df["quantidade_quartos"].to_list()
    preco = df["preco_venda"].to_list()

    # Matriz de design X: coluna de 1s (intercepto), área e quartos
    n = len(preco)
    X = np.column_stack((np.ones(n), np.array(area), np.array(quartos)))
    y = np.array(preco).reshape(-1, 1)

    # Equação normal: (X^T X) beta = X^T y
    beta = np.linalg.solve(X.T @ X, X.T @ y)

    return [float(b) for b in beta.flatten()]
