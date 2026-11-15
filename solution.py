import numpy as np
import pandas as pd


def solution(arquivo):
    df = pd.read_csv(arquivo)

    area = df["area_util_m2"].to_list()
    quartos = df["quantidade_quartos"].to_list()
    preco = df["preco_venda"].to_list()

    X = np.column_stack([np.ones(len(area)), np.array(area), np.array(quartos)])
    y = np.array(preco, dtype=float).reshape(-1, 1)

    beta = np.linalg.inv(X.T @ X) @ X.T @ y

    return [float(beta[0, 0]), float(beta[1, 0]), float(beta[2, 0])]


if __name__ == "__main__":
    b0, b1, b2 = solution("imoveis_lar_perfeito.csv")
    print(f"Preco = {b0:.4f} + {b1:.4f} * Area + {b2:.4f} * Quartos")