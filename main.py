import numpy as np


def resolver_problema(n, m, objetivo, restricoes, variacoes):
    # Construção do Tableau Inicial
    tableau = []
    # Linha 0: coeficientes negativos da função objetivo (maximização)
    linha0 = [-c for c in objetivo] + [0] * m + [0]
    tableau.append(linha0)

    # Linhas das restrições
    for i, rest in enumerate(restricoes):
        linha = rest['coefs'] + [1 if j == i else 0 for j in range(m)] + [rest['rhs']]
        tableau.append(linha)

    tableau = np.array(tableau, dtype=float)
    basis = list(range(n, n + m))  # Índices das variáveis de folga iniciais

    # Execução do Método Simplex
    while True:
        # Encontrar coluna pivô (coeficiente mais negativo na linha 0)
        min_val = 0
        pivot_col = -1
        for j in range(n + m):
            if tableau[0, j] < min_val - 1e-10:
                min_val = tableau[0, j]
                pivot_col = j
        if pivot_col == -1:
            break  # Solução ótima

        # Encontrar linha pivô (menor razão não negativa)
        min_ratio = None
        pivot_row = -1
        for i in range(1, m + 1):
            if tableau[i, pivot_col] > 1e-10:
                ratio = tableau[i, -1] / tableau[i, pivot_col]
                if min_ratio is None or ratio < min_ratio:
                    min_ratio = ratio
                    pivot_row = i
        if pivot_row == -1:
            print("Problema ilimitado.")
            return None, None, None, None

        # Atualizar base
        basis[pivot_row - 1] = pivot_col

        # Pivotear: normalizar linha pivô
        pivot_val = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_val

        # Atualizar outras linhas
        for i in range(m + 1):
            if i == pivot_row:
                continue
            factor = tableau[i, pivot_col]
            tableau[i, :] -= factor * tableau[pivot_row, :]

    # Extrair solução ótima
    solucao = [0.0] * (n + m)
    for i in range(m):
        j = basis[i]
        solucao[j] = tableau[i + 1, -1]
    valor_otimo = tableau[0, -1]

    # Extrair preços-sombra (coeficientes das folgas na linha 0)
    precos_sombra = [tableau[0, n + i] for i in range(m)]

    # Verificar viabilidade das alterações
    S = tableau[1:m + 1, n:n + m]  # Matriz de sensibilidade
    b_atual = tableau[1:m + 1, -1]  # Lados direitos atuais
    b_novo = b_atual + S.dot(variacoes)  # Novos lados direitos
    viavel = all(b_novo >= -1e-10)

    # Calcular novo lucro e limites de validade, se viável
    novo_lucro = None
    limites = []
    if viavel:
        novo_lucro = valor_otimo + sum(p * v for p, v in zip(precos_sombra, variacoes))
        # Calcular limites de validade para cada restrição
        for i in range(m):
            coluna = S[:, i]
            t_min = -np.inf
            t_max = np.inf
            for k in range(m):
                if coluna[k] > 1e-10:
                    t_min = max(t_min, -b_atual[k] / coluna[k])
                elif coluna[k] < -1e-10:
                    t_max = min(t_max, -b_atual[k] / coluna[k])
            limites.append((t_min, t_max))

    return solucao, valor_otimo, precos_sombra, viavel, novo_lucro, limites


def main():
    # Entrada do número de variáveis e restrições
    n = int(input("Número de variáveis (2, 3 ou 4): "))
    m = int(input("Número de restrições: "))

    # Entrada da função objetivo
    print("\nFunção Objetivo (coeficientes de x1 a xn):")
    objetivo = list(map(float, input().split()))
    if len(objetivo) != n:
        print("Número incorreto de coeficientes.")
        return

    # Entrada das restrições
    restricoes = []
    for i in range(m):
        print(f"\nRestrição {i + 1} (coeficientes de x1 a xn, sinal [<=, >=, =], termo independente):")
        entrada = input().split()
        sinal = entrada[-2]
        if sinal not in ['≤', '<=', '≥', '>=', '=']:
            print("Sinal inválido. Use <=, >= ou =.")
            return
        if sinal == '<=':
            sinal = '≤'
        if sinal == '>=':
            sinal = '≥'
        coefs = list(map(float, entrada[:-2]))
        rhs = float(entrada[-1])
        if len(coefs) != n:
            print("Número incorreto de coeficientes.")
            return
        restricoes.append({'coefs': coefs, 'rhs': rhs, 'sinal': sinal})

    while True:
        # Entrada das variações desejadas
        print("\nVariações desejadas nos termos independentes (uma por linha):")
        variacoes = [float(input()) for _ in range(m)]

        # Resolver o problema
        solucao, valor_otimo, precos_sombra, viavel, novo_lucro, limites = resolver_problema(
            n, m, objetivo, restricoes, variacoes)

        if solucao is None:
            return

        # Saída dos resultados
        print("\n--- RESULTADOS ---")
        print("\nSolução Ótima:")
        for i in range(n):
            print(f"x{i + 1} = {solucao[i]:.4f}")
        print(f"Valor Ótimo: {valor_otimo:.4f}")

        print("\nPreços-Sombra:")
        for i, ps in enumerate(precos_sombra):
            print(f"Restrição {i + 1}: {ps:.4f}")

        print("\nAlterações Desejadas:")
        if viavel:
            print("Viáveis.")
            print(f"Novo Lucro Ótimo: {novo_lucro:.4f}")
            print("Limites de Validade para Preços-Sombra:")
            for i, (t_min, t_max) in enumerate(limites):
                print(f"Restrição {i + 1}: [{t_min:.4f}, {t_max:.4f}]")
        else:
            print("Inviáveis.")

        # Perguntar se deseja fazer novas variações
        while True:
            resposta = input("\nDeseja fazer novas variações? (s/n): ").strip().lower()
            if resposta in ['s', 'n']:
                break
            print("Por favor, digite 's' para sim ou 'n' para não.")

        if resposta == 'n':
            break


if __name__ == "__main__":
    main()