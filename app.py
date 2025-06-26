import numpy as np
import streamlit as st

def resolver_problema(n, m, objetivo, restricoes, variacoes):
    # Primeira fase: construir problema auxiliar para minimizar soma das artificiais

    # Preparação
    folgas = []
    excessos = []
    artificiais = []

    A = []
    b = []
    tipo_restricao = []

    for i, restr in enumerate(restricoes):
        A.append(restr['coefs'])
        b.append(restr['rhs'])
        tipo_restricao.append(restr['sinal'])

        if restr['sinal'] == '≤':
            folgas.append(i)
        elif restr['sinal'] == '≥':
            excessos.append(i)
            artificiais.append(i)
        elif restr['sinal'] == '=':
            artificiais.append(i)

    total_vars = n + len(folgas) + len(excessos) + len(artificiais)
    var_names = []

    # Índices das variáveis
    slack_idx = {}
    excess_idx = {}
    artif_idx = {}
    idx = n

    for i in range(m):
        if i in folgas:
            slack_idx[i] = idx
            idx += 1
        elif i in excessos:
            excess_idx[i] = idx
            idx += 1
        if i in artificiais:
            artif_idx[i] = idx
            idx += 1

    # Montar matriz auxiliar da Fase 1
    tableau = []

    # Linha da função objetivo da Fase 1 (minimizar soma das artificiais)
    linha0 = [0.0] * idx + [0.0]
    for i in artificiais:
        linha0[artif_idx[i]] = 1.0
    tableau.append(linha0)

    # Restrições
    for i in range(m):
        linha = [0.0] * idx
        for j in range(n):
            linha[j] = A[i][j]

        if i in slack_idx:
            linha[slack_idx[i]] = 1.0
        elif i in excess_idx:
            linha[excess_idx[i]] = -1.0

        if i in artif_idx:
            linha[artif_idx[i]] = 1.0

        linha.append(b[i])
        tableau.append(linha)

    tableau = np.array(tableau, dtype=float)

    # Tornar linha 0 correta (subtrair linhas com artificiais)
    for i in artificiais:
        tableau[0, :] -= tableau[i + 1, :]

    # Base inicial
    basis = []
    for i in range(m):
        if i in slack_idx:
            basis.append(slack_idx[i])
        elif i in artif_idx:
            basis.append(artif_idx[i])
        else:
            basis.append(-1)

    # Simplex da Fase 1
    def simplex(tableau, basis, phase=1):
        rows, cols = tableau.shape
        while True:
            pivot_col = np.argmin(tableau[0, :-1])
            if tableau[0, pivot_col] >= -1e-8:
                break  # Ótimo

            ratios = []
            for i in range(1, rows):
                if tableau[i, pivot_col] > 1e-8:
                    ratios.append(tableau[i, -1] / tableau[i, pivot_col])
                else:
                    ratios.append(np.inf)

            pivot_row = np.argmin(ratios) + 1
            if ratios[pivot_row - 1] == np.inf:
                return None, None  # Ilimitado

            # Pivoteamento
            pivot_val = tableau[pivot_row, pivot_col]
            tableau[pivot_row, :] /= pivot_val
            for i in range(rows):
                if i != pivot_row:
                    tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

            basis[pivot_row - 1] = pivot_col

        return tableau, basis

    tableau, basis = simplex(tableau, basis, phase=1)

    # Verificar se solução da fase 1 é viável
    if tableau[0, -1] > 1e-5:
        st.error("Problema inviável (fase 1).")
        return None, None, None, None, None, None

    # Remover variáveis artificiais do tableau
    cols_keep = [j for j in range(tableau.shape[1] - 1) if j not in artif_idx.values()]
    cols_keep.append(tableau.shape[1] - 1)
    tableau = tableau[:, cols_keep]

    # Atualizar número de variáveis
    total_vars_final = n + len(folgas) + len(excessos)
    nova_obj = [0.0] * total_vars_final
    for i in range(n):
        nova_obj[i] = -objetivo[i]

    linha0 = nova_obj + [0.0]
    tableau[0, :] = linha0

    # Substituir linha 0 corretamente com a nova função objetivo
    for i, var in enumerate(basis):
        if var < len(nova_obj):
            coef = nova_obj[var]
            tableau[0, :] += coef * tableau[i + 1, :]

    # Fase 2
    tableau, basis = simplex(tableau, basis, phase=2)

    if tableau is None:
        st.error("Problema ilimitado.")
        return None, None, None, None, None, None

    # Extração de solução
    solucao = [0.0] * total_vars_final
    for i, var in enumerate(basis):
        if var < total_vars_final:
            solucao[var] = tableau[i + 1, -1]
    valor_otimo = tableau[0, -1]

    # Preços-sombra: coeficientes das variáveis de folga
    precos_sombra = []
    for i in range(m):
        if i in slack_idx:
            precos_sombra.append(tableau[0, slack_idx[i]])
        elif i in excess_idx:
            precos_sombra.append(tableau[0, excess_idx[i]])
        else:
            precos_sombra.append(0.0)

    # Verificação de viabilidade com variações
    S = tableau[1:m + 1, n:total_vars_final]
    b_atual = tableau[1:m + 1, -1]
    b_novo = b_atual + S.dot(variacoes)
    viavel = all(b_novo >= -1e-10)

    novo_lucro = None
    limites = []
    if viavel:
        novo_lucro = valor_otimo + sum(p * v for p, v in zip(precos_sombra, variacoes))
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

    return solucao[:n], valor_otimo, precos_sombra, viavel, novo_lucro, limites


def main():
    st.title("Método Simplex com Análise de Sensibilidade")
    st.markdown("""
    Esta aplicação resolve problemas de programação linear usando o método simplex e 
    realiza análise de sensibilidade para variações nos termos independentes das restrições.
    """)

    # Sidebar com informações
    st.sidebar.header("Sobre")
    st.sidebar.info("""
    Desenvolvido para resolver problemas de otimização linear com até 4 variáveis e 
    qualquer número de restrições. Fornece solução ótima, preços-sombra e análise 
    de sensibilidade.
    """)

    # Seção de configuração do problema
    st.header("Configuração do Problema")

    # Número de variáveis e restrições
    col1, col2 = st.columns(2)
    with col1:
        n = st.selectbox("Número de variáveis de decisão", [2, 3, 4], index=0)
    with col2:
        m = st.number_input("Número de restrições", min_value=1, value=2, step=1)

    # Função objetivo
    st.subheader("Função Objetivo (Maximização)")
    st.write(f"Digite os coeficientes para x1 a x{n}:")

    objetivo = []
    cols = st.columns(n)
    for i in range(n):
        with cols[i]:
            objetivo.append(st.number_input(f"Coeficiente x{i + 1}", value=0.0, step=0.1, key=f"obj_{i}"))

    # Restrições
    st.subheader("Restrições (≤, ≥, =)")
    restricoes = []

    for i in range(m):
        st.write(f"Restrição {i + 1}:")
        cols = st.columns(n + 2)
        coefs = []

        # Coeficientes
        for j in range(n):
            with cols[j]:
                coefs.append(st.number_input(f"x{j + 1}", value=0.0, step=0.1, key=f"restr_{i}_{j}"))

        # Sinal (fixo como <=)
        with cols[n]:
            sinal = st.selectbox("Sinal", ["≤", "≥", "="], key=f"sinal_{i}")

        # Termo independente
        with cols[n + 1]:
            rhs = st.number_input("Termo indep.", value=0.0, step=0.1, key=f"rhs_{i}")

        restricoes.append({'coefs': coefs, 'rhs': rhs, 'sinal': sinal})

    # Seção de análise de sensibilidade
    st.header("Análise de Sensibilidade")
    st.write("Digite as variações desejadas nos termos independentes das restrições:")

    variacoes = []
    for i in range(m):
        variacoes.append(st.number_input(
            f"Variação para Restrição {i + 1}",
            value=0.0,
            step=0.1,
            key=f"var_{i}"
        ))

    # Botão para resolver
    if st.button("Resolver Problema"):
        with st.spinner("Calculando solução ótima..."):
            solucao, valor_otimo, precos_sombra, viavel, novo_lucro, limites = resolver_problema(
                n, m, objetivo, restricoes, variacoes)

        if solucao is not None:
            # Mostrar resultados
            st.success("Solução encontrada!")

            st.subheader("Resultados")

            # Solução ótima
            st.write("**Solução Ótima:**")
            for i in range(n):
                st.write(f"x{i + 1} = {solucao[i]:.4f}")
            st.write(f"**Valor Ótimo:** {valor_otimo:.4f}")

            # Preços-sombra
            st.write("\n**Preços-Sombra:**")
            for i, ps in enumerate(precos_sombra):
                st.write(f"Restrição {i + 1}: {ps:.4f}")

            # Análise de sensibilidade
            st.write("\n**Análise de Sensibilidade:**")
            if viavel:
                st.success("As variações propostas são viáveis.")
                st.write(f"Novo Lucro Ótimo: {novo_lucro:.4f}")

                st.write("**Limites de Validade para Preços-Sombra:**")
                for i, (t_min, t_max) in enumerate(limites):
                    st.write(f"Restrição {i + 1}: [{t_min:.4f}, {t_max:.4f}]")
            else:
                st.error("As variações propostas tornam o problema inviável.")


if __name__ == "__main__":
    main()