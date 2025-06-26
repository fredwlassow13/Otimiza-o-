import streamlit as st
import numpy as np
from simplex_solver import solve_lp, apply_variation, is_variation_valid

st.set_page_config(page_title="Simplex com Análise de Sensibilidade", layout="wide")
st.title("📈 Resolução de PPL com Análise de Sensibilidade")

st.header("1. Entrada de Dados")

num_vars = st.number_input("Número de variáveis", min_value=2, max_value=4, value=3)
c = st.text_input("Coeficientes da função objetivo (separados por vírgula)", "50,70,100")
c = [float(x.strip()) for x in c.split(",")]

num_rest = st.number_input("Número de restrições", min_value=1, value=5)
A, b, signs = [], [], []

st.write("Digite os coeficientes, o sinal e o lado direito de cada restrição:")

for i in range(num_rest):
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        row = st.text_input(f"Restrição {i + 1} (ex: 1,0,0)", key=f"A{i}")
        if row.strip():
            try:
                coef = [float(x.strip()) for x in row.split(",")]
                if len(coef) != num_vars:
                    st.error(f"A restrição {i + 1} deve ter exatamente {num_vars} coeficientes.")
                    A.append([0.0] * num_vars)
                else:
                    A.append(coef)
            except ValueError:
                st.error(f"Coeficientes inválidos na restrição {i + 1}.")
                A.append([0.0] * num_vars)
        else:
            st.warning(f"Preencha os coeficientes da restrição {i + 1}.")
            A.append([0.0] * num_vars)

    with col2:
        sinal = st.selectbox("Sinal", ["<=", ">=", "="], key=f"sinal_{i}")
        signs.append(sinal)

    with col3:
        b_val = st.number_input("b", key=f"b{i}")
        b.append(b_val)

if st.button("🔍 Resolver"):
    try:
        x_opt, lucro, shadow_prices, res = solve_lp(c, A, b, signs=signs)

        st.success("✅ Solução encontrada!")
        st.write("**Ponto ótimo de operação:**", np.round(x_opt, 4))
        st.write("**Lucro ótimo:**", np.round(lucro, 2))
        st.write("**Preços-sombra:**")
        for i, val in enumerate(shadow_prices):
            st.write(f"Restrição {i+1}: {val:.4f}")

        if st.checkbox("Deseja alterar disponibilidade das restrições?"):
            delta = []
            for i in range(num_rest):
                val = st.number_input(f"∆ Restrição {i+1}", value=0.0, key=f"delta{i}")
                delta.append(val)

            new_b = apply_variation(b, delta)
            if is_variation_valid(res, delta):
                x_new, lucro_new, *_ = solve_lp(c, A, new_b, signs=signs)
                st.success("✅ Alterações são viáveis!")
                st.write("**Novo ponto ótimo:**", np.round(x_new, 4))
                st.write("**Novo lucro ótimo:**", np.round(lucro_new, 2))
            else:
                st.error("❌ As alterações inviabilizam a solução atual.")

    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
