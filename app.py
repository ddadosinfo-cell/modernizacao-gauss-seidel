import streamlit as st
import numpy as np

# Configuração da página (Estilo Dark/Moderno)
st.set_page_config(page_title="Gauss-Seidel Solver", page_icon="📊", layout="wide")

st.title("📊 Resolução de Sistemas Lineares — Gauss-Seidel")
st.markdown("Insira os dados do seu sistema linear abaixo para calcular a solução interativamente.")

# Sidebar para parâmetros de controle
st.sidebar.header("⚙️ Parâmetros de Controle")
tol = st.sidebar.number_input("Tolerância do Erro", value=0.00001, format="%.5f")
max_its = st.sidebar.number_input("Máximo de Iterações", value=100, step=10)

# 1. Definição do Tamanho do Sistema
n = st.number_input("Número de Equações (N)", min_value=2, max_value=10, value=3)

st.subheader("📝 Dados do Sistema")
col1, col2 = st.columns(2)

with col1:
    st.write("Matriz de Coeficientes (A) — *Preencha como uma planilha*")
    # Cria uma matriz inicial preenchida com zeros para o usuário editar
    init_A = np.zeros((n, n))
    A_input = st.data_editor(init_A, key="matrix_A", hide_index=True)

with col2:
    st.write("Vetor de Termos Independentes (B)")
    init_b = np.zeros((n, 1))
    b_input = st.data_editor(init_b, key="vector_b", hide_index=True)

# Palpite Inicial X
st.subheader("💡 Palpite Inicial (X₀)")
init_x = np.zeros((1, n))
x_input = st.data_editor(init_x, key="vector_x", hide_index=True)

# Botão de Ação Principal
if st.button("🚀 Calcular Solução", type="primary"):
    A = np.array(A_input, dtype=float)
    b = np.array(b_input, dtype=float).flatten()
    x = np.array(x_input, dtype=float).flatten()
    
    # Validação da diagonal
    if np.any(np.diag(A) == 0):
        st.error("Erro: A matriz possui elementos zero na diagonal principal. O método não pode prosseguir.")
    else:
        # Algoritmo de Gauss-Seidel Modernizado
        diagonal = np.diag(A)
        A_norm = A / diagonal[:, np.newaxis]
        b_norm = b / diagonal
        L_mod = np.tril(A_norm)
        U_mod = A_norm - L_mod
        
        iters = 0
        converged = False
        
        with st.spinner("Processando iterações numéricas..."):
            while iters < max_its and not converged:
                iters += 1
                x_old = np.copy(x)
                rhs = b_norm - np.dot(U_mod, x_old)
                for i in range(n):
                    sum_j = np.dot(L_mod[i, :i], x[:i])
                    x[i] = rhs[i] - sum_j
                
                big = np.max(np.abs(x))
                if big > 0:
                    error = np.max(np.abs(x - x_old)) / big
                    if error <= tol:
                        converged = True

        # 3. Exibição Resultante Amigável
        st.success(f"⚡ Convergência atingida em {iters} iterações!")
        
        st.subheader("🎯 Vetor Solução Final (X):")
        # Exibe o resultado de forma elegante em colunas
        cols = st.columns(n)
        for idx, val in enumerate(x):
            cols[idx].metric(label=f"X_{idx+1}", value=f"{val:.5f}")