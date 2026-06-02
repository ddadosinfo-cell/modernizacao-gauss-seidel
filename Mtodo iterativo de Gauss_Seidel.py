import numpy as np

def gauss_seidel_solver():
    """
    Executa o método iterativo de Gauss-Seidel para resolver o sistema Ax = b.
    Substitui a lógica legada em Fortran por operações vetoriais modernas utilizando NumPy.
    """
    print("*** Método Gauss-Seidel Modernizado ***\n")
    
    # 1. LEITURA E ENTRADA DE DADOS
    try:
        n = int(input("Digite o número de equações (N): "))
        
        print("\nDigite os coeficientes da Matriz A (linha por linha, separados por espaço):")
        matrix_rows = []
        for i in range(n):
            row = list(map(float, input(f"Linha {i+1}: ").split()))
            if len(row) != n:
                raise ValueError(f"A linha deve conter exatamente {n} elementos.")
            matrix_rows.append(row)
        A = np.array(matrix_rows, dtype=float)
        
        print("\nDigite o vetor de termos independentes B (separados por espaço):")
        b = np.array(list(map(float, input().split())), dtype=float)
        
        print("\nDigite os palpites iniciais para o vetor X (separados por espaço):")
        x = np.array(list(map(float, input().split())), dtype=float)
        
        tol = float(input("\nDigite a tolerância (ex: 0.0001): "))
        max_its = int(input("Digite o número máximo de iterações: "))
        
    except ValueError as e:
        print(f"\n[Erro de Entrada]: {e}. Certifique-se de digitar números válidos.")
        return

    # Exibição dos dados iniciais (Equivalente ao PRINTA e PRINTV originais)
    print("\n" + "="*40)
    print("MATRIZ DE COEFICIENTES (A):")
    print(A)
    print("\nVETOR RESULTANTE (B):")
    print(b)
    print("="*40 + "\n")

    # 2. PRÉ-PROCESSAMENTO (Normalização pela Diagonal Principal)
    # No Fortran original: DO I=1,N -> A(I,J)=A(I,J)/DIAG -> B(I)=B(I)/DIAG
    diagonal = np.diag(A)
    if np.any(diagonal == 0):
        print("[Erro]: A matriz possui elementos zero na diagonal principal. O método não pode prosseguir.")
        return
        
    A_norm = A / diagonal[:, np.newaxis]
    b_norm = b / diagonal

    # 3. DECOMPOSIÇÃO DA MATRIZ
    # No legado, a matriz era quebrada em U (triangular superior negativa) e A modificada.
    # Abordagem moderna: Extraímos diretamente as matrizes L (Lower) e U (Upper).
    # Como a diagonal de A_norm agora é 1, L_mod terá 1 na diagonal.
    L_mod = np.tril(A_norm) 
    U_mod = A_norm - L_mod  # Contém a parte estritamente superior positiva

    print("PRIMEIRAS ITERAÇÕES:")
    iters = 0
    converged = False
    
    # 4. LOOP PRINCIPAL DE ITERAÇÃO (Substitui o '4 ITERS=ITERS+1' e o 'GO TO 4')
    while iters < max_its and not converged:
        iters += 1
        x_old = np.copy(x)
        
        # Equivalente moderno à combinação de MVMULT, VECADD e SUBFOR:
        # Na lógica original: x_new = U * x_old + b_norm, seguido por substituição para frente com L.
        # Matematicamente, o passo de Gauss-Seidel é: L_mod * x_new = b_norm - U_mod * x_old
        rhs = b_norm - np.dot(U_mod, x_old)
        
        # Resolve o sistema triangular inferior (Substituição para frente - antiga SUBFOR)
        for i in range(n):
            sum_j = np.dot(L_mod[i, :i], x[:i])
            x[i] = rhs[i] - sum_j

        # Exibe as 5 primeiras iterações conforme regra do código legado
        if iters <= 5:
            print(f"Iteração {iters}: {x}")

        # CRITÉRIO DE CONVERGÊNCIA (Antiga Subrotina CHECON)
        # Calcula o erro relativo usando a norma do infinito (maior valor absoluto)
        big = np.max(np.abs(x))
        if big > 0:
            error = np.max(np.abs(x - x_old)) / big
            if error <= tol:
                converged = True

    # 5. RESULTADOS FINAIS
    print("\n" + "="*40)
    print(f"ITERAÇÕES ATÉ A CONVERGÊNCIA: {iters}")
    if not converged:
        print("[Aviso]: O método atingiu o limite máximo de iterações sem convergir.")
    print("\nVETOR SOLUÇÃO (X):")
    print(x)
    print("="*40)

if __name__ == "__main__":
    gauss_seidel_solver()