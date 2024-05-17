def print_matrix(M):
    for row in M:
        print(row)
    print()

def gauss_jordan_verbose(A, b):
    n = len(A)
    
    # Matriks augmented [A | b]
    M = [A[i] + [b[i]] for i in range(n)]
    
    print("Matriks awal:")
    print_matrix(M)
    
    # Proses eliminasi Gauss
    for i in range(n):
        print(f"Iterasi {i + 1}:")
        
        # Pilih baris dengan elemen maksimum pada kolom i
        max_row = max(range(i, n), key=lambda x: abs(M[x][i]))
        M[i], M[max_row] = M[max_row], M[i]
        
        print("Pertukaran baris:")
        print_matrix(M)
        
        # Buat elemen diagonal menjadi 1
        pivot = M[i][i]
        for j in range(i, n + 1):
            M[i][j] /= pivot
        
        print("Buat elemen diagonal menjadi 1:")
        print_matrix(M)
        
        # Eliminasi elemen di bawah pivot
        for k in range(n):
            if k != i:
                factor = M[k][i]
                for j in range(i, n + 1):
                    M[k][j] -= factor * M[i][j]
        
        print("Eliminasi elemen di bawah pivot:")
        print_matrix(M)
    
    # Ambil solusi dari kolom terakhir
    x = [M[i][-1] for i in range(n)]
    
    print("Solusi:")
    print("x =", x)
    print()

# Contoh penggunaan
A = [[2, 1, -1],
     [-3, -1, 2],
     [-2, 1, 2]]
b = [8, -11, -3]

gauss_jordan_verbose(A, b)
