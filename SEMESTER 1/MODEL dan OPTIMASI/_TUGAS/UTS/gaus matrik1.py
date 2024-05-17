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
        
        print(f"Baris {i + 1} dipilih untuk pivot:")
        print_matrix(M)
        
        # Buat elemen diagonal menjadi 1
        pivot = M[i][i]
        for j in range(i, n + 1):
            M[i][j] /= pivot
        
        print(f"Buat elemen diagonal pada baris {i + 1} menjadi 1:")
        print_matrix(M)
        
        # Eliminasi elemen di bawah pivot
        for k in range(n):
            if k != i:
                factor = M[k][i]
                for j in range(i, n + 1):
                    M[k][j] -= factor * M[i][j]
        
        print(f"Eliminasi elemen di bawah pivot pada kolom {i + 1}:")
        print_matrix(M)
        
        # Print hasil setiap iterasi
        print(f"Hasil setelah iterasi {i + 1}:")
        print_matrix([[M[row][col] for col in range(n, n + 1)] for row in range(n)])
    
    # Ambil solusi dari kolom terakhir
    x = [M[i][-1] for i in range(n)]
    
    print("Solusi:")
    print("x =", x)
    print()

# Contoh penggunaan
A = [[1, 2, 1],
     [7, 6, 4],
     [6, 3, 5]]
b = [800, 700, 1000]

gauss_jordan_verbose(A, b)
