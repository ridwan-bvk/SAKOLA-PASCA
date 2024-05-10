def kirchhoff():
    # Meminta input nilai resistor
    Rab = float(input("Masukkan nilai resistor Rab: "))
    Rac = float(input("Masukkan nilai resistor Rac: "))
    Rbc = float(input("Masukkan nilai resistor Rbc: "))
    Rbd = float(input("Masukkan nilai resistor Rbd: "))
    Rcd = float(input("Masukkan nilai resistor Rcd: "))
    
    # Terapkan hukum simpul
    A_simpul = [[-1, 3, 0], [-1, 0, 6]]  # [koefisien I1, koefisien I2, koefisien I3]
    b_simpul = [0, 0]  # Nilai konstanta di sebelah kanan
    
    # Terapkan hukum loop
    loop_ABC = [Rab, -Rac, -Rbc]  # Koefisien untuk I1, I2, dan (I1 - I2)
    loop_ABD = [Rab, 0, -Rbd]      # Koefisien untuk I1, I3, dan (I1 - I3)
    loop_BCD = [-Rbc, Rcd, -Rbd]   # Koefisien untuk (I1 - I2), I3, dan I2
    b_loop = [0, 0, 0]             # Nilai konstanta di sebelah kanan
    
    # Selesaikan sistem persamaan
    simpul_solutions = solve_system(A_simpul, b_simpul)
    I1 = simpul_solutions[0]  # Nilai I1 dari solusi simpul
    I3 = simpul_solutions[1]  # Nilai I3 dari solusi simpul
    
    # Hitung I2 menggunakan I3 yang sudah diketahui
    I2 = 2 * I3
    
    # Tampilkan nilai arus
    print("Arus pada resistor:")
    print("I1 =", round(I1, 2), "A")
    print("I2 =", round(I2, 2), "A")
    print("I3 =", round(I3, 2), "A")

def solve_system(A, b):
    # Solusi sistem persamaan menggunakan metode eliminasi Gauss
    n = len(b)
    for i in range(n):
        # Lakukan pivot untuk mendapatkan nilai diagonal yang tidak nol
        if A[i][i] == 0:
            for j in range(i + 1, n):
                if A[j][i] != 0:
                    # Tukar baris i dan j
                    A[i], A[j] = A[j], A[i]
                    b[i], b[j] = b[j], b[i]
                    break
        # Normalisasi baris i
        pivot = A[i][i]
        for j in range(i, n):
            A[i][j] /= pivot
        b[i] /= pivot
        # Eliminasi pada kolom i
        for k in range(i + 1, n):
            factor = A[k][i]
            for j in range(i, n):
                A[k][j] -= factor * A[i][j]
            b[k] -= factor * b[i]
    # Substitusi mundur
    solutions = [0] * n
    for i in range(n - 1, -1, -1):
        solutions[i] = b[i]
        for j in range(i + 1, n):
            solutions[i] -= A[i][j] * solutions[j]
    return solutions

kirchhoff()
