import numpy as np

def f(x):
    return np.exp(x) - x**2 - 2

def regula_falsi_method(f, x_start, x_end, tolerance):
    if f(x_start) * f(x_end) > 0:
        return "Titik awal tidak valid"
    
    iterations = []  # Untuk menyimpan informasi per iterasi
    
    while abs(x_end - x_start) > tolerance:
        x2 = (x_start * f(x_end) - x_end * f(x_start)) / (f(x_end) - f(x_start))
        f_x2 = f(x2)
        if f_x2 == 0:
            return x2, iterations
        elif f(x_start) * f_x2 < 0:
            x_end = x2
        else:
            x_start = x2
        
        iterations.append((x_start, x_end))
    
    return (x_start + x_end) / 2, iterations

def main():
    x_start = 0.5  # Titik awal
    x_end = 1.5    # Titik akhir
    tolerance = 0.01 * (x_end - x_start)  # Toleransi 1.0% dari panjang interval
    
    root, iterations = regula_falsi_method(f, x_start, x_end, tolerance)
    
    if isinstance(root, str):
        print(root)
    else:
        print("Akar yang ditemukan:", root)
        
        # Menampilkan informasi per iterasi
        print("\nInformasi per iterasi:")
        print("Iterasi\t  x_start\t  x_end")
        for i, (x_start, x_end) in enumerate(iterations):
            print(f"{i+1}\t{x_start:.6f}\t{x_end:.6f}")

if __name__ == "__main__":
    main()
