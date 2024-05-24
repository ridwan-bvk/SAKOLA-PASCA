import numpy as np

def f(x):
    return np.exp(x) - x**2 - 2

def SecantMethod(x0, x1, iterations):
    tolerance = 0.01 * (x1 - x0)  # Toleransi
    
    for i in range(1, iterations+1):
        fx0 = f(x0)
        fx1 = f(x1)
        x2 = x1 - fx1 * ((x1 - x0) / (fx1 - fx0))
        
        print("Iterasi", i, ":", x2)
        
        if abs(f(x2)) < tolerance:
            return x2
        
        x0 = x1
        x1 = x2
    
    return "Metode tidak konvergen setelah jumlah iterasi maksimum"

def main():
    x0 = 0.5  # Tebakan awal
    x1 = 1.5  # Tebakan kedua
    iterations = 10  # Jumlah iterasi
    
    root = SecantMethod(x0, x1, iterations)
    
    print("\nAkar yang ditemukan:", root)

if __name__ == "__main__":
    main()
