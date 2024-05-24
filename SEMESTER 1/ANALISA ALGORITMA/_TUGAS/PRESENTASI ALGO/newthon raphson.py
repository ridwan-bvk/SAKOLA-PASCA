import numpy as np

def f(x):
    return np.exp(x) - x**2 - 2

def f_prime(x):
    return np.exp(x) - 2*x

def NewtonRaphson(x0, xn, iterations):
    x = x0  # Tebakan awal
    tolerance = 0.01 * (xn - x0)  # Toleransi
    
    print("Iterasi\t  x")
    for i in range(iterations):
        fx = f(x)
        f_prime_x = f_prime(x)
        
        if f_prime_x == 0:
            return "Turunan fungsi adalah nol. Metode tidak konvergen."
        
        x_new = x - fx / f_prime_x
        
        print(f"{i+1}\t  {x_new:.6f}")
        
        if abs(x_new - x) < tolerance:
            return x_new  # Akar yang ditemukan
        
        x = x_new
    
    return "Metode tidak konvergen setelah jumlah iterasi maksimum."

def main():
    x0 = 0.5  # Tebakan awal
    xn = 1.5  # Tebakan akhir
    iterations = 10  # Jumlah iterasi maksimum
    
    root = NewtonRaphson(x0, xn, iterations)
    
    print("\nAkar yang ditemukan:", root)

if __name__ == "__main__":
    main()
