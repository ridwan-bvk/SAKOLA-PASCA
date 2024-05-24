import numpy as np
import matplotlib.pyplot as plt

# Fungsi yang akan dicari nilai f(x) nya
def f(x):
    return np.exp(x) - 2 - x**2

# Metode grafik untuk mencari nilai f(x) pada titik x yang ditentukan
def graphical_method(x_values, delta_x):
    fx_values = [f(x) for x in x_values]  # Menghitung nilai f(x) untuk setiap titik x
    return fx_values

def main():
    # Titik awal, akhir, dan delta x
    x_start = 0.5
    x_end = 1.5
    delta_x = 0.5

    # Membuat daftar titik-titik x dalam selang (∆x)
    x_values = np.arange(x_start, x_end + delta_x, delta_x)

    # Melakukan pencarian nilai f(x) dengan metode grafik
    fx_values = graphical_method(x_values, delta_x)

    # Menampilkan hasil
    for i, x in enumerate(x_values):
        print(f"Nilai f({x}) = {fx_values[i]}")

    # Menampilkan grafik f(x)
    x = np.linspace(min(x_values), max(x_values), 100)
    y = f(x)
    plt.plot(x, y)
    plt.scatter(x_values, fx_values, color='red')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Grafik f(x) = e^x - 2 - x^2')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
