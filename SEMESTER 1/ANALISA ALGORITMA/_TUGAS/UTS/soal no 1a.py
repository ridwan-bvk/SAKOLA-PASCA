from sympy import symbols, solve

# Minta input resistansi dari pengguna
RAB = float(input("Masukkan nilai resistansi RAB: "))
RAC = float(input("Masukkan nilai resistansi RAC: "))
RBC = float(input("Masukkan nilai resistansi RBC: "))
RBD = float(input("Masukkan nilai resistansi RBD: "))
RCD = float(input("Masukkan nilai resistansi RCD: "))

# Membuat simbol untuk arus
IAB, IAC, IBC, IBD, ICD = symbols('IAB IAC IBC IBD ICD')

# Persamaan untuk simpul
eq1 = IAB + IAC - IBC
eq2 = IBC - IBD
eq3 = IAC - ICD

# Persamaan untuk loop
eq4 = 5 * IAB + 3 * (IAC - IAB) - 6 * IBC - 4 * ICD

# Selesaikan sistem persamaan
solution = solve((eq1, eq2, eq3, eq4), (IAB, IAC, IBC, ICD))

# Keluarkan hasil
print("Nilai arus yang mengalir melalui setiap resistor:")
print(f"IAB = {solution[IAB]} A")
print(f"IAC = {solution[IAC]} A")
print(f"IBC = {solution[IBC]} A")
print(f"IBD = {solution[IBC]} A")  # Karena IBC = IBD
