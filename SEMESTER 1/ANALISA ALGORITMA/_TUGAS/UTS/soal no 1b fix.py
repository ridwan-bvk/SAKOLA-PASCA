def solve_circuit(R_AB, R_AC, R_BC, R_BD, R_CD):
    # Menghitung nilai arus
    I_BC = 3  # Asumsikan nilai awal untuk I_BC
    I_AC = 3 * I_BC
    I_AB = I_AC
    I_BD = 6 * I_BC
    I_CD = I_AC - I_BC

    # Menampilkan hasil
    print("Arus yang mengalir melalui setiap resistor:")
    print(f"I_AB = {I_AB} A")
    print(f"I_AC = {I_AC} A")
    print(f"I_BC = {I_BC} A")
    print(f"I_BD = {I_BD} A")
    print(f"I_CD = {I_CD} A")

if __name__ == "__main__":
    # Input nilai resistansi antara simpul
    R_AB = float(input("Masukkan nilai resistansi antara simpul A dan B (ohm): "))
    R_AC = float(input("Masukkan nilai resistansi antara simpul A dan C (ohm): "))
    R_BC = float(input("Masukkan nilai resistansi antara simpul B dan C (ohm): "))
    R_BD = float(input("Masukkan nilai resistansi antara simpul B dan D (ohm): "))
    R_CD = float(input("Masukkan nilai resistansi antara simpul C dan D (ohm): "))

    # Memanggil fungsi untuk menyelesaikan sirkuit
    solve_circuit(R_AB, R_AC, R_BC, R_BD, R_CD)