def schedule_activities(X, P, D):
    n = len(X)
    activities = []

    # Buat daftar aktivitas dari X, P, dan D
    for i in range(n):
        activities.append((X[i], P[i], D[i]))

    # Urutkan aktivitas berdasarkan keuntungan (P) secara menurun
    activities.sort(key=lambda x: x[1], reverse=True)

    # List untuk menyimpan hasil jadwal aktivitas
    schedule = []

    # Set waktu sekarang
    current_time = 0

    # Jadwalkan aktivitas berdasarkan urutan keuntungan
    for activity in activities:
        X, P, D = activity
        start_time = current_time + 1
        end_time = start_time + D - 1

        # Pastikan aktivitas dapat dilakukan sebelum waktu penyelesaian
        if end_time <= D:
            schedule.append((X, start_time, end_time))
            current_time = end_time

    return schedule

# Fungsi untuk meminta input data dari pengguna
def input_data():
    X = []
    P = []
    D = []

    n = int(input("Masukkan jumlah aktivitas: "))

    print("Masukkan keuntungan setiap aktivitas:")
    for i in range(n):
        profit = int(input(f"Keuntungan Aktivitas {i + 1}: "))
        P.append(profit)

    print("Masukkan waktu penyelesaian setiap aktivitas:")
    for i in range(n):
        deadline = int(input(f"Waktu Penyelesaian Aktivitas {i + 1}: "))
        D.append(deadline)

    # Mengisi X dengan nomor aktivitas 1, 2, ..., n
    X = list(range(1, n + 1))

    return X, P, D

# Meminta input data dari pengguna
X, P, D = input_data()

# Panggil fungsi untuk menjadwalkan aktivitas
result_schedule = schedule_activities(X, P, D)

# Tampilkan hasil jadwal aktivitas
print("\nUrutan aktivitas yang memberikan keuntungan maksimum:")
for activity in result_schedule:
    print(f"Aktivitas {activity} dimulai pada waktu {activity} dan selesai pada waktu {activity[2]}")
