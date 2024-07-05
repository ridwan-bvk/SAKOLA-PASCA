
def activity_selection(X, P, D):
    # Menggabungkan data Xi, Pi, dan Di ke dalam satu list activities
    activities = list(zip(X, P, D))
    
    # Langkah 1: Urutkan aktivitas berdasarkan waktu selesai (Xi + Di)
    activities.sort(key=lambda x: x[0] + x[2])
    
    n = len(activities)
    M = [0] * n  # Inisialisasi array M dengan M[0] = 0
    
    # Langkah 3: Algoritma pemrograman dinamis
    for i in range(n):
        P_i = activities[i][1]  # Keuntungan dari aktivitas i
        M[i] = P_i  # Inisialisasi dengan keuntungan awal aktivitas i
        
        for j in range(i):
            if activities[j][0] + activities[j][2] <= activities[i][0]:
                M[i] = max(M[i], P_i + M[j])
    
    # Langkah 4: Rekonstruksi urutan aktivitas dari array M
    max_profit = max(M)
    max_index = M.index(max_profit)
    selected_activities = []
    
    while max_index >= 0:
        selected_activities.append(activities[max_index])
        current_profit = M[max_index]
        
        j = max_index - 1
        while j >= 0:
            if activities[j][0] + activities[j][2] <= activities[max_index][0] and M[j] == current_profit - activities[max_index][1]:
                max_index = j
                break
            j -= 1
        
        if j < 0:
            break
    
    selected_activities.reverse()  # Urutan dari awal ke akhir
    return max_profit, selected_activities

# Data Aktivitas
Xi = [1, 2, 3, 4, 5, 6, 7]
Pi = [50, 75, 25, 30, 60, 45, 25]
Di = [12, 13, 11, 12, 13, 9, 10]

# Memanggil fungsi activity_selection untuk mendapatkan hasil
max_profit, selected_activities = activity_selection(Xi, Pi, Di)

# Menampilkan hasil
print("Keuntungan maksimum yang dapat diperoleh:", max_profit)
print("Urutan aktivitas yang memberikan keuntungan maksimum:")
for activity in selected_activities:
    print("Aktivitas {}: Keuntungan = {}, Waktu Selesai = {}, Durasi = {}".format(activity[0], activity[1], activity[0] + activity[2], activity[2]))