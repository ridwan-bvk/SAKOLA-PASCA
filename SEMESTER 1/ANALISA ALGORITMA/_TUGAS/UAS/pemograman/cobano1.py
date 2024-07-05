# Daftar aktivitas
activities = [
    {"Xi": 1, "Pi": 50, "Di": 12},
    {"Xi": 2, "Pi": 75, "Di": 13},
    {"Xi": 3, "Pi": 25, "Di": 11},
    {"Xi": 4, "Pi": 30, "Di": 12},
    {"Xi": 5, "Pi": 60, "Di": 13},
    {"Xi": 6, "Pi": 45, "Di": 9},
    {"Xi": 7, "Pi": 25, "Di": 10}
]

# Tambahkan waktu penyelesaian
for activity in activities:
    activity["finish"] = activity["Xi"] + activity["Di"]

# Urutkan aktivitas berdasarkan keuntungan menurun dan waktu penyelesaian
activities.sort(key=lambda x: (-x["Pi"], x["finish"]))

# Fungsi untuk menemukan aktivitas yang tidak bertumpuk terakhir
def find_last_non_conflicting(activities, n):
    for j in range(n - 1, -1, -1):
        if activities[j]["finish"] <= activities[n]["Xi"]:
            return j
    return -1

# Array dp untuk menyimpan keuntungan maksimum hingga aktivitas ke-i
n = len(activities)
dp = [0] * n
dp[0] = activities[0]["Pi"]

for i in range(1, n):
    incl_profit = activities[i]["Pi"]
    l = find_last_non_conflicting(activities, i)
    if l != -1:
        incl_profit += dp[l]
    dp[i] = max(incl_profit, dp[i-1])

# Keuntungan maksimum
max_profit = dp[n-1]

# Rekonstruksi urutan aktivitas
selected_activities = []
i = n - 1
while i >= 0:
    if i == 0 or dp[i] != dp[i-1]:
        selected_activities.append(activities[i])
        i = find_last_non_conflicting(activities, i)
    else:
        i -= 1

# Balikkan urutan untuk mendapatkan urutan asli
selected_activities.reverse()

max_profit, selected_activities, activities


print(max_profit)
# print(selected_activities)
print(activities)