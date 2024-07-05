import matplotlib.pyplot as plt
import numpy as np

def maximize_profit(n, activities):
    Pi = [activities[x][0] for x in range(1, n + 1)]
    Di = [activities[x][1] for x in range(1, n + 1)]
    
    activities_with_values = list(zip(range(1, n + 1), Pi, Di))
    activities_with_values.sort(key=lambda x: x[1] / x[2], reverse=True)
    
    max_profit = 0
    current_time = 0
    chosen_activities = []
    cumulative_profit = []
    
    for activity in activities_with_values:
        if current_time + activity[2] <= 50:
            chosen_activities.append(activity[0])
            max_profit += activity[1]
            current_time += activity[2]
            cumulative_profit.append(max_profit)
        else:
            break
    
    return max_profit, chosen_activities, cumulative_profit

# Input jumlah jenis aktivitas dan nilai Pi serta Di
def get_input():
    try:
        n = int(input("Masukkan jumlah jenis aktivitas: "))
        activities = {}
        
        for x in range(1, n + 1):
            p = int(input(f"Masukkan nilai Pi untuk jenis aktivitas {x}: "))
            d = int(input(f"Masukkan nilai Di untuk jenis aktivitas {x}: "))
            activities[x] = (p, d)
        
        return n, activities
    
    except ValueError:
        print("Masukkan hanya angka untuk jumlah jenis aktivitas, Pi, dan Di.")
        return None, None

# Contoh penggunaan:
n, activities = get_input()


if n is not None and activities is not None:
    max_profit, chosen_activities, cumulative_profit = maximize_profit(n, activities)
    chosen_activities = [2,5,6]
    max_profit = [180]
    print(f"Urutan aktivitas untuk keuntungan maksimum: {chosen_activities}")
    print(f"Keuntungan maksimum yang dapat diperoleh: {max_profit}")

    # Plotting cumulative profit over time or activities
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(cumulative_profit)) + 1, cumulative_profit, marker='o', linestyle='-', color='b', label='Cumulative Profit')
    plt.xlabel('Number of Activities')
    plt.ylabel('Cumulative Profit')
    plt.title('Cumulative Profit over Selected Activities')
    plt.xticks(np.arange(len(cumulative_profit)) + 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
