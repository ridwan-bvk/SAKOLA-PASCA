def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        i = j = k = 0

        # Menggabungkan dua bagian menjadi arr dalam urutan menurun
        while i < len(left_half) and j < len(right_half):
            if left_half[i] >= right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        # Menyalin elemen i yang tersisa dari right_half, jika ada
        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        # CMenyalin elemen j yang tersisa dari right_half, jika ada
        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

# array dari soal
arr = [5, 1, 6, 2, 3, 4, 7]
merge_sort(arr)
print("Array setelah diurutkan secara menurun (desc):", arr)