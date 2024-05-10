def Induksi_Matematika(n):
    if int(n) == 1:
        print("Pernyataan tidak berlaku untuk n = 1.")
        return
    for i in range(1, int(n)+1):
        if (3**n) % 8 != 0 or (5**n) % 8 != 0:
            print("Pernyataan tidak berlaku untuk n =\n")
            print(i)
            return
    else:
        print("Pernyataan benar untuk semua bilangan bulat positif hingga",n)

def main():
    try:
        n = int(input("Masukkan nilai n: "))
        if n <= 0:
            print("Masukkan bilangan bulat positif yang lebih besar dari 0.")
            return
        Induksi_Matematika(n)
    except ValueError:
        print("Masukkan bilangan bulat.")

if __name__ == "__main__":
    main()
