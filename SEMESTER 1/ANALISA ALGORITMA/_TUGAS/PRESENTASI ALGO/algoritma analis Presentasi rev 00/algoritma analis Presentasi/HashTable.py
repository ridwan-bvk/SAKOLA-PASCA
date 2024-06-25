class HashTable:
    def __init__(self, size):
        self.size = size
        self.table = [None] * size

    def _hash(self, key):
        # Simple hash function
        return hash(key) % self.size

    def insert(self, key, value):
        index = self._hash(key)
        if self.table[index] is None:
            self.table[index] = []
        for kvp in self.table[index]:
            if kvp[0] == key:
                kvp[1] = value
                return
        self.table[index].append([key, value])
        print(f"Inserted ({key}: {value})")

    def search(self, key):
        index = self._hash(key)
        if self.table[index] is not None:
            for kvp in self.table[index]:
                if kvp[0] == key:
                    return kvp[1]
        return None

    def delete(self, key):
        index = self._hash(key)
        if self.table[index] is not None:
            for i, kvp in enumerate(self.table[index]):
                if kvp[0] == key:
                    del self.table[index][i]
                    print(f"Deleted ({key})")
                    return
        print(f"Key ({key}) not found")

    def display(self):
        for i, slot in enumerate(self.table):
            if slot:
                print(f"Index {i}: {slot}")
            else:
                print(f"Index {i}: Empty")

def main():
    size = int(input("Enter size of hash table: "))
    ht = HashTable(size)

    while True:
        print("\n1. Insert data")
        print("2. Search data")
        print("3. Delete data")
        print("4. Display data")
        print("5. Exit")
        choice = int(input("Enter your choice: "))

        if choice == 1:
            key = input("Enter key: ")
            value = input("Enter value: ")
            ht.insert(key, value)
        elif choice == 2:
            key = input("Enter key to search: ")
            value = ht.search(key)
            if value:
                print(f"Found: ({key}: {value})")
            else:
                print(f"Key ({key}) not found")
        elif choice == 3:
            key = input("Enter key to delete: ")
            ht.delete(key)
        elif choice == 4:
            ht.display()
        elif choice == 5:
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
