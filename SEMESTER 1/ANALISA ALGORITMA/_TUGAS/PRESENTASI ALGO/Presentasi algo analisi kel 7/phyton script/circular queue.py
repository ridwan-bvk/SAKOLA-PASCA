class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

class CircularQueue:
    def __init__(self):
        self.front = None
        self.rear = None

    def is_empty(self):
        return self.front is None

    def enqueue(self, data):
        new_node = Node(data)
        if self.is_empty():
            self.front = self.rear = new_node
            self.front.next = self.front.prev = self.front
        else:
            new_node.prev = self.rear
            new_node.next = self.front
            self.rear.next = new_node
            self.front.prev = new_node
            self.rear = new_node
        print(f"Enqueued: {data}")

    def dequeue(self):
        if self.is_empty():
            print("Queue is empty")
            return None
        data = self.front.data
        if self.front == self.rear:
            self.front = self.rear = None
        else:
            self.front = self.front.next
            self.front.prev = self.rear
            self.rear.next = self.front
        print(f"Dequeued: {data}")
        return data

    def display(self):
        if self.is_empty():
            print("Queue is empty")
            return
        print("Queue elements:")
        current = self.front
        while True:
            print(current.data, end=" ")
            if current == self.rear:
                break
            current = current.next
        print()

def main():
    cq = CircularQueue()

    while True:
        print("\nMenu:")
        print("1. Enqueue")
        print("2. Dequeue")
        print("3. Display")
        print("4. Exit")
        choice = int(input("Silahkan pilih : "))

        if choice == 1:
            data = int(input("masukan nilai enqueue: "))
            cq.enqueue(data)
        elif choice == 2:
            cq.dequeue()
        elif choice == 3:
            cq.display()
        elif choice == 4:
            print("Exiting...")
            break
        else:
            print("Tidak ada dalam menu,silahkan coba kembali.")

if __name__ == "__main__":
    main()
