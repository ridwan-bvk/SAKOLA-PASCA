class CircularQueue:
    def __init__(self, size):
        self.size = size
        self.queue = [None] * size
        self.front = self.rear = -1

    def is_full(self):
        return (self.rear + 1) % self.size == self.front

    def is_empty(self):
        return self.front == -1

    def enqueue(self, data):
        if self.is_full():
            print("Queue is full")
        else:
            if self.front == -1:
                self.front = 0
            self.rear = (self.rear + 1) % self.size
            self.queue[self.rear] = data

    def dequeue(self):
        if self.is_empty():
            print("Queue is empty")
        else:
            data = self.queue[self.front]
            if self.front == self.rear:
                self.front = self.rear = -1
            else:
                self.front = (self.front + 1) % self.size
            return data

    def display(self):
        if self.is_empty():
            print("Queue is empty")
        else:
            idx = self.front
            while True:
                print(self.queue[idx], end=" ")
                if idx == self.rear:
                    break
                idx = (idx + 1) % self.size
            print()

# Contoh penggunaan
cq = CircularQueue(5)
cq.enqueue(1)
cq.enqueue(2)
cq.enqueue(3)
cq.enqueue(4)
cq.display()
cq.dequeue()
cq.display()
cq.enqueue(5)
cq.enqueue(6)
cq.display()
