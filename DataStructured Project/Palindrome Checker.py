import tkinter as tk
from tkinter import messagebox

class Queue:
    def __init__(self):
        self.queue = []

    def enqueue(self, item):
        self.queue.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("Queue is Empty.")
        return self.queue.pop(0)

    def peek(self):
        if self.is_empty():
            raise IndexError("Queue is Empty.")
        return self.queue[0]

    def is_empty(self):
        return len(self.queue) == 0

    def size(self):
        return len(self.queue)

    def __str__(self):
        return "Queue: " + str(self.queue)

class Stack:
    def __init__(self):
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("Stack is Empty.")
        return self.stack.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("Stack is Empty.")
        return self.stack[-1]

    def is_empty(self):
        return len(self.stack) == 0

    def size(self):
        return len(self.stack)

    def _str_(self):
        return "Stack: " + str(self.stack)

class Palindrome:
    def __init__(self, user_input):
        self.UserInput = user_input

    def Palindrome_Checker(self):
        stc = Stack()
        qeu = Queue()
        for item in self.UserInput:
            stc.push(item)
            qeu.enqueue(item)
        for _ in range(stc.size()):
            if stc.peek() != qeu.peek():
                return "Not a Palindrome"
            stc.pop()
            qeu.pop()
        return "Palindrome"

class PalindromeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Palindrome Checker")
        self.root.geometry("400x300")
        self.root.configure(bg="#f0f0f0")
        self.title_label = tk.Label(root, text="Palindrome Checker", font=("Helvetica", 16, "bold"), bg="#f0f0f0")
        self.title_label.pack(pady=20)
        self.label = tk.Label(root, text="Enter a string:", font=("Helvetica", 12), bg="#f0f0f0")
        self.label.pack()
        self.entry = tk.Entry(root, font=("Helvetica", 12), width=30)
        self.entry.pack(pady=10)
        self.check_button = tk.Button(root, text="Check", command=self.check_palindrome, font=("Helvetica", 12), bg="#4CAF50", fg="white", activebackground="#45a049")
        self.check_button.pack(pady=10)
        self.result_label = tk.Label(root, text="", font=("Helvetica", 12), bg="#f0f0f0")
        self.result_label.pack(pady=20)

    def check_palindrome(self):
        user_input = self.entry.get()
        if not user_input:
            messagebox.showwarning("Input Error", "Please enter a string.")
            return
        palindrome_checker = Palindrome(user_input)
        result = palindrome_checker.Palindrome_Checker()
        self.result_label.config(text=result)

if __name__ == "__main__":
    root = tk.Tk()
    app = PalindromeApp(root)
    root.mainloop()