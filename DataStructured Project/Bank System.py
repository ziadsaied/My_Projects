# Node class
class Node:
    def __init__(self, data):
        self.data = {"Name": data["Name"], "Money": data["Money"]}  # Account details
        self.next = None


# LinkedList class
class LinkedList:
    def __init__(self):
        self.head = None

    # Add a new account (node)
    def insert(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
        else:
            temp = self.head
            while temp.next:
                temp = temp.next
            temp.next = new_node

    # Search for a node by name
    def search(self, name):
        temp = self.head
        while temp:
            if temp.data["Name"].lower() == name.lower():
                return temp  # Return the node
            temp = temp.next
        return None  # Return None if not found

    # Delete a node by name
    def delete(self, name):
        name_lower = name.lower()

        # Check if node is the head
        if self.head and self.head.data["Name"].lower() == name_lower:
            temp = self.head
            self.head = self.head.next
            temp = None
            return True

        # Traverse to find and delete the node
        temp = self.head
        while temp and temp.next:
            if temp.next.data["Name"].lower() == name_lower:
                temp.next = temp.next.next
                return True
            temp = temp.next

        return False  # Return False if account not found

    # Display all accounts
    def display(self):
        accounts_list = []
        temp = self.head
        while temp:
            accounts_list.append(temp.data)  # Add the account data to the list
            temp = temp.next
        return accounts_list  # Return the list of accounts


# Bank class
class Bank:
    def __init__(self):
        self.accounts = LinkedList()  # Use LinkedList to store accounts

    # Add a new account
    def add_account(self, name, initial_money):
        name = name.strip().lower()
        if self.accounts.search(name):
            return f"Account '{name}' already exists."
        self.accounts.insert({"Name": name, "Money": initial_money})
        return f"Account '{name}' created with balance {initial_money}."

    # Delete an account
    def delete_account(self, name):
        name = name.strip().lower()
        if self.accounts.delete(name):
            return f"Account '{name}' deleted."
        return f"Account '{name}' not found."

    # Deposit money into an account
    def deposit(self, name, amount):
        name = name.strip().lower()
        account = self.accounts.search(name)
        if account:
            account.data["Money"] += amount
            return f"Deposited {amount} into '{name}'. New balance: {account.data['Money']}"
        return f"Account '{name}' not found."

    # Withdraw money from an account
    def withdraw(self, name, amount):
        name = name.strip().lower()
        account = self.accounts.search(name)
        if account:
            if account.data["Money"] >= amount:
                account.data["Money"] -= amount
                return f"Withdrew {amount} from '{name}'. New balance: {account.data['Money']}"
            return f"Insufficient balance in account '{name}'."
        return f"Account '{name}' not found."

    # Transfer money between accounts
    def transfer(self, source_name, destination_name, amount):
        source_name = source_name.strip().lower()
        destination_name = destination_name.strip().lower()

        if amount <= 0:
            return "Transfer amount must be positive."

        from_account = self.accounts.search(source_name)
        to_account = self.accounts.search(destination_name)

        if not from_account:
            return f"Source account '{source_name}' not found."

        if not to_account:
            return f"Destination account '{destination_name}' not found."

        if from_account.data["Money"] < amount:
            return f"Insufficient balance in source account '{source_name}'."

        from_account.data["Money"] -= amount
        to_account.data["Money"] += amount
        return f"Transferred {amount} from '{source_name}' to '{destination_name}'.\n" \
               f"New balance - {source_name}: {from_account.data['Money']}, {destination_name}: {to_account.data['Money']}"

    # Display all accounts
    def display_accounts(self):
        accounts = self.accounts.display()
        if not accounts:
            return "No accounts found."
        result = []
        for account in accounts:
            result.append(f"Name: {account['Name']}, Balance: {account['Money']}")
        return "\n".join(result)



bank = Bank()

while True:
        print("\n# Bank System :")
        print("1 - Create Account")
        print("2 - Deposit")
        print("3 - Withdraw")
        print("4 - Transfer")
        print("5 - Delete Account")
        print("6 - Display Accounts")
        print("7 - Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            name = input("Enter account holder's name: ")
            initial_money = float(input("Enter initial deposit amount: "))
            print(bank.add_account(name, initial_money))

        elif choice == "2":
            name = input("Enter account holder's name: ")
            amount = float(input("Enter deposit amount: "))
            print(bank.deposit(name, amount))

        elif choice == "3":
            name = input("Enter account holder's name: ")
            amount = float(input("Enter withdrawal amount: "))
            print(bank.withdraw(name, amount))

        elif choice == "4":
            source_name = input("Enter source account holder's name: ")
            destination_name = input("Enter destination account holder's name: ")
            amount = float(input("Enter transfer amount: "))
            print(bank.transfer(source_name, destination_name, amount))

        elif choice == "5":
            name = input("Enter account holder's name to delete: ")
            print(bank.delete_account(name))

        elif choice == "6":
            accounts = bank.display_accounts()
            print(accounts)

        elif choice == "7":
            print("Exiting the system. Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")
