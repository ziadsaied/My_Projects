graph = { 
    'Fever': ['Flu', 'Malaria', 'Dengue', 'Food Poisoning', 'Measles'],
    'Food Poisoning': ['Fever'],
    'Flu': ['Cough', 'Fatigue', 'Fever', 'Chills'],
    'Cough': ['Measles', 'Flu'],
    'Fatigue': ['Flu', 'Dengue'],
    'Chills': ['Flu', 'Malaria'],
    'Measles': ['Cough', 'Rash', 'Fever'],
    'Rash': ['Measles', 'Dengue'],
    'Malaria': ['Headache', 'Fever', 'Chills'],
    'Headache': ['Dengue', 'Malaria'],
    'Dengue': ['Fatigue', 'Rash', 'Fever', 'Headache']
}

def BFS(graph, start, goal):
    queue = [[start]]
    visited = []
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node in visited:
            continue
        visited.append(node)
        if node == goal:
            return path
        adj_nodes = graph.get(node, [])
        for node2 in adj_nodes:
            queue.append(path + [node2])

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier

data = pd.read_csv(r'D:\Collage\LEVEL 2\Sem 1\Artificial Intelligence Principles and Techniques\Project\Disease_Data.csv')

data_input = data.drop(columns='Outcome')
data_output = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(data_input, data_output, test_size=0.15, random_state=0)

clf = DecisionTreeClassifier(max_depth=2, random_state=0)
clf = clf.fit(X_train, y_train)
y_pred_dt = clf.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt) * 100
precision_dt = precision_score(y_test, y_pred_dt, average='weighted') * 100
recall_dt = recall_score(y_test, y_pred_dt, average='weighted') * 100
f1_dt = f1_score(y_test, y_pred_dt, average='weighted') * 100

mlp = MLPClassifier(
    hidden_layer_sizes=(10, 10),
    max_iter=1000,
    random_state=0,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    alpha=0.02,
    early_stopping=True
)

mlp.fit(X_train, y_train)
y_pred_bp = mlp.predict(X_test)

accuracy_bp = accuracy_score(y_test, y_pred_bp) * 100
precision_bp = precision_score(y_test, y_pred_bp, average='weighted') * 100
recall_bp = recall_score(y_test, y_pred_bp, average='weighted') * 100
f1_bp = f1_score(y_test, y_pred_bp, average='weighted') * 100

start_symptom = "Dengue"
end_symptom = "Cough"
result = BFS(graph, start_symptom, end_symptom)

print("1. Decision Tree:")
print(f"   - Accuracy: {accuracy_dt:.2f}%")
print(f"   - F1 Score: {f1_dt:.2f}%")
print(f"   - Recall: {recall_dt:.2f}%")
print(f"   - Precision: {precision_dt:.2f}%")

print("\n2. Backpropagation:")
print(f"   - Accuracy: {accuracy_bp:.2f}%")
print(f"   - F1 Score: {f1_bp:.2f}%")
print(f"   - Recall: {recall_bp:.2f}%")
print(f"   - Precision: {precision_bp:.2f}%")

print("\n3. Path:")
if result:
    print(f"   - Path from {start_symptom} to {end_symptom} is {result}")
else:
    print(f"   - No path found from {start_symptom} to {end_symptom}")
