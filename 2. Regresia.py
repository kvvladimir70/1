import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt

class RegressionApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Регрессия")
        
        self.create_initial_widgets()
        
        self.root.mainloop()
    
    def create_initial_widgets(self):
        self.frame_initial = tk.Frame(self.root)
        self.frame_initial.pack(padx=10, pady=10)
        
        tk.Label(self.frame_initial, text='Количество характеристик:').grid(row=0, column=0, sticky='e')
        self.entry_characteristics = tk.Entry(self.frame_initial)
        self.entry_characteristics.grid(row=0, column=1)
        
        tk.Label(self.frame_initial, text='Константа (интерцепт):').grid(row=1, column=0, sticky='e')
        self.entry_intercept = tk.Entry(self.frame_initial)
        self.entry_intercept.grid(row=1, column=1)
        
        self.button_continue = tk.Button(self.frame_initial, text='Продолжить', command=self.continue_to_weights)
        self.button_continue.grid(row=2, column=0, columnspan=2, pady=(10, 0))
    
    def continue_to_weights(self):
        try:
            self.n_features = int(self.entry_characteristics.get())
            if self.n_features <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", 'Количество характеристик должно быть положительным целым числом.')
            return
        
        try:
            self.intercept = float(self.entry_intercept.get())
        except ValueError:
            messagebox.showerror("Ошибка", 'Константа должна быть числом.')
            return
        
        # Destroy initial frame
        self.frame_initial.destroy()
        
        # Proceed to weights input
        self.create_weights_widgets()
    
    def create_weights_widgets(self):
        self.frame_weights = tk.Frame(self.root)
        self.frame_weights.pack(padx=10, pady=10)
        
        tk.Label(self.frame_weights, text='Введите веса и количество переменных для каждой характеристики').grid(row=0, column=0, columnspan=5)
        
        self.weight_entries = []
        self.var_count_entries = []
        
        for i in range(self.n_features):
            tk.Label(self.frame_weights, text=f'Характеристика {i+1}').grid(row=i+1, column=0)
            tk.Label(self.frame_weights, text='Вес:').grid(row=i+1, column=1)
            weight_entry = tk.Entry(self.frame_weights)
            weight_entry.grid(row=i+1, column=2)
            self.weight_entries.append(weight_entry)
            
            tk.Label(self.frame_weights, text='Количество переменных:').grid(row=i+1, column=3)
            var_count_entry = tk.Entry(self.frame_weights)
            var_count_entry.grid(row=i+1, column=4)
            self.var_count_entries.append(var_count_entry)
        
        self.button_calculate = tk.Button(self.frame_weights, text='Рассчитать', command=self.calculate)
        self.button_calculate.grid(row=self.n_features+1, column=0, columnspan=5, pady=(10, 0))
    
    def calculate(self):
        self.weights = []
        self.feature_counts = []
        
        for i in range(self.n_features):
            try:
                weight = float(self.weight_entries[i].get())
                self.weights.append(weight)
            except ValueError:
                messagebox.showerror("Ошибка", f'Вес характеристики {i+1} должен быть числом.')
                return
            try:
                var_count = int(self.var_count_entries[i].get())
                if var_count <= 0:
                    raise ValueError
                self.feature_counts.append(var_count)
            except ValueError:
                messagebox.showerror("Ошибка", f'Количество переменных для характеристики {i+1} должно быть положительным целым числом.')
                return
        
        # Now proceed to calculations
        self.perform_regression()
    
    def perform_regression(self):
        # For demonstration, let's create X as a random binary matrix
        total_vars = sum(self.feature_counts)
        n_samples = 100  # Let's assume a fixed number of samples
        X = np.random.randint(0, 2, size=(n_samples, total_vars))
        
        # Generate weights vector
        weights_expanded = []
        idx = 0
        for i, count in enumerate(self.feature_counts):
            weights_expanded.extend([self.weights[i]] * count)
        weights_array = np.array(weights_expanded)
        
        # Calculate probabilities
        linear_combination = self.intercept + np.dot(X, weights_array)
        probabilities = 1 / (1 + np.exp(-linear_combination))
        
        # For the target variable y, let's use a threshold
        y = (probabilities >= 0.5).astype(int)
        
        # Display output in application window
        self.display_results(X, y, probabilities)
        
        # Plot the results
        self.plot_results(probabilities)
    
    def display_results(self, X, y, probabilities):
        self.frame_weights.destroy()
        
        self.frame_results = tk.Frame(self.root)
        self.frame_results.pack(padx=10, pady=10)
        
        # For demonstration, just show first 10 samples
        tk.Label(self.frame_results, text='Первые 10 результатов:').pack()
        text = tk.Text(self.frame_results, width=80, height=20)
        text.pack()
        
        for i in range(10):
            text.insert(tk.END, f"X[{i}]: {X[i]}\n")
            text.insert(tk.END, f"y[{i}]: {y[i]}\n")
            text.insert(tk.END, f"P(y=1): {probabilities[i]:.4f}\n")
            text.insert(tk.END, "\n")
        
        self.button_exit = tk.Button(self.frame_results, text='Выйти', command=self.root.destroy)
        self.button_exit.pack(pady=(10, 0))
    
    def plot_results(self, probabilities):
        # Plotting a histogram of probabilities
        plt.figure(figsize=(6,4))
        plt.hist(probabilities, bins=20, edgecolor='k')
        plt.title('Распределение вероятностей')
        plt.xlabel('Вероятность')
        plt.ylabel('Частота')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    app = RegressionApp()
