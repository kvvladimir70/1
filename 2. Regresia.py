import tkinter as tk
from tkinter import messagebox
import numpy as np
class UserInputDialog:
    def init(self):
        self.root = tk.Tk()
        self.root.title("Регрессия")
    def enter(self):
        frame = tk.LabelFrame(self.root, text='1')
        frame.pack(padx=8, pady=8)
        tk.Label(frame, text='Характеристики').grid(row=0, column=0)
        tk.Label(frame, text='Константа').grid(row=2, column=0)
        self.entry_characteristics = tk.Entry(frame)
        self.entry_characteristics.grid(row=0, column=1)
        self.entry_intercept = tk.Entry(frame)
        self.entry_intercept.grid(row=2, column=1)
        return_button = tk.Button(frame, text='Продолжить', command=self.submit)
        return_button.grid(row=3, columnspan=2)
        def validate_input():
            try:
                characteristics = float(self.entry_characteristics.get())
                if not isinstance(characteristics, (int, float)):
                    messagebox.showerror("Ошибка", "Характеристика должна быть числом.")
                    return
                elif not isinstance(characteristics, int):
                    messagebox.showerror("Ошибка", "Характеристика должна быть целым числом.")
                    return
            except ValueError:
                messagebox.showerror("Ошибка", "Характеристика должна быть числом.")
                return
            self.root.mainloop()
        return_button['command'] = validate_input
    def submit(self):
        frame = tk.LabelFrame(self.root, text="Ввод данных")
        frame.pack(padx=8, pady=8)
        characteristics = int(self.entry_characteristics.get().strip())
        feature_counts = range(characteristics)
        vars_per_feature = []
        weights = []
        intercept = float(self.entry_intercept.get())
        labels = []
        entries = []
        for feature in feature_counts:
                label = tk.Label(frame, text=f'Весовой коэффициент {feature + 1}')
                labels.append(label)
                entry = tk.Entry(frame)
                entries.append(entry)
                label.grid(row=feature + 4, column=0)
                entry.grid(row=feature + 4, column=1)
                value = entry.get()
                if value != '':
                    try:
                        weight = float(value)
                    except ValueError:
                        messagebox.showerror("Ошибка", f"Неправильный формат веса для {feature + 1}. Должно быть число.")
                        return
                    weights.append(weight)
                else:
                    messagebox.showerror("Ошибка", f"Весовой коэффициент {feature + 1} не может быть пустым.")
                    return
                label = tk.Label(frame, text=f'Количество переменных для характеристики {feature + 1}')
                label.grid(row=feature + 4, column=2)
                entry = tk.Entry(frame)
                entry.grid(row=feature + 4, column=3)
                var = int(entry.get())
                vars_per_feature.append(var)
        total_vars = sum(vars_per_feature)
        variables = np.ones((total_vars, total_vars))
        create_single_matrix(characteristics, vars_per_feature, total_vars, variables)
        messagebox.showinfo("Успешный ввод данных")
        self.root.destroy()
        self.root.mainloop()
        return characteristics, weights, intercept, vars_per_feature
    def exit(self, exc_type, exc_value, traceback):
        self.root.quit()
        return None
def create_single_matrix(n_test, n_features, vars_per_feature):
    variables = np.zeros((sum(vars_per_feature), sum(vars_per_feature)))
    offset = 0
    matrix = np.zeros((n_test, sum(vars_per_feature)))
    for feature in range(n_features):
        current_vars = vars_per_feature[feature]
        variables[offset:offset+current_vars, offset:offset+current_vars] = 1
        offset += current_vars
    X = np.dot(variables, matrix)
    return X

def calculate_logistic_function(X, feature_counts, intercept):
    probas = intercept + np.dot(X, feature_counts)
    return 1 / (1 + np.exp(-probas))
def main():
    with UserInputDialog() as dialog:
        if not dialog:
            return
        try:
            n_features, weights, intercept, feature_counts = dialog.submit()
        except TypeError:
            return
        n_test = 1
        X = create_single_matrix(n_test, n_features, feature_counts)
        y = np.sum(X, axis=1) % 2
        probabilities = calculate_logistic_function(X, weights, n_features, intercept)
        for i in range(len(X)):
            print(f"X: {X[i]}")
            print(f"y: {y[i]}")
            print(f"Probability: {probabilities[i]:.4f}")
            print()

if __name__ == "__main__":
    main()