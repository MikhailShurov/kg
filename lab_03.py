import tkinter as tk
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches


class RasterizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rasterization Algorithms")
        self.root.geometry("600x600")

        # Create the interface
        self.label = tk.Label(root, text="Choose an Algorithm:")
        self.label.pack(pady=10)

        # Entry fields for line coordinates
        self.start_x_label = tk.Label(root, text="Start X:")
        self.start_x_label.pack(pady=5)
        self.start_x_entry = tk.Entry(root)
        self.start_x_entry.pack(pady=5)

        self.start_y_label = tk.Label(root, text="Start Y:")
        self.start_y_label.pack(pady=5)
        self.start_y_entry = tk.Entry(root)
        self.start_y_entry.pack(pady=5)

        self.end_x_label = tk.Label(root, text="End X:")
        self.end_x_label.pack(pady=5)
        self.end_x_entry = tk.Entry(root)
        self.end_x_entry.pack(pady=5)

        self.end_y_label = tk.Label(root, text="End Y:")
        self.end_y_label.pack(pady=5)
        self.end_y_entry = tk.Entry(root)
        self.end_y_entry.pack(pady=5)

        self.draw_line_button = tk.Button(root, text="Draw Line", command=self.draw_line)
        self.draw_line_button.pack(pady=5)

        self.circle_button = tk.Button(root, text="Bresenham Circle", command=self.bresenham_circle)
        self.circle_button.pack(pady=5)

        self.quit_button = tk.Button(root, text="Quit", command=root.quit)
        self.quit_button.pack(pady=20)

        # Widget for the plot
        self.figure = Figure(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def draw_line(self):
        try:
            x1 = int(self.start_x_entry.get())
            y1 = int(self.start_y_entry.get())
            x2 = int(self.end_x_entry.get())
            y2 = int(self.end_y_entry.get())

            points = self.bresenham(x1, y1, x2, y2)
            self.plot_points(points, "Bresenham Line", x1, y1, x2, y2)
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid integer coordinates.")

    def bresenham_circle(self):
        try:
            x_center = int(self.start_x_entry.get())
            y_center = int(self.start_y_entry.get())
            radius = int(self.end_x_entry.get())  # Use End X as radius

            points = self.bresenham_circle_algorithm(x_center, y_center, radius)
            self.plot_points(points, "Bresenham Circle", x_center - radius, y_center - radius,
                             x_center + radius, y_center + radius)
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid integer coordinates.")

    def bresenham(self, x1, y1, x2, y2):
        points = []
        dx = x2 - x1
        dy = y2 - y1
        sx = 1 if dx > 0 else -1
        sy = 1 if dy > 0 else -1
        dx = abs(dx)
        dy = abs(dy)

        if dx > dy:
            err = dx / 2.0
            while x1 != x2:
                points.append((x1, y1))
                err -= dy
                if err < 0:
                    y1 += sy
                    err += dx
                x1 += sx
        else:
            err = dy / 2.0
            while y1 != y2:
                points.append((x1, y1))
                err -= dx
                if err < 0:
                    x1 += sx
                    err += dy
                y1 += sy
        points.append((x2, y2))
        return points

    def bresenham_circle_algorithm(self, x_center, y_center, radius):
        points = []
        x = radius
        y = 0
        decision_parameter = 1 - radius

        while x >= y:
            points.extend([
                (x_center + x, y_center + y),
                (x_center - x, y_center + y),
                (x_center + x, y_center - y),
                (x_center - x, y_center - y),
                (x_center + y, y_center + x),
                (x_center - y, y_center + x),
                (x_center + y, y_center - x),
                (x_center - y, y_center - x)
            ])
            y += 1
            if decision_parameter <= 0:
                decision_parameter += 2 * y + 1
            else:
                x -= 1
                decision_parameter += 2 * (y - x) + 1

        return points

    def plot_points(self, points, title, x_min, y_min, x_max, y_max):
        self.figure.clear()  # Clear the previous plot
        ax = self.figure.add_subplot(111)

        # Draw squares instead of points
        square_size = 1  # Size of the square
        for (x, y) in points:
            square = patches.Rectangle((x - square_size / 2, y - square_size / 2),
                                       square_size, square_size,
                                       color='red')
            ax.add_patch(square)

        ax.set_xlim(x_min - 10, x_max + 10)  # Add some padding
        ax.set_ylim(y_min - 10, y_max + 10)  # Add some padding
        ax.axhline(0, color='black', linewidth=0.5, ls='--')
        ax.axvline(0, color='black', linewidth=0.5, ls='--')
        ax.grid()
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='box')
        self.canvas.draw()  # Update the plot on the widget


if __name__ == "__main__":
    root = tk.Tk()
    app = RasterizationApp(root)
    root.mainloop()
