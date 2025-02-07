import tkinter as tk
import numpy as np
import tensorflow as tf

class TruePixelClassifier:
    def __init__(self, root):
        self.root = root
        self.root.title("True 28x28 Classifier")
        
        # Force window size (may not work on all OS)
        self.root.geometry("32x60")  # Canvas + buttons
        
        # Load model
        self.model = tf.keras.models.load_model('handwritten_num_reader.keras')
        
        # True 28x28 canvas
        self.canvas = tk.Canvas(root, width=28, height=28, bg='black', 
                              highlightthickness=0, cursor="crosshair")
        self.canvas.pack()
        
        # Initialize pixel grid
        self.grid = np.zeros((28, 28), dtype=np.float32)
        
        # Drawing setup
        self.canvas.bind("<Motion>", self.paint)  # Draw while moving mouse
        self.canvas.bind("<Button-1>", self.paint)
        
        # Tiny buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack()
        
        tk.Button(btn_frame, text="P", command=self.predict, width=1).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="C", command=self.clear, width=1).pack(side=tk.LEFT)

    def paint(self, event):
        # Direct pixel coordinates (no scaling)
        x = min(max(event.x, 0), 27)
        y = min(max(event.y, 0), 27)
        
        # Update grid and canvas
        self.grid[y][x] = 1.0
        self.canvas.create_rectangle(x, y, x+1, y+1, 
                                   outline='white', fill='white', width=0)

    def clear(self):
        self.canvas.delete("all")
        self.grid = np.zeros((28, 28), dtype=np.float32)

    def predict(self):
        input_data = self.grid.reshape(1, 28, 28, 1)
        prediction = self.model.predict(input_data)
        digit = np.argmax(prediction)
        print(f"Prediction: {digit}")  # No space for label - print to console

if __name__ == "__main__":
    root = tk.Tk()
    app = TruePixelClassifier(root)
    root.mainloop()