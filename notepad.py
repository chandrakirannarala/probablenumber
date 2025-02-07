import tkinter as tk
import numpy as np
import tensorflow as tf

class SimpleDigitClassifier:
    def __init__(self, root):
        self.countw = 0
        self.countr = 0
        self.root = root
        self.root.title("28x28 Digit Classifier")
        
        # Load model
        self.model = tf.keras.models.load_model('handwritten_num_reader.keras')
        
        # Create 28x28 canvas (scaled 10x for visibility)
        self.cell_size = 10
        self.canvas = tk.Canvas(root, 
                              width=28*self.cell_size, 
                              height=28*self.cell_size,
                              bg='black')
        self.canvas.pack(pady=10)
        
        # Initialize pixel grid
        self.grid = np.zeros((28, 28), dtype=np.float32)
        
        # Set up drawing
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        
        # Controls
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)
        
        tk.Button(btn_frame, text="Predict", command=self.predict).pack(side=tk.LEFT, padx=5)
        # tk.Button(btn_frame, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Right", command=self.right).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Wrong", command=self.wrong).pack(side=tk.LEFT, padx=5)
        
        self.label = tk.Label(root, text="Draw a digit", font=('Arial', 14))
        self.label.pack()

    def paint(self, event):
        # Convert mouse position to grid coordinates
        x = event.x // self.cell_size
        y = event.y // self.cell_size
        
        # Mark pixel and draw rectangle
        if 0 <= x < 28 and 0 <= y < 28:
            self.grid[y][x] = 1.0  # MNIST-style (white on black)
            self.canvas.create_rectangle(
                x*self.cell_size, y*self.cell_size,
                (x+1)*self.cell_size, (y+1)*self.cell_size,
                fill='white', outline='white'
            )

    def reset(self, event):
        self.prev_x, self.prev_y = None, None

    # def clear(self):
    #     self.canvas.delete("all")
    #     self.grid = np.zeros((28, 28), dtype=np.float32)
    #     self.label.config(text="Draw a digit")

    def right(self):
        self.countr += 1
        self.canvas.delete("all")
        self.grid = np.zeros((28, 28), dtype=np.float32)
        self.label.config(text="Draw a digit")
    
    def wrong(self):
        self.countw += 1
        self.canvas.delete("all")
        self.grid = np.zeros((28, 28), dtype=np.float32)
        self.label.config(text="Draw a digit")

    def predict(self):
        # Reshape for model input (28x28 -> 1x28x28x1)
        input_data = self.grid.reshape(1, 28, 28, 1)
        
        # Make prediction
        prediction = self.model.predict(input_data)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        self.label.config(text=f"Prediction: {digit} ({confidence:.0%} sure)")

if __name__ == "__main__":
    root = tk.Tk()
    app = SimpleDigitClassifier(root)
    root.mainloop()
    print(f"accuracy is {app.countr/(app.countr+app.countw):.0%}")
    