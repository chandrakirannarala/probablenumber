import tkinter as tk
from tkinter import Canvas, Button
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

class DigitClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Digit Classifier")
        
        # Load your saved model
        self.model = tf.keras.models.load_model('handwritten_num_reader.keras')
        
        # Create 28x28 drawing canvas (scaled up for visibility)
        self.canvas_size = 280  # Display size (10x scale of 28x28)
        self.canvas = Canvas(root, width=self.canvas_size, height=self.canvas_size, bg='black')
        self.canvas.pack(side=tk.TOP, pady=2)
        
        # Setup drawing variables
        self.last_x = None
        self.last_y = None
        self.line_width = 20  # Thicker lines for better downscaling
        self.color = 'white'
        
        # Bind mouse events
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.reset_position)
        
        # Create buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(side=tk.BOTTOM, pady=2)
        
        Button(btn_frame, text="Evaluate", command=self.predict_digit).pack(side=tk.LEFT, padx=5)
        Button(btn_frame, text="Reset", command=self.clear_canvas).pack(side=tk.LEFT, padx=5)
        
        # Prediction display
        self.prediction_label = tk.Label(root, text="Draw a digit...", font=('Helvetica', 18))
        self.prediction_label.pack(pady=2)

        # Create off-screen image for accurate pixel data
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw_image = ImageDraw.Draw(self.image)

    def draw(self, event):
        if self.last_x and self.last_y:
            # Draw on both canvas and off-screen image
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                   width=self.line_width, fill=self.color,
                                   capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw_image.line([(self.last_x, self.last_y), (event.x, event.y)],
                                fill=255, width=self.line_width)
        self.last_x = event.x
        self.last_y = event.y

    def reset_position(self, event):
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw_image = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="Draw a digit...")

    def preprocess_image(self):
        # Directly use the off-screen image (no screen scraping)
        img = self.image.resize((28, 28), Image.LANCZOS)  # High-quality downsampling
        
        # Convert to numpy array and normalize
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)  # Add batch and channel dimensions
        
        return img_array

    def predict_digit(self):
        img_array = self.preprocess_image()
        prediction = self.model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)
        self.prediction_label.config(text=f"Prediction: {digit} (Confidence: {confidence:.2%})")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitClassifierApp(root)
    root.mainloop()