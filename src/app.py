# viewer.py
import os
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import pandas as pd

DATA_DIR = "data"
IMG_DIR = "img"

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Options Viewer")
        self.geometry("1000x700")

        # Input frame
        input_frame = ttk.Frame(self)
        input_frame.pack(pady=10)

        ttk.Label(input_frame, text="Ticker:").grid(row=0, column=0, padx=5)
        self.ticker_var = tk.StringVar()
        self.ticker_entry = ttk.Entry(input_frame, textvariable=self.ticker_var, width=10)
        self.ticker_entry.grid(row=0, column=1, padx=5)

        run_btn = ttk.Button(input_frame, text="Run Pipeline", command=self.run_pipeline)
        run_btn.grid(row=0, column=2, padx=5)

        refresh_btn = ttk.Button(input_frame, text="Refresh Files", command=self.refresh_files)
        refresh_btn.grid(row=0, column=3, padx=5)

        # File lists
        files_frame = ttk.Frame(self)
        files_frame.pack(fill="both", expand=True)

        self.csv_list = tk.Listbox(files_frame, width=40)
        self.csv_list.pack(side="left", fill="y", padx=5, pady=5)
        self.csv_list.bind("<<ListboxSelect>>", self.show_csv)

        self.img_list = tk.Listbox(files_frame, width=40)
        self.img_list.pack(side="right", fill="y", padx=5, pady=5)
        self.img_list.bind("<<ListboxSelect>>", self.show_image)

        # Viewer panes
        viewer_frame = ttk.Notebook(self)
        viewer_frame.pack(fill="both", expand=True)

        # CSV tab
        self.csv_frame = ttk.Frame(viewer_frame)
        viewer_frame.add(self.csv_frame, text="CSV Viewer")

        self.tree = ttk.Treeview(self.csv_frame, show="headings")
        self.tree.pack(fill="both", expand=True)

        # Image tab
        self.img_frame = ttk.Frame(viewer_frame)
        viewer_frame.add(self.img_frame, text="Image Viewer")

        self.img_label = ttk.Label(self.img_frame)
        self.img_label.pack(fill="both", expand=True)

        # Initialize
        self.refresh_files()

    def run_pipeline(self):
        ticker = self.ticker_var.get().strip().upper()
        if not ticker:
            messagebox.showerror("Error", "Please enter a ticker.")
            return

        try:
            # 🔹 Change this depending on whether you call Python or Go
            subprocess.run(["python", "src/main.py", ticker], check=True)
            # subprocess.run(["go", "run", "main.go", "0", ticker, "2024-01-01", "2025-01-01"], check=True)

            messagebox.showinfo("Success", f"Pipeline finished for {ticker}")
            self.refresh_files()
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Pipeline failed:\n{e}")

    def refresh_files(self):
        self.csv_list.delete(0, tk.END)
        self.img_list.delete(0, tk.END)

        if os.path.exists(DATA_DIR):
            for f in os.listdir(DATA_DIR):
                if f.endswith(".csv"):
                    self.csv_list.insert(tk.END, os.path.join(DATA_DIR, f))

        if os.path.exists(IMG_DIR):
            for f in os.listdir(IMG_DIR):
                if f.endswith(".png"):
                    self.img_list.insert(tk.END, os.path.join(IMG_DIR, f))

    def show_csv(self, event):
        if not self.csv_list.curselection():
            return
        filepath = self.csv_list.get(self.csv_list.curselection()[0])

        for col in self.tree.get_children():
            self.tree.delete(col)
        self.tree["columns"] = []

        try:
            df = pd.read_csv(filepath)
            self.tree["columns"] = list(df.columns)
            for col in df.columns:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=100, anchor="center")

            for _, row in df.iterrows():
                self.tree.insert("", "end", values=list(row))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV:\n{e}")

    def show_image(self, event):
        if not self.img_list.curselection():
            return
        filepath = self.img_list.get(self.img_list.curselection()[0])

        try:
            self.original_img = Image.open(filepath)  # keep original
            self.resize_image()  # draw first time
            self.img_frame.bind("<Configure>", lambda e: self.resize_image())
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")

    def resize_image(self):
        if hasattr(self, "original_img"):
            frame_w = self.img_frame.winfo_width()
            frame_h = self.img_frame.winfo_height()
            img = self.original_img.copy().resize((frame_w, frame_h), Image.LANCZOS)
            self.tk_img = ImageTk.PhotoImage(img)
            self.img_label.config(image=self.tk_img)

if __name__ == "__main__":
    app = App()
    app.mainloop()
