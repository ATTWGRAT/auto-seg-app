from customtkinter import CTk, CTkFrame, CTkLabel, CTkEntry, CTkButton, CTkFont
from tkinter import filedialog

import os

from src.utils.datasets import load_dataset_with_pydicom

from src.utils.monai_helper import download_model_if_empty

from src.utils.segmentator import segment_and_save

import threading

class App(CTk):
    def __init__(self):
        super().__init__()

        self.path = None
        self.datasets = None
        self.model_path = os.path.join(os.path.dirname(__file__), "../model/")

        download_model_if_empty(self.model_path)

        self.title("Auto-Seg")
        self.geometry("1200x600")

        self.main_frame = CTkFrame(self)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        self.main_frame.grid_rowconfigure((0, 1, 2), weight=1)
        self.main_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.create_widgets()

    def create_widgets(self):
        self.path_label = CTkLabel(self.main_frame, text="Series: None", font=CTkFont(size=20, family="Arial"))
        self.path_label.grid(row=0, column=0, padx=10, pady=10, columnspan=3, sticky="nsew")

        self.seg_num_entry = CTkEntry(self.main_frame, placeholder_text="Max Segments", font=CTkFont(size=20, family="Arial"), justify="center")
        self.seg_num_entry.grid(row=0, column=3, padx=10, pady=10, sticky="nsew")

        self.logger = CTKLogger(self.main_frame)
        self.logger.grid(row=1, column=0, columnspan=4, padx=10, pady=10, sticky="ew")

        self.file_button = CTkButton(self.main_frame, text="Select Series", command=self.select_file, font=CTkFont(size=16, family="Arial"))
        self.file_button.grid(row=2, column=0, padx=10, pady=10, columnspan=2, sticky="nsew")

        self.segment_button = CTkButton(self.main_frame, text="Segment", command=self.segment, state="disabled", font=CTkFont(size=16, family="Arial"))
        self.segment_button.grid(row=2, column=2, padx=10, pady=10, columnspan=2, sticky="nsew")

    def segment(self):
        num_segments = self.seg_num_entry.get()

        try:
            seg_num = int(num_segments) if num_segments else 10
            if seg_num <= 0:
                self.logger.log("Please enter a positive integer for max segments.")
                return
        except ValueError:
            self.logger.log("Invalid input for max segments. Please enter a positive integer.")
            return
        
        threading.Thread(target=segment_and_save, args=(self.path, os.path.join(self.model_path, "model.pt"), self.datasets, self.logger, seg_num)).start()

        self.segment_button.configure(state="disabled")

    def select_file(self):
        self.path = filedialog.askdirectory(mustexist=True)

        if not self.path:
            self.logger.log("No series selected.")
            return
        
        try:
            self.datasets = load_dataset_with_pydicom(self.path)
        except Exception as e:
            self.logger.log(f"Error loading dataset: {e}")
            self.path = None
            self.datasets = None
            self.path_label.configure(text="Series: None")
            self.segment_button.configure(state="disabled")
            return

        if not self.datasets:
            self.logger.log("No DICOM files found in the selected directory.")
            self.path = None
            self.datasets = None
            self.path_label.configure(text="Series: None")
            self.segment_button.configure(state="disabled")
            return
        
        self.path_label.configure(text=f"Series: {self.path}")
        self.segment_button.configure(state="normal")
        self.logger.log("Series loaded successfully.")
        

class CTKLogger(CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.logger = CTkLabel(self, width=400, height=200, text="Select dicom series to start", font=CTkFont(size=24, family="Arial"))
        self.logger.pack(pady=10, padx=10)

    def log(self, message):
        self.logger.configure(text=message)