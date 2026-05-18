from pathlib import Path
import json
import re
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk


BASE_DIR = Path(__file__).resolve().parents[1]
training_set = True

DATASET_DIR = (
    BASE_DIR / "symbols_dataset"
    if training_set
    else BASE_DIR / "symbols_dataset_testing"
)
LABELS_PATH = DATASET_DIR / "labels.json"

CARD_LABELS = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "skip",
    "reverse",
    "wild",
    "draw_2",
    "draw_4",
]


def symbol_number(path):
    match = re.search(r"symbol_(\d+)\.png$", path.name)
    if match is None:
        return float("inf")
    return int(match.group(1))


def load_labels():
    if not LABELS_PATH.exists():
        return {}

    with LABELS_PATH.open("r") as file:
        return json.load(file)


def save_labels(labels):
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = LABELS_PATH.with_suffix(".tmp")

    with tmp_path.open("w") as file:
        json.dump(labels, file, indent=2, sort_keys=True)

    tmp_path.replace(LABELS_PATH)


class DatasetLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Symbol Dataset Labeling")

        self.symbol_paths = sorted(DATASET_DIR.glob("symbol_*.png"), key=symbol_number)
        self.labels = load_labels()
        self.current_index = self.first_unlabeled_index()
        self.current_photo = None

        self.title_var = tk.StringVar()
        self.progress_var = tk.StringVar()

        tk.Label(root, textvariable=self.title_var, font=("Arial", 16, "bold")).pack(pady=(12, 4))
        tk.Label(root, textvariable=self.progress_var).pack(pady=(0, 8))

        self.image_label = tk.Label(root, bg="white", width=420, height=420)
        self.image_label.pack(padx=16, pady=8)

        button_frame = tk.Frame(root)
        button_frame.pack(padx=16, pady=12)

        for idx, label in enumerate(CARD_LABELS):
            button = tk.Button(
                button_frame,
                text=label,
                width=10,
                command=lambda value=label: self.label_current_symbol(value),
            )
            button.grid(row=idx // 5, column=idx % 5, padx=4, pady=4)

        action_frame = tk.Frame(root)
        action_frame.pack(pady=(0, 16))

        delete_button = tk.Button(
            action_frame,
            text="Delete Image",
            width=18,
            command=self.delete_current_symbol,
        )
        delete_button.grid(row=0, column=0, padx=4)

        stop_button = tk.Button(action_frame, text="Stop", width=18, command=self.stop)
        stop_button.grid(row=0, column=1, padx=4)

        self.bind_number_keys()
        self.bind_action_keys()
        self.show_current_symbol()

    def bind_number_keys(self):
        for label in CARD_LABELS[:10]:
            self.root.bind(label, lambda event, value=label: self.label_current_symbol(value))

    def bind_action_keys(self):
        self.root.bind("<Delete>", lambda event: self.delete_current_symbol())
        self.root.bind("<BackSpace>", lambda event: self.delete_current_symbol())
        self.root.bind("d", lambda event: self.delete_current_symbol())

    def first_unlabeled_index(self):
        for idx, path in enumerate(self.symbol_paths):
            if path.name not in self.labels:
                return idx
        return len(self.symbol_paths)

    def show_current_symbol(self):
        if not self.symbol_paths:
            messagebox.showinfo("No Symbols", f"No symbol_*.png files found in {DATASET_DIR}")
            self.root.destroy()
            return

        if self.current_index >= len(self.symbol_paths):
            self.title_var.set("All symbols labeled")
            self.progress_var.set(f"{len(self.labels)}/{len(self.symbol_paths)} complete")
            self.image_label.config(image="", text="Done")
            messagebox.showinfo("Complete", "All symbols in the dataset are labeled.")
            self.root.destroy()
            return

        path = self.symbol_paths[self.current_index]
        image = Image.open(path).convert("L")
        image = image.resize((360, 360), Image.Resampling.NEAREST)
        self.current_photo = ImageTk.PhotoImage(image)

        self.title_var.set(path.name)
        self.progress_var.set(f"{self.current_index + 1}/{len(self.symbol_paths)}")
        self.image_label.config(image=self.current_photo, text="")

    def label_current_symbol(self, label):
        if self.current_index >= len(self.symbol_paths):
            return

        path = self.symbol_paths[self.current_index]
        self.labels[path.name] = label
        save_labels(self.labels)

        self.current_index += 1
        while self.current_index < len(self.symbol_paths):
            if self.symbol_paths[self.current_index].name not in self.labels:
                break
            self.current_index += 1

        self.show_current_symbol()

    def delete_current_symbol(self):
        if self.current_index >= len(self.symbol_paths):
            return

        path = self.symbol_paths[self.current_index]

        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as error:
            messagebox.showerror("Delete Failed", f"Could not delete {path.name}:\n{error}")
            return

        self.labels.pop(path.name, None)
        save_labels(self.labels)
        del self.symbol_paths[self.current_index]

        self.show_current_symbol()

    def stop(self):
        save_labels(self.labels)
        self.root.destroy()


def main():
    root = tk.Tk()
    DatasetLabeler(root)
    root.mainloop()


if __name__ == "__main__":
    main()
