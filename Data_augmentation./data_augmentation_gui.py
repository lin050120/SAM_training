import json
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import (
    BooleanVar,
    Button,
    Checkbutton,
    END,
    Entry,
    Frame,
    Label,
    OptionMenu,
    StringVar,
    Text,
    Tk,
    filedialog,
    messagebox,
)


SEGMENTATION_FORMATS = {
    "Auto (match source)": "auto",
    "Polygon (point lists)": "polygon",
    "RLE (mask)": "rle",
}


APP_DIR = Path(__file__).resolve().parent
AUGMENTATION_SCRIPT = APP_DIR / "scripts" / "random_mixup_aug.py"
SAMPLE_NAME = "peteck202401-202412_133_20260503"


class DataAugmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("COCO Data Augmentation")
        self.root.geometry("940x720")
        self.root.minsize(820, 640)

        self.input_images = StringVar(value=str(APP_DIR / "inputs" / SAMPLE_NAME))
        self.input_json = StringVar(
            value=str(APP_DIR / "inputs" / f"{SAMPLE_NAME}_mod.json")
        )
        self.output_dir = StringVar(value=str(APP_DIR / "outputs" / "augmented_train"))
        self.augmentations_per_image = StringVar(value="5")
        self.mixup_ratio = StringVar(value="0")
        self.seed = StringVar(value="42")
        self.segmentation_format_label = StringVar(
            value=next(iter(SEGMENTATION_FORMATS))
        )
        self.workers = StringVar(value="0")
        self.include_originals = BooleanVar(value=True)
        self.overwrite = BooleanVar(value=False)
        self.weight_color_jitter = StringVar(value="0.25")
        self.weight_rotate = StringVar(value="0.20")
        self.weight_scale = StringVar(value="0.18")
        self.weight_crop = StringVar(value="0.18")
        self.weight_translate = StringVar(value="0.16")
        self.weight_flip = StringVar(value="0.15")
        self.weight_noise = StringVar(value="0.14")
        self.weight_cutout = StringVar(value="0.08")
        self.running = BooleanVar(value=False)

        self.build_ui()

    def build_ui(self):
        self.add_path_row(
            0, "Input image folder", self.input_images, self.choose_input_images
        )
        self.add_path_row(
            1, "COCO annotation JSON", self.input_json, self.choose_input_json
        )
        self.add_path_row(
            2, "Output dataset folder", self.output_dir, self.choose_output_dir
        )

        Label(self.root, text="Augmentations per source image").grid(
            row=3, column=0, padx=12, pady=6, sticky="w"
        )
        Entry(self.root, textvariable=self.augmentations_per_image, width=12).grid(
            row=3, column=1, padx=8, pady=6, sticky="w"
        )

        Label(self.root, text="MixUp ratio").grid(
            row=4, column=0, padx=12, pady=6, sticky="w"
        )
        Entry(self.root, textvariable=self.mixup_ratio, width=12).grid(
            row=4, column=1, padx=8, pady=6, sticky="w"
        )

        Label(self.root, text="Random seed").grid(
            row=5, column=0, padx=12, pady=6, sticky="w"
        )
        Entry(self.root, textvariable=self.seed, width=12).grid(
            row=5, column=1, padx=8, pady=6, sticky="w"
        )

        Label(self.root, text="Segmentation format").grid(
            row=6, column=0, padx=12, pady=6, sticky="w"
        )
        OptionMenu(
            self.root, self.segmentation_format_label, *SEGMENTATION_FORMATS
        ).grid(row=6, column=1, padx=8, pady=6, sticky="w")

        Label(self.root, text="Parallel workers (0 = all CPU cores)").grid(
            row=7, column=0, padx=12, pady=6, sticky="w"
        )
        Entry(self.root, textvariable=self.workers, width=12).grid(
            row=7, column=1, padx=8, pady=6, sticky="w"
        )

        options_frame = Frame(self.root)
        options_frame.grid(row=8, column=1, columnspan=2, padx=8, pady=4, sticky="w")
        Checkbutton(
            options_frame,
            text="Include original images and annotations",
            variable=self.include_originals,
        ).grid(row=0, column=0, padx=(0, 20), sticky="w")
        Checkbutton(
            options_frame,
            text="Back up and replace a non-empty output folder",
            variable=self.overwrite,
        ).grid(row=0, column=1, sticky="w")

        Label(self.root, text="Method weights").grid(
            row=9, column=0, padx=12, pady=(12, 4), sticky="w"
        )
        weights_frame = Frame(self.root)
        weights_frame.grid(
            row=9, column=1, columnspan=2, padx=8, pady=(12, 4), sticky="w"
        )
        weight_fields = [
            ("Object hue", self.weight_color_jitter),
            ("Rotate", self.weight_rotate),
            ("Scale", self.weight_scale),
            ("Crop/zoom", self.weight_crop),
            ("Translate", self.weight_translate),
            ("Flip", self.weight_flip),
            ("Noise", self.weight_noise),
            ("Cutout", self.weight_cutout),
        ]
        for index, (label, variable) in enumerate(weight_fields):
            row = index // 4
            column = (index % 4) * 2
            Label(weights_frame, text=label).grid(
                row=row, column=column, padx=(0, 4), pady=3, sticky="e"
            )
            Entry(weights_frame, textvariable=variable, width=8).grid(
                row=row, column=column + 1, padx=(0, 16), pady=3, sticky="w"
            )

        self.run_button = Button(
            self.root,
            text="Generate dataset",
            command=self.start_generation,
            width=18,
        )
        self.run_button.grid(row=10, column=2, padx=12, pady=10, sticky="e")

        self.log = Text(self.root, height=18, wrap="word")
        self.log.grid(
            row=11, column=0, columnspan=3, padx=12, pady=(8, 12), sticky="nsew"
        )

        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(11, weight=1)
        self.write_log(
            "Select a train image folder and its COCO JSON. "
            "The result is a complete COCO train dataset.\n"
        )

    def add_path_row(self, row, label, variable, command):
        Label(self.root, text=label).grid(
            row=row, column=0, padx=12, pady=6, sticky="w"
        )
        Entry(self.root, textvariable=variable, width=80).grid(
            row=row, column=1, padx=8, pady=6, sticky="we"
        )
        Button(self.root, text="Browse", command=command, width=14).grid(
            row=row, column=2, padx=12, pady=6, sticky="e"
        )

    def choose_input_images(self):
        folder = filedialog.askdirectory(
            title="Choose the input image folder",
            initialdir=self.input_images.get() or str(APP_DIR),
        )
        if folder:
            self.input_images.set(folder)

    def choose_input_json(self):
        file_path = filedialog.askopenfilename(
            title="Choose the COCO annotation JSON",
            initialdir=str(Path(self.input_json.get() or APP_DIR).expanduser().parent),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if file_path:
            self.input_json.set(file_path)

    def choose_output_dir(self):
        folder = filedialog.askdirectory(
            title="Choose or create the output dataset folder",
            initialdir=str(Path(self.output_dir.get() or APP_DIR).expanduser().parent),
            mustexist=False,
        )
        if folder:
            self.output_dir.set(folder)

    def validate_inputs(self):
        input_images_text = self.input_images.get().strip()
        input_json_text = self.input_json.get().strip()
        output_dir_text = self.output_dir.get().strip()
        if not input_images_text or not input_json_text or not output_dir_text:
            raise ValueError(
                "Input image folder, COCO JSON, and output folder are required."
            )

        input_images = Path(input_images_text).expanduser().resolve()
        input_json = Path(input_json_text).expanduser().resolve()
        output_dir = Path(output_dir_text).expanduser().resolve()
        if not input_images.is_dir():
            raise ValueError(f"Input image folder does not exist: {input_images}")
        if not input_json.is_file():
            raise ValueError(f"COCO JSON does not exist: {input_json}")
        if output_dir.exists() and not output_dir.is_dir():
            raise ValueError(f"Output path is not a folder: {output_dir}")
        if (
            output_dir.exists()
            and any(output_dir.iterdir())
            and not self.overwrite.get()
        ):
            raise ValueError(
                "The output folder is not empty. Choose another folder or enable backup and replace."
            )

        augmentations_per_image = int(self.augmentations_per_image.get().strip())
        if augmentations_per_image <= 0:
            raise ValueError(
                "Augmentations per source image must be a positive integer."
            )
        mixup_ratio = float(self.mixup_ratio.get().strip())
        if not 0.0 <= mixup_ratio <= 1.0:
            raise ValueError("MixUp ratio must be between 0.0 and 1.0.")
        seed = int(self.seed.get().strip())
        segmentation_format = SEGMENTATION_FORMATS.get(
            self.segmentation_format_label.get()
        )
        if segmentation_format is None:
            raise ValueError("Choose a segmentation format from the list.")
        workers = int(self.workers.get().strip())
        if workers < 0:
            raise ValueError("Parallel workers must be 0 (all cores) or positive.")

        weights = {
            "color_jitter": float(self.weight_color_jitter.get().strip()),
            "rotate": float(self.weight_rotate.get().strip()),
            "scale": float(self.weight_scale.get().strip()),
            "crop": float(self.weight_crop.get().strip()),
            "translate": float(self.weight_translate.get().strip()),
            "flip": float(self.weight_flip.get().strip()),
            "noise": float(self.weight_noise.get().strip()),
            "cutout": float(self.weight_cutout.get().strip()),
        }
        if any(value < 0 for value in weights.values()):
            raise ValueError("Method weights must be zero or positive.")
        if sum(1 for value in weights.values() if value > 0) < 3:
            raise ValueError("At least three method weights must be greater than zero.")

        with input_json.open("r", encoding="utf-8") as file:
            data = json.load(file)
        counts = {
            "images": len(data.get("images", [])),
            "annotations": len(data.get("annotations", [])),
            "categories": len(data.get("categories", [])),
        }
        if any(count == 0 for count in counts.values()):
            raise ValueError(
                "COCO JSON must contain non-empty images, annotations, and categories lists."
            )

        return {
            "input_images": input_images,
            "input_json": input_json,
            "output_dir": output_dir,
            "augmentations_per_image": augmentations_per_image,
            "mixup_ratio": mixup_ratio,
            "seed": seed,
            "segmentation_format": segmentation_format,
            "workers": workers,
            "include_originals": bool(self.include_originals.get()),
            "overwrite": bool(self.overwrite.get()),
            "weights": weights,
            "counts": counts,
        }

    def start_generation(self):
        if self.running.get():
            return
        try:
            settings = self.validate_inputs()
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        self.running.set(True)
        self.run_button.config(state="disabled")
        self.write_log(
            "\nStarting generation: "
            f"{settings['counts']['images']} source images, "
            f"{settings['counts']['annotations']} annotations.\n"
        )
        threading.Thread(
            target=self.run_generation,
            args=(settings,),
            daemon=True,
        ).start()

    def run_generation(self, settings):
        weights = settings["weights"]
        cmd = [
            sys.executable,
            str(AUGMENTATION_SCRIPT),
            "--input-images",
            str(settings["input_images"]),
            "--input-json",
            str(settings["input_json"]),
            "--output-dir",
            str(settings["output_dir"]),
            "--augmentations-per-image",
            str(settings["augmentations_per_image"]),
            "--mixup-ratio",
            str(settings["mixup_ratio"]),
            "--seed",
            str(settings["seed"]),
            "--segmentation-format",
            settings["segmentation_format"],
            "--workers",
            str(settings["workers"]),
            "--weight-color-jitter",
            str(weights["color_jitter"]),
            "--weight-rotate",
            str(weights["rotate"]),
            "--weight-scale",
            str(weights["scale"]),
            "--weight-crop",
            str(weights["crop"]),
            "--weight-translate",
            str(weights["translate"]),
            "--weight-flip",
            str(weights["flip"]),
            "--weight-noise",
            str(weights["noise"]),
            "--weight-cutout",
            str(weights["cutout"]),
        ]
        if not settings["include_originals"]:
            cmd.append("--no-include-originals")
        if settings["overwrite"]:
            cmd.append("--overwrite")

        self.write_log("Command:\n" + subprocess.list2cmdline(cmd) + "\n\n")
        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(APP_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None
            for line in process.stdout:
                self.write_log(line)
            return_code = process.wait()
        except Exception as exc:
            self.write_log(f"\nERROR: {exc}\n")
            self.root.after(0, self.finish_generation, False)
            return

        self.root.after(0, self.finish_generation, return_code == 0)

    def finish_generation(self, success):
        self.running.set(False)
        self.run_button.config(state="normal")
        if success:
            self.write_log("\nFinished successfully.\n")
            messagebox.showinfo("Done", "COCO data augmentation finished successfully.")
        else:
            self.write_log("\nGeneration failed. Check the log above.\n")
            messagebox.showerror("Failed", "Data augmentation failed. Check the log.")

    def write_log(self, content):
        def append():
            self.log.insert(END, content)
            self.log.see(END)

        self.root.after(0, append)


def main():
    root = Tk()
    DataAugmentationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
