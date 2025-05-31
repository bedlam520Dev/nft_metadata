#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0301
# pylint: disable=C0302
# pylint: disable=W0718
"""
NFT Trait Classifier Trainer -
Part 1 (GUI Application) - Enhanced
This script provides a GUI application to read the first 1294
NFT images and their corresponding metadata to train visual
classifiers for identifying trait values.
Enhancements:
- Data Augmentation: Increases training data variety.
- Hyperparameter Tuning: Uses GridSearchCV to
find better model parameters.
- Feedback Loop: Incorporates corrections
from a 'trait_corrections.json' file.
It incorporates improved feature extraction
(color histograms for backgrounds) and GUI-based
human verification and correction. Models are saved
using gzip-compressed pickle format (.pkl.gz).
Image URLs are accessed via an images.json file
(handling various formats), and metadata JSON files
are read from the "metadata" folder.
"""
import gzip
import json
import logging
import os
import pickle
import random
import sys
import tkinter as tk
from collections import Counter
from http.cookiejar import LoadError
from tkinter import messagebox, scrolledtext, ttk
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageTk, UnidentifiedImageError
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = "."
LOG_DIR = os.path.join(BASE_DIR, "logs")
CONFIG_DIR = os.path.join(BASE_DIR, "config")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
CLASSIFIER_DIR = os.path.join(CONFIG_DIR, "classifiers")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
IMAGE_JSON_PATH = os.path.join(CONFIG_DIR, "images.json")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
LOG_FILE_PATH = os.path.join(LOG_DIR, "trait_training.log")
CORRECTIONS_FILE_PATH = os.path.join(CONFIG_DIR, "trait_corrections.json")
DATA_COLLECTION_CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "part1_data_collection_checkpoint.json")
CLASSIFIER_TRAINING_CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "part1_classifier_training_checkpoint.json")
COLLECTED_DATA_FILE = os.path.join(CHECKPOINT_DIR, "part1_collected_training_data.pkl.gz")
SCHEMA_FILE_PATH = os.path.join(CONFIG_DIR, "schema.json")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(CLASSIFIER_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

NUM_TOKENS_TO_TRAIN = 1294
TOKEN_ID_START = 1
IMAGE_SIZE = (128, 128)
DISPLAY_IMAGE_SIZE = (400, 400)
HISTOGRAM_BINS = 16
VERIFICATION_ENABLED = True
VERIFICATION_FREQUENCY = 200
VERIFICATION_SAMPLE_SIZE = 1
ENABLE_DATA_AUGMENTATION = True
ENABLE_HYPERPARAMETER_TUNING = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode= 'w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

def augment_image(image: Image.Image) -> List[Image.Image]:
    """
    Applies random augmentations to an image.
    Args:
        image: The input PIL Image object (RGB).
    Returns:
        A list of PIL Image objects, including
        the original and augmented versions.
    """
    if not ENABLE_DATA_AUGMENTATION:
        try:
            if image.size != IMAGE_SIZE:
                return [image.copy().resize(IMAGE_SIZE, Image.Resampling.LANCZOS)]
            return [image.copy()]
        except (IOError, ValueError, AttributeError, TypeError, RuntimeError) as e:
            logger.error("Failed to process original image when augmentation is disabled: %s", e)
            return []

    all_generated_imgs = [image.copy()]

    try:
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            fill_color = image.getpixel((0, 0)) if image.width > 0 and image.height > 0 else (0, 0, 0)
            rotated = image.rotate(angle, expand=False, fillcolor=fill_color)
            all_generated_imgs.append(rotated)
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            brightness_enhancer = ImageEnhance.Brightness(image)
            brightness = brightness_enhancer.enhance(factor)
            all_generated_imgs.append(brightness)
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            contrast_enhancer = ImageEnhance.Contrast(image)
            contrast = contrast_enhancer.enhance(factor)
            all_generated_imgs.append(contrast)
        if random.random() > 0.5:
            mirrored = ImageOps.mirror(image)
            all_generated_imgs.append(mirrored)
        if random.random() > 0.3:
            scale = random.uniform(0.9, 1.0)
            w, h = image.size
            new_w, new_h = int(w * scale), int(h * scale)
            if new_w > 0 and new_h > 0 and w > 0 and h > 0:
                left = (w - new_w) // 2
                top = (h - new_h) // 2
                right = left + new_w
                bottom = top + new_h
                cropped = image.crop((left, top, right, bottom))
                resized_zoom = cropped.resize(image.size, Image.Resampling.LANCZOS)
                all_generated_imgs.append(resized_zoom)
    except (FileNotFoundError, IOError, RuntimeError, TypeError, ValueError) as e:
        logger.warning("Augmentation failed for an image: %s", e)

    final_processed_images = []
    seen_hashes = set()

    for aug_img in all_generated_imgs:
        try:
            if aug_img.size != IMAGE_SIZE:
                processed_img = aug_img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
            else:
                processed_img = aug_img

            img_hash = hash(processed_img.tobytes())
            if img_hash not in seen_hashes:
                final_processed_images.append(processed_img)
                seen_hashes.add(img_hash)
        except (IOError, RuntimeError, TypeError, AttributeError, KeyError, ValueError) as e:
            logger.warning("Could not resize or hash image during augmentation post-processing: %s. Skipping this instance.", e)

    if not final_processed_images and image:
        try:
            original_resized = image.copy().resize(IMAGE_SIZE, Image.Resampling.LANCZOS) if image.size != IMAGE_SIZE else image.copy()
            final_processed_images.append(original_resized)
            logger.debug("Augmentation resulted in empty list; returning only original (resized).")
        except (IOError, ValueError, AttributeError, TypeError, RuntimeError) as e_orig:
            logger.error("Failed to prepare even the original image in augmentation fallback: %s", e_orig)
            return []

    logger.debug("Generated %d unique augmented images (incl. original, all resized to %s).", len(final_processed_images), IMAGE_SIZE)
    return final_processed_images

def load_image(token_id: int, image_map: Dict[str, str]) -> Optional[Tuple[Image.Image, Image.Image]]:
    """
    Loads an image from the local IMAGES_DIR based on token_id and image_map.
    The image_map maps token_id (string) to filename.
    Args:
        token_id: The integer ID of the token.
        image_map: A dictionary mapping string token IDs to image filenames.
    Returns:
        A tuple (image_for_display, image_for_features_at_IMAGE_SIZE),
        or None if loading fails. image_for_display is sized for GUI,
        image_for_features is sized to IMAGE_SIZE for model input.
    """
    filename = image_map.get(str(token_id))
    if not filename:
        logger.error("Filename not found in image_map for token ID %d.", token_id)
        return None

    local_image_path = os.path.join(IMAGES_DIR, filename)
    logger.debug("Attempting to load image for token %d from local path: %s", token_id, local_image_path)

    try:
        if not os.path.exists(local_image_path):
            logger.error("Local image file not found for token %d: %s", token_id, local_image_path)
            return None

        original_full_image = Image.open(local_image_path).convert("RGB")
        image_for_display = original_full_image.copy()
        image_for_display.thumbnail(DISPLAY_IMAGE_SIZE, Image.Resampling.LANCZOS)
        image_for_features = original_full_image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        logger.debug("Successfully loaded and processed image for token %d from %s", token_id, local_image_path)
        return image_for_display, image_for_features

    except FileNotFoundError:
        logger.error("Local image file not found (FileNotFoundError) for token %d: %s", token_id, local_image_path)
        return None
    except UnidentifiedImageError:
        logger.error("Failed to load image for token %d from %s: Cannot identify image file", token_id, local_image_path)
        return None
    except LoadError as e:
        logger.error("Cookie/Load error processing image from URL %s: %s", local_image_path, e, exc_info=True)
        return None
    except (OSError, IOError, ValueError) as e:
        logger.error("Failed processing image for token %d from %s: %s", token_id, local_image_path, str(e))
        return None
    except RuntimeError as e:
        logger.error("Unexpected error loading image for token %d from %s: %s", token_id, local_image_path, e, exc_info=True)
        return None

def load_metadata(token_id: int) -> Optional[Dict]:
    """
    Loads metadata from
    the metadata directory
    for a given token ID.
    """
    metadata_path = os.path.join(METADATA_DIR, str(token_id))
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.info("Metadata file not found for token ID %d: %s (Expected for training)", token_id, metadata_path)
        return None
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in metadata file %s: %s", metadata_path, e)
        return None
    except OSError as e:
        logger.error("OS error reading metadata file %s: %s", metadata_path, e)
        return None
    except RuntimeError as e:
        logger.error("Unexpected error loading metadata for token ID %d: %s", token_id, e, exc_info=True)
    return None


def extract_traits(metadata: Optional[Dict]) -> Dict[str, str]:
    """
    Extracts a dictionary
    of trait_type: value
    from the metadata attributes.
    """
    if not metadata:
        return {}
    attributes = metadata.get("attributes", [])
    if not isinstance(attributes, list):
        logger.warning("Metadata attributes format is not a list for token %s. Cannot extract traits.", metadata.get('name', 'Unknown'))
        return {}
    traits = {}
    for attr in attributes:
        if isinstance(attr, dict) and "trait_type" in attr and "value" in attr:
            traits[str(attr["trait_type"]).strip()] = str(attr["value"]).strip()
        else:
            logger.debug("Skipping invalid attribute item in token %s: %s", metadata.get('name', 'Unknown'), attr)
    return traits

def image_to_vector(image: Image.Image) -> np.ndarray:
    """
    Converts an image
    into a flattened numpy
    array vector.
    """
    try:
        if image.size != IMAGE_SIZE:
            logger.warning("Resizing image in image_to_vector (expected %s, got %s)", IMAGE_SIZE, image.size)
            image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32).flatten()
    except (FileNotFoundError, IOError, RuntimeError) as e:
        logger.error("Failed to convert image to vector: %s", e, exc_info=True)
        return np.array([])

def image_to_color_histogram(image: Image.Image, bins: int = HISTOGRAM_BINS) -> np.ndarray:
    """
    Calculates a flattened,
    normalized 3D color histogram
    for an RGB image.
    """
    try:
        if image.size != IMAGE_SIZE:
            logger.warning("Resizing image in image_to_color_histogram (expected %s, got %s)", IMAGE_SIZE, image.size)
            image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        img_array = np.asarray(image)
        hist_r = np.histogram(img_array[:, :, 0], bins=bins, range=(0, 256))[0]
        hist_g = np.histogram(img_array[:, :, 1], bins=bins, range=(0, 256))[0]
        hist_b = np.histogram(img_array[:, :, 2], bins=bins, range=(0, 256))[0]
        total_pixels = image.width * image.height
        if total_pixels == 0:
            logger.error("Cannot calculate histogram for zero-pixel image.")
            return np.array([])
        norm_factor = total_pixels if total_pixels > 0 else 1
        hist_r = hist_r / norm_factor
        hist_g = hist_g / norm_factor
        hist_b = hist_b / norm_factor
        return np.concatenate((hist_r, hist_g, hist_b))
    except (RuntimeError, TypeError, AttributeError, KeyError, ValueError) as e:
        logger.error("Failed to calculate color histogram: %s", e, exc_info=True)
        return np.array([])

def get_feature_extractor_and_data(
    trait_type: str,
    image_vector: np.ndarray,
    image_histogram: np.ndarray
) -> Optional[np.ndarray]:
    """
    Selects the appropriate
    feature vector based
    on trait type.
    """
    if trait_type == "Background":
        if image_histogram.size > 0:
            return image_histogram
        else:
            logger.warning("Requested histogram features for '%s', but they are empty.", trait_type)
            return None
    else:
        if image_vector.size > 0:
            return image_vector
        else:
            logger.warning("Requested vector features for '%s', but they are empty.", trait_type)
            return None

class VerificationWindow(tk.Toplevel):
    """
    GUI window for verifying
    and correcting token
    traits (Modal Dialog).
    """ # Fix: Dedent the code block
    def __init__(self, master: tk.Tk, token_id: int, image: Image.Image, initial_traits: Dict[str, str], trait_schema_order: List[str]):
        super().__init__(master)
        self.title(f"Verify & Correct Token {token_id}")
        self.grab_set()
        self.transient(master)
        self.token_id = token_id
        self.trait_schema_order = trait_schema_order if trait_schema_order else sorted(initial_traits.keys())
        self.image = image
        self.initial_traits = initial_traits
        self.corrected_traits: Optional[Dict[str, str]] = None
        self.trait_widgets: Dict[str, Dict[str, Any]] = {}
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        self.style.configure("Dialog.TFrame", background="#535353")
        self.style.configure("Dialog.TLabel", padding=6, font=('Segoe UI', 10), background="#575757", foreground="#ECECEC")
        self.style.configure("Verify.TButton", foreground="#ECECEC", background="#388E3C", font=('Segoe UI', 10, 'bold'))
        self.style.map("Verify.TButton", background=[('active', '#2E7D32')])
        self.style.configure("Reject.TButton", foreground="#ECECEC", background="#D32F2F", font=('Segoe UI', 10, 'bold'))
        self.style.map("Reject.TButton", background=[('active', '#C62828')])

        max_win_width = int(master.winfo_screenwidth() * 0.8)
        max_win_height = int(master.winfo_screenheight() * 0.8)
        self.geometry("850x750")
        self.maxsize(max_win_width, max_win_height)
        self.minsize(600, 500)

        main_frame = ttk.Frame(self, padding="10", style="Dialog.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.rowconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=1)

        img_frame = ttk.Frame(main_frame, style="Dialog.TFrame")
        img_frame.grid(row=0, column=0, pady=10, sticky="n")
        try:
            display_img = image.copy()
            display_img.thumbnail(DISPLAY_IMAGE_SIZE, Image.Resampling.LANCZOS)
            self.img_tk = ImageTk.PhotoImage(display_img)
            img_label = tk.Label(img_frame, image=self.img_tk, relief="groove", borderwidth=1)
            img_label.pack()
        except LoadError as e:
            img_label = tk.Label(img_frame, text=f"Error displaying image:\n{e}", fg="#C73C3C", bg="#B4B4B4", relief="groove", borderwidth=1, width=int(DISPLAY_IMAGE_SIZE[0]/8), height=int(DISPLAY_IMAGE_SIZE[1]/15))
            img_label.pack(padx=10, pady=10)
            logger.error("Tkinter (Dialog): Failed to display image for token %d: %s", token_id, e)

        traits_outer_frame = ttk.Frame(main_frame, style="Dialog.TFrame")
        traits_outer_frame.grid(row=1, column=0, sticky="nsew", pady=10)
        traits_outer_frame.rowconfigure(0, weight=1)
        traits_outer_frame.columnconfigure(0, weight=1)

        traits_canvas = tk.Canvas(traits_outer_frame, borderwidth=0, background="#3B3B3B")
        scrollbar = ttk.Scrollbar(traits_outer_frame, orient="vertical", command=traits_canvas.yview)
        scrollable_frame = ttk.Frame(traits_canvas, style="Dialog.TFrame")

        scrollable_frame.bind("<Configure>", lambda e: traits_canvas.configure(scrollregion=traits_canvas.bbox("all")))
        canvas_window = traits_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        traits_canvas.configure(yscrollcommand=scrollbar.set)
        traits_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        traits_canvas.bind('<Configure>', lambda e: traits_canvas.itemconfig(canvas_window, width=e.width))

        header_label = ttk.Label(scrollable_frame, text="Verify / Correct Traits:", font=('Segoe UI', 12, 'bold'), style="Dialog.TLabel")
        header_label.grid(row=0, column=0, columnspan=2, pady=(5, 10), padx=10, sticky='w')

        row_num = 1
        displayed_traits_count = 0
        for trait_type_from_schema in self.trait_schema_order:
            # Use initial_traits.get to handle cases where a schema trait might not be in the current token's metadata
            value = initial_traits.get(trait_type_from_schema, "")

            ttk.Label(scrollable_frame, text=f"{trait_type_from_schema}:", anchor="e", style="Dialog.TLabel").grid(row=row_num, column=0, padx=(10, 5), pady=3, sticky='ew')
            correction_entry = ttk.Entry(scrollable_frame, width=40, font=('Segoe UI', 10))
            correction_entry.insert(0, value)
            correction_entry.grid(row=row_num, column=1, padx=5, pady=3, sticky='ew')
            self.trait_widgets[trait_type_from_schema] = {"entry": correction_entry}
            row_num += 1
            displayed_traits_count +=1
        if displayed_traits_count == 0 and initial_traits: # Fallback if schema was empty but traits exist
            logger.warning("Trait schema order was empty or resulted in no traits displayed for token %d. Falling back to sorted initial traits.", token_id)
            for trait_type, value in sorted(initial_traits.items()):
                ttk.Label(scrollable_frame, text=f"{trait_type}:", anchor="e", style="Dialog.TLabel").grid(row=row_num, column=0, padx=(10,5), pady=3, sticky='ew')
                correction_entry = ttk.Entry(scrollable_frame, width=40, font=('Segoe UI', 10))
                correction_entry.insert(0, value)
                correction_entry.grid(row=row_num, column=1, padx=5, pady=3, sticky='ew')
                self.trait_widgets[trait_type] = {"entry": correction_entry}
                row_num += 1

        scrollable_frame.columnconfigure(0, weight=1, pad=5)
        scrollable_frame.columnconfigure(1, weight=3, pad=5)

        button_frame = ttk.Frame(main_frame, style="Dialog.TFrame")
        button_frame.grid(row=2, column=0, pady=(10, 5), sticky="ew")
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=0)
        button_frame.columnconfigure(2, weight=0)
        button_frame.columnconfigure(3, weight=1)

        submit_button = ttk.Button(button_frame, text="Submit Corrections", command=self._submit, style="Verify.TButton", width=18)
        submit_button.grid(row=0, column=1, padx=10, pady=5, ipady=3)
        discard_button = ttk.Button(button_frame, text="Discard Token", command=self._discard, style="Reject.TButton", width=18)
        discard_button.grid(row=0, column=2,  padx=10, pady=5, ipady=3)

        self.update_idletasks()
        master_x = master.winfo_x()
        master_y = master.winfo_y()
        master_width = master.winfo_width()
        master_height = master.winfo_height()
        win_width = self.winfo_width()
        win_height = self.winfo_height()

        x_coord = master_x + int((master_width / 2) - (win_width / 2))
        y_coord = master_y + int((master_height / 2) - (win_height / 2))
        self.geometry(f'+{x_coord}+{y_coord}')

        first_entry = next((widgets["entry"] for widgets in self.trait_widgets.values()), None)
        if first_entry:
            first_entry.focus_set()

        self.bind('<Return>', lambda event=None: self._submit())
        self.bind('<Escape>', lambda event=None: self._discard())

    def _submit(self):
        """
        Collect corrected traits
        and close the window.
        """
        self.corrected_traits = {}
        found_change = False
        try:
            for trait_type, widgets in self.trait_widgets.items():
                corrected_value = widgets["entry"].get().strip()
                if corrected_value:
                    self.corrected_traits[trait_type] = corrected_value
                    if corrected_value != self.initial_traits.get(trait_type, ""):
                        found_change = True
                else:
                    if trait_type in self.initial_traits:
                        logger.warning("User submitted empty value for existing trait '%s' on token %d. Excluding trait.", trait_type, self.token_id)
                        found_change = True

            if not self.corrected_traits:
                if messagebox.askyesno("Confirm Empty Traits", "You haven't provided any trait values (or cleared them all).\nDo you want to DISCARD this token instead?", parent=self, icon='warning'):
                    self.corrected_traits = None
                    logger.warning("User confirmed discarding token %d due to empty traits submission.", self.token_id)
                else:
                    return

            if self.corrected_traits is not None:
                if found_change:
                    logger.info("User submitted corrections for token %d.", self.token_id)
                else:
                    logger.info("User confirmed original traits for token %d (no changes made).", self.token_id)

            self.destroy()

        except (tk.TclError, ValueError, AttributeError) as e:
            logger.error("Error submitting corrections for token %d: %s", self.token_id, e, exc_info=True)
            messagebox.showerror("Error", f"An error occurred while submitting:\n{e}", parent=self)
            if messagebox.askyesno("Error Handling", "An error occurred submitting. Discard this token?", parent=self):
                self.corrected_traits = None
                self.destroy()

    def _discard(self):
        """
        Signal that the token should
        be discarded after confirmation.
        """
        if messagebox.askyesno("Confirm Discard", "Are you sure you want to discard all data for this token?\nIt will NOT be used for training.", parent=self, icon='question'):
            logger.warning("User chose to discard token %d via GUI button.", self.token_id)
            self.corrected_traits = None
            self.destroy()

    def wait_for_input(self) -> Optional[Dict[str, str]]:
        """
        Blocks execution until
        the window is closed
        and returns the result.
        """
        self.wait_window(self)
        return self.corrected_traits

class TraitTrainerApp:
    """
    Main GUI application
    for training trait
    classifiers.
    """
    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("NFT Trait Classifier Trainer (Enhanced GUI)")
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        win_width = max(800, int(screen_width * 0.7))
        win_height = max(600, int(screen_height * 0.7))
        x_coord = max(0, int((screen_width / 2) - (win_width / 2)))
        y_coord = max(0, int((screen_height / 2) - (win_height / 2)))
        self.master.geometry(f"{win_width}x{win_height}+{x_coord}+{y_coord}")
        self.master.minsize(700, 500)
        self.style = ttk.Style(self.master)
        self.style.theme_use('clam')
        self.style.configure("TFrame", background="#3B3B3B")
        self.style.configure("TLabel", padding=6, font=('Segoe UI', 10), background="#535353", foreground="#ECECEC")
        self.style.configure("Header.TLabel", font=('Segoe UI', 14, 'bold'), foreground="#ECECEC", background="#1E3A5F")
        self.style.configure("Status.TLabel", font=('Segoe UI', 11, 'italic'), background="#535353", foreground="#ECECEC")
        self.style.configure("TButton", padding=(10, 5), font=('Segoe UI', 10, 'bold'), background="#7D7D7D", relief="raised")
        self.style.map("TButton", background=[('active', "#86B58D")])
        self.style.configure("Start.TButton", foreground="white", background="#388E3C", font=('Segoe UI', 11, 'bold'))
        self.style.map("Start.TButton", background=[('active', '#2E7D32')])
        self.style.configure("Stop.TButton", foreground="white", background="#D32F2F", font=('Segoe UI', 11, 'bold'))
        self.style.map("Stop.TButton", background=[('active', '#C62828')])
        self.style.configure("TProgressbar", thickness=25)
        self.image_map: Optional[Dict[str, str]] = None
        self.training_data: Dict[str, List[Tuple[np.ndarray, str]]] = {}
        self.classifiers: Dict[str, Pipeline] = {}
        self.token_ids_to_process: List[int] = []
        self.verification_indices: set = set()
        self.verification_possible: bool = False
        self.current_token_index: int = -1
        self.all_trait_types: List[str] = []
        self.current_classifier_index: int = -1
        self.is_running: bool = False
        self.processed_tokens_count: int = 0
        self.augmented_data_count: int = 0
        self.verified_correct_count: int = 0
        self.corrected_count: int = 0
        self.discarded_count: int = 0
        self.skipped_verification_count: int = 0
        self.trained_classifier_count: int = 0
        self.failed_classifier_count: int = 0
        self.busy_indicator_pb: Optional[ttk.Progressbar] = None
        self.gui_log_handler: Optional[GuiLogger] = None

        self.trait_schema_order: List[str] = []
        self.corrections_data: Dict[str, Dict[str, str]] = {}
        self.used_corrections_count: int = 0

        self._setup_gui()
        self._load_image_map()
        self._load_trait_schema()
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        self._initialize_checkpoint_methods()
        self._load_corrections()

    def _load_trait_schema(self):
        """
        Loads the trait order schema from schema.json.
        """
        self.update_status("Loading trait schema...", show_busy_indicator=True)
        try:
            with open(SCHEMA_FILE_PATH, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            if not isinstance(schema, list) or not all(isinstance(item, str) for item in schema):
                logger.error("Schema file %s is not a valid list of strings. Verification dialog may not follow strict order.", SCHEMA_FILE_PATH)
                self.trait_schema_order = []
            else:
                self.trait_schema_order = schema
                logger.info("Successfully loaded trait schema with %d traits from %s.", len(self.trait_schema_order), SCHEMA_FILE_PATH)
        except FileNotFoundError:
            logger.warning("Trait schema file not found: %s. Verification dialog may not follow strict order.", SCHEMA_FILE_PATH)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Error reading or parsing trait schema file %s: %s. Verification dialog may not follow strict order.", SCHEMA_FILE_PATH, e)
        self.update_status("Status: Idle" if not self.image_map else f"Status: Ready. Image map loaded ({len(self.image_map)} entries).", show_busy_indicator=False)

    def _setup_gui(self):
        """
        Creates and arranges
        the main GUI widgets.
        """
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.rowconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=0)
        main_frame.columnconfigure(0, weight=1)
        control_frame = ttk.Frame(main_frame, padding=(0, 0, 0, 10))
        control_frame.grid(row=0, column=0, sticky="ew")
        self.start_button = ttk.Button(control_frame, text="Start Training Process", command=self._start_process_confirmed, style="Start.TButton", width=25)
        self.start_button.pack(side=tk.LEFT, padx=10)
        self.stop_button = ttk.Button(control_frame, text="Stop Process", command=self._stop_process, style="Stop.TButton", width=25)
        self.stop_button.pack(side=tk.LEFT, padx=10)
        self.stop_button.config(state=tk.DISABLED)

        log_frame = ttk.Frame(main_frame)
        log_frame.grid(row=1, column=0, sticky="nsew", pady=5)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=15, font=('Consolas', 9), state=tk.DISABLED, bg="#535353", relief="sunken", borderwidth=1)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.progress_label = ttk.Label(main_frame, text="Progress:", style="TLabel")
        self.progress_label.grid(row=2, column=0, sticky="w", padx=5, pady=(5, 0))
        self.progress_bar = ttk.Progressbar(main_frame, orient='horizontal', mode='determinate', length=400)
        self.progress_bar.grid(row=3, column=0, sticky="ew", padx=5, pady=(0, 5))
        self.status_label = ttk.Label(main_frame, text="Status: Idle. Load image map first.", style="Status.TLabel", anchor="w")
        self.status_label.grid(row=4, column=0, sticky="ew", padx=5, pady=(5, 0))
        self.busy_indicator_pb = ttk.Progressbar(main_frame, orient='horizontal', mode='indeterminate', length=100)
        self.busy_indicator_pb.grid(row=5, column=0, sticky="ew", padx=5, pady=(2, 5))
        self.busy_indicator_pb.grid_remove()
        self.gui_log_handler = GuiLogger(self.log_text)
        logging.getLogger().addHandler(self.gui_log_handler)

    def _load_image_map(self):
        """
        Loads the image
        map and updates
        GUI state.
        """
        self.update_status("Loading image map...")
        self.image_map = None
        try:
            logger.info("Loading image map from %s", IMAGE_JSON_PATH)
            with open(IMAGE_JSON_PATH, "r", encoding="utf-8") as f:
                loaded_map = json.load(f)

            if not isinstance(loaded_map, dict):
                raise TypeError("Image map is not a dictionary.")
            if not loaded_map:
                raise ValueError("Image map is empty.")

            self.image_map = loaded_map
            logger.info("Image map loaded successfully (%d entries).", len(self.image_map))
            self.update_status(f"Status: Ready. Image map loaded ({len(self.image_map)} entries).")
            self.start_button.config(state=tk.NORMAL)

        except FileNotFoundError:
            errmsg = f"CRITICAL: Image map file not found: {IMAGE_JSON_PATH}. Cannot proceed."
            logger.critical(errmsg)
            self.update_status("Status: ERROR - Image map not found. Check logs.")
            messagebox.showerror("Error", errmsg)
            self.start_button.config(state=tk.DISABLED)
        except (TypeError, ValueError) as e:
            errmsg = f"CRITICAL: Invalid or empty image map file {IMAGE_JSON_PATH}: {e}. Cannot proceed."
            logger.critical(errmsg)
            self.update_status("Status: ERROR - Invalid image map. Check logs.")
            messagebox.showerror("Error", errmsg)
            self.start_button.config(state=tk.DISABLED)
        except OSError as e:
            errmsg = f"CRITICAL: OS error reading image map file {IMAGE_JSON_PATH}: {e}. Cannot proceed."
            logger.critical(errmsg)
            self.update_status("Status: ERROR - Cannot read image map. Check logs.")
            messagebox.showerror("Error", errmsg)
            self.start_button.config(state=tk.DISABLED)
        except (RuntimeError) as e:
            errmsg = f"CRITICAL: Unexpected error loading image map {IMAGE_JSON_PATH}: {e}. Cannot proceed."
            logger.critical(errmsg, exc_info=True)
            self.update_status("Status: ERROR - Unexpected error loading image map. Check logs.")
            messagebox.showerror("Error", errmsg)
            self.start_button.config(state=tk.DISABLED)

    def _initialize_checkpoint_methods(self):
        """
        Placeholder or for future use if
        complex init for checkpoints is needed.
        """

    def _get_checkpoint_data_collection(self) -> int:
        """
        Loads the last processed
        token ID for data collection.
        """
        default_start_token_id = TOKEN_ID_START - 1
        if os.path.exists(DATA_COLLECTION_CHECKPOINT_FILE):
            try:
                with open(DATA_COLLECTION_CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    last_processed = data.get("last_processed_token_id_for_data_collection", default_start_token_id)
                    if not isinstance(last_processed, int) or last_processed < default_start_token_id:
                        logger.warning("Invalid data collection checkpoint value '%s'. Resetting.", last_processed)
                        return default_start_token_id
                    logger.info("Data collection checkpoint: Last processed token ID was %d.", last_processed)
                    return last_processed
            except (json.JSONDecodeError, OSError, TypeError) as e:
                logger.error("Error reading data collection checkpoint %s: %s. Starting from beginning.", DATA_COLLECTION_CHECKPOINT_FILE, e)
                return default_start_token_id
        logger.info("No data collection checkpoint file found. Starting from beginning.")
        return default_start_token_id

    def _update_checkpoint_data_collection(self, token_id: int):
        """
        Saves the last successfully
        processed token ID for data collection.
        """
        try:
            with open(DATA_COLLECTION_CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
                json.dump({"last_processed_token_id_for_data_collection": token_id}, f, indent=2)
            logger.debug("Data collection checkpoint updated to token ID %d.", token_id)
        except OSError as e:
            logger.error("Failed to update data collection checkpoint %s: %s", DATA_COLLECTION_CHECKPOINT_FILE, e)

    def _get_checkpoint_classifier_training(self) -> Optional[str]:
        """
        Loads the last successfully
        trained/loaded classifier's trait type.
        """
        if os.path.exists(CLASSIFIER_TRAINING_CHECKPOINT_FILE):
            try:
                with open(CLASSIFIER_TRAINING_CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    last_trained_trait = data.get("last_successfully_trained_classifier_trait_type")
                    if last_trained_trait and isinstance(last_trained_trait, str):
                        logger.info("Classifier training checkpoint: Last successfully trained/loaded trait was '%s'.", last_trained_trait)
                        return last_trained_trait
                    elif last_trained_trait:
                        logger.warning("Invalid classifier training checkpoint trait type '%s'. Resetting.", last_trained_trait)
                        return None
            except (json.JSONDecodeError, OSError, TypeError) as e:
                logger.error("Error reading classifier training checkpoint %s: %s. Starting from beginning.", CLASSIFIER_TRAINING_CHECKPOINT_FILE, e)
                return None
        logger.info("No classifier training checkpoint file found. Starting from beginning.")
        return None

    def _update_checkpoint_classifier_training(self, trait_type: str):
        """
        Saves the last successfully
        trained/loaded classifier's trait type.
        """
        try:
            with open(CLASSIFIER_TRAINING_CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
                json.dump({"last_successfully_trained_classifier_trait_type": trait_type}, f, indent=2)
            logger.debug("Classifier training checkpoint updated to trait '%s'.", trait_type)
        except OSError as e:
            logger.error("Failed to update classifier training checkpoint %s: %s", CLASSIFIER_TRAINING_CHECKPOINT_FILE, e)

    def _load_corrections(self):
        """
        Loads the trait corrections
        file for the feedback loop. Populates `self.corrections_data`.
        `self.corrections_data` and `self.used_corrections_count` are
        assumed to be initialized in `__init__`.
        """
        if not os.path.exists(CORRECTIONS_FILE_PATH):
            logger.info("Corrections file not found at '%s'. Proceeding without feedback.", CORRECTIONS_FILE_PATH)
            return

        logger.info("Loading corrections data from %s", CORRECTIONS_FILE_PATH)
        try:
            with open(CORRECTIONS_FILE_PATH, 'r', encoding='utf-8') as f:
                loaded_corrections = json.load(f)

            if not isinstance(loaded_corrections, dict):
                logger.warning("Corrections file '%s' does not contain a valid dictionary. Ignoring.", CORRECTIONS_FILE_PATH)
                return

            valid_entries = 0
            for token_id_str, traits_dict in loaded_corrections.items():
                if isinstance(traits_dict, dict):
                    self.corrections_data[token_id_str] = traits_dict
                    valid_entries += 1
                else:
                    logger.warning("Invalid entry in corrections file for token '%s'. Expected dict, got %s.", token_id_str, type(traits_dict))

            logger.info("Successfully loaded %d valid correction entries from %s.", valid_entries, CORRECTIONS_FILE_PATH)

        except json.JSONDecodeError as e:
            logger.error("Error decoding JSON from corrections file %s: %s. Ignoring corrections.", CORRECTIONS_FILE_PATH, e)
            self.corrections_data = {}
        except OSError as e:
            logger.error("Error reading corrections file %s: %s. Ignoring corrections.", CORRECTIONS_FILE_PATH, e)
            self.corrections_data = {}
        except RuntimeError as e:
            logger.error("Unexpected error loading corrections file %s: %s. Ignoring corrections.", CORRECTIONS_FILE_PATH, e, exc_info=True)
            self.corrections_data = {}

    def update_status(self, message: str, show_busy_indicator: bool = False):
        """
        Updates the
        status bar label.
        """
        self.status_label.config(text=message)
        if self.busy_indicator_pb:
            if show_busy_indicator:
                if not self.busy_indicator_pb.winfo_ismapped():
                    self.busy_indicator_pb.grid()
                    self.busy_indicator_pb.start(10)
            else:
                if self.busy_indicator_pb.winfo_ismapped():
                    self.busy_indicator_pb.stop()
                    self.busy_indicator_pb.grid_remove()
        self.master.update_idletasks()

    def _start_process_confirmed(self):
        """
        Confirms before starting
        the potentially long process.
        """
        if not self.image_map:
            messagebox.showerror("Error", "Image map is not loaded. Cannot start.")
            return

        confirm_msg = "This will start the data collection and training process. It may take a significant amount of time, especially with hyperparameter tuning enabled. Ensure classifiers from previous runs (if any) are backed up if you don't want them overwritten. Proceed?"
        if messagebox.askyesno("Confirm Start", confirm_msg):
            self._start_training_process()

    def _start_training_process(self):
        """
        Initializes and starts
        the data collection and
        training sequence.
        """
        if self.is_running:
            logger.warning("Process already running.")
            return

        logger.info("--- Starting Enhanced Training Process via GUI ---")
        logger.info("Data Augmentation: %s", "Enabled" if ENABLE_DATA_AUGMENTATION else "Disabled")
        logger.info("Hyperparameter Tuning: %s", "Enabled" if ENABLE_HYPERPARAMETER_TUNING else "Disabled")
        logger.info("Feedback Loop (Corrections): %s", f"{len(self.corrections_data)} entries loaded"if self.corrections_data else "Disabled/Not Found")

        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        self.training_data = {}
        self.classifiers = {}
        self.token_ids_to_process = list(range(TOKEN_ID_START, NUM_TOKENS_TO_TRAIN + 1))
        self.verification_indices = set()
        self.all_trait_types = []
        self.verification_possible = VERIFICATION_ENABLED
        self.current_token_index = 0
        self.all_trait_types = []
        self.current_classifier_index = -1
        self.processed_tokens_count = 0
        self.augmented_data_count = 0
        self.verified_correct_count = 0
        self.corrected_count = 0
        self.discarded_count = 0
        self.skipped_verification_count = 0
        self.trained_classifier_count = 0
        self.failed_classifier_count = 0
        self.used_corrections_count = 0

        last_processed_token_for_data = self._get_checkpoint_data_collection()
        data_collection_fully_complete = last_processed_token_for_data == NUM_TOKENS_TO_TRAIN
        data_loaded_from_checkpoint = False

        if data_collection_fully_complete:
            logger.info("Data collection checkpoint indicates all %d tokens were processed. Attempting to load collected data from %s.", NUM_TOKENS_TO_TRAIN, COLLECTED_DATA_FILE)
            try:
                if os.path.exists(COLLECTED_DATA_FILE):
                    with gzip.open(COLLECTED_DATA_FILE, "rb") as f:
                        loaded_data_checkpoint = pickle.load(f)
                    self.training_data = loaded_data_checkpoint.get('training_data', {})
                    self.all_trait_types = loaded_data_checkpoint.get('all_trait_types', [])
                    self.processed_tokens_count = loaded_data_checkpoint.get('processed_tokens_count', NUM_TOKENS_TO_TRAIN)
                    self.augmented_data_count = loaded_data_checkpoint.get('augmented_data_count', 0)
                    self.used_corrections_count = loaded_data_checkpoint.get('used_corrections_count', 0)
                    self.verified_correct_count = loaded_data_checkpoint.get('verified_correct_count', 0)
                    self.corrected_count = loaded_data_checkpoint.get('corrected_count', 0)
                    self.discarded_count = loaded_data_checkpoint.get('discarded_count', 0)
                    self.skipped_verification_count = loaded_data_checkpoint.get('skipped_verification_count', 0)

                    if self.training_data and self.all_trait_types:
                        logger.info("Successfully loaded previously collected training data and %d trait types from %s.", len(self.all_trait_types), COLLECTED_DATA_FILE)
                        data_loaded_from_checkpoint = True
                        self.current_token_index = len(self.token_ids_to_process)
                    else:
                        logger.warning("Loaded data from %s is empty or incomplete. Will re-collect data.", COLLECTED_DATA_FILE)
                        data_collection_fully_complete = False
                else:
                    logger.warning("Data collection checkpoint indicates completion, but %s not found. Will re-collect data.", COLLECTED_DATA_FILE)
                    data_collection_fully_complete = False
            except (pickle.PickleError, EOFError, gzip.BadGzipFile, OSError, KeyError, AttributeError) as e:
                logger.error("Failed to load collected training data from %s: %s. Will re-collect data.", COLLECTED_DATA_FILE, e)
                data_collection_fully_complete = False

        if not data_collection_fully_complete:
            self.current_token_index = 0

        if VERIFICATION_ENABLED:
            k_sample = 0
            if VERIFICATION_FREQUENCY > 0 and VERIFICATION_SAMPLE_SIZE > 0:
                num_to_verify = max(1, (NUM_TOKENS_TO_TRAIN // VERIFICATION_FREQUENCY) * VERIFICATION_SAMPLE_SIZE)
                num_to_verify = min(num_to_verify, NUM_TOKENS_TO_TRAIN)
                k_sample = 0
                available_indices_for_verification: List[int] = []
                try:
                    current_checkpoint_for_verification = self._get_checkpoint_data_collection()
                    available_indices_for_verification = [i for i, tid in enumerate(self.token_ids_to_process) if tid > current_checkpoint_for_verification]
                    k_sample = min(num_to_verify, len(available_indices_for_verification))

                    self.verification_indices = set(random.sample(available_indices_for_verification, k=k_sample))
                    logger.info("GUI Verification enabled. Will attempt to verify/correct %d (remaining out of %d eligible) tokens.", len(self.verification_indices), len(available_indices_for_verification))
                except ValueError as e:
                    logger.error("Error during verification sampling (k=%d, available_population=%d): %s. Disabling verification.", k_sample, len(available_indices_for_verification), e)
                    self.verification_possible = False
            else:
                logger.warning("Verification frequency or sample size is zero. Disabling verification.")
                self.verification_possible = False
        else:
            self.verification_possible = False

        if not self.verification_possible:
            logger.info("Verification is disabled for this run.")
            self.verification_indices = set()

        if data_loaded_from_checkpoint:
            logger.info("Skipping data collection stage as data was loaded from checkpoint file %s.", COLLECTED_DATA_FILE)
            self.progress_bar['maximum'] = len(self.token_ids_to_process)
            self.progress_bar['value'] = len(self.token_ids_to_process)
            self.progress_label.config(text="Stage 1: Data Collection Complete (Loaded from Checkpoint)")
            self.update_status("Status: Loaded collected data. Proceeding to Stage 2...")
            self.master.after(100, self._finish_data_collection)
        else:
            self.progress_label.config(text=f"Stage 1: Collecting Data (0 / {len(self.token_ids_to_process)})")
            self.progress_bar['maximum'] = len(self.token_ids_to_process)
            self.progress_bar['value'] = 0
            self.update_status("Status: Stage 1 - Collecting data...")
            logger.info("--- Stage 1: Collecting Training Data & Performing Verification ---")
            self.master.after(100, self._process_next_token)

    def _stop_process(self):
        """
        Signals the running
        process to stop gracefully.
        """
        if not self.is_running:
            return
        logger.warning("Stop requested by user.")
        self.update_status("Status: Stopping process...")
        self.is_running = False
        self.stop_button.config(state=tk.DISABLED)

    def _process_next_token(self):
        """
        Processes a single token
        for data collection, including
        feedback loop, verification,
        and augmentation.
        """
        if not self.is_running:
            logger.info("Data collection loop stopped by user request.")
            self.update_status("Status: Process stopped by user.")
            self.start_button.config(state=tk.NORMAL)
            return

        if self.current_token_index >= len(self.token_ids_to_process):
            self._finish_data_collection()
            return

        token_id = self.token_ids_to_process[self.current_token_index]
        idx = self.current_token_index

        self.progress_label.config(text=f"Stage 1: Data ({idx + 1}/{len(self.token_ids_to_process)}) - Token {token_id}")

        last_processed_token_for_data = self._get_checkpoint_data_collection()
        if token_id <= last_processed_token_for_data:
            logger.debug("Token ID %d already processed for data collection (checkpoint: %d). Skipping.", token_id, last_processed_token_for_data)
            self._schedule_next_token_step()
            return

        if self.image_map is None:
            logger.critical("CRITICAL: Image map is None in _process_next_token. This should not happen. Stopping process.")
            self.update_status("Status: ERROR - Image map became None unexpectedly.")
            self._stop_process()
            return

        load_result = load_image(token_id, self.image_map)
        if load_result is None:
            logger.warning("Failed to load image for token ID %d. Skipping.", token_id)
            self._schedule_next_token_step()
            return
        image_for_display, image_for_features = load_result

        metadata = load_metadata(token_id)
        if not metadata:
            logger.warning("Failed to load metadata for token ID %d. Skipping.", token_id)
            self._schedule_next_token_step()
            return

        initial_traits = extract_traits(metadata)
        if not initial_traits:
            logger.warning("No traits extracted for token ID %d from metadata. Skipping.", token_id)
            self._schedule_next_token_step()
            return

        current_traits = initial_traits
        used_correction = False
        token_id_str = str(token_id)
        if token_id_str in self.corrections_data:
            corrected_traits_from_file = self.corrections_data[token_id_str]
            if corrected_traits_from_file != initial_traits:
                logger.info("Applying corrections from feedback file for token %d.", token_id)
                current_traits = corrected_traits_from_file
                self.used_corrections_count += 1
                used_correction = True
            else:
                logger.debug("Corrections file data matches original metadata for token %d.", token_id)
                self.used_corrections_count += 1
                used_correction = True

        needs_verification = self.verification_possible and (
            idx in self.verification_indices) and not used_correction

        if needs_verification:
            logger.info("Requesting verification/correction for Token ID: %d", token_id)
            self.update_status(f"Status: Verifying Token {token_id}...")
            verification_result = self._run_verification_dialog(token_id, image_for_display, initial_traits)
            self.update_status(f"Status: Stage 1 - Collecting data... (Token {token_id})")

            if verification_result is None:
                self.discarded_count += 1
                logger.warning("Skipping data collection for token %d due to user discard.", token_id)
                self._schedule_next_token_step()
                return
        elif current_traits == initial_traits:
            self.verified_correct_count += 1
            logger.info("User confirmed original traits via GUI for token %d.", token_id)
        elif self.verification_possible and not used_correction:
            self.skipped_verification_count += 1
        elif used_correction:
            logger.debug("Skipping GUI verification for token %d as corrections were applied from file.", token_id)

        if not current_traits:
            logger.warning("Skipping token %d as it has no traits after potential correction/verification.", token_id)
            self._schedule_next_token_step()
            return

        images_to_process = [image_for_features]
        if ENABLE_DATA_AUGMENTATION:
            logger.debug("Augmenting image for token %d...", token_id)
            augmented_images = augment_image(image_for_features.copy())
            images_to_process = augmented_images
            self.augmented_data_count += max(0, len(images_to_process) - 1)

        token_data_added_count = 0
        for i, img_to_process in enumerate(images_to_process):
            is_original_feature_img = (i == 0 and not ENABLE_DATA_AUGMENTATION) or (i == 0 and images_to_process[0] is image_for_features)

            vector = image_to_vector(img_to_process)
            histogram = image_to_color_histogram(img_to_process)

            if vector.size == 0 and histogram.size == 0:
                logger.warning("Skipping %s image for token ID %d due to all feature extraction failing. Original feature" if is_original_feature_img else "augmented", token_id)
                continue
            if vector.size == 0:
                logger.warning("Vector feature extraction failed for %s image of token %d. Original feature" if is_original_feature_img else "augmented", token_id)
            elif histogram.size == 0:
                logger.warning("Histogram feature extraction failed for %s image of token %d. Original feature" if is_original_feature_img else "augmented", token_id)

            instance_data_added = False
            for trait_type, value in current_traits.items():
                if not value:
                    logger.debug("Skipping empty value for trait '%s' on token %d.", trait_type, token_id)
                    continue

                features = get_feature_extractor_and_data(trait_type, vector, histogram)

                if features is not None and features.size > 0:
                    self.training_data.setdefault(trait_type, []).append((features, value))
                    instance_data_added = True
                    if trait_type not in self.all_trait_types:
                        self.all_trait_types.append(trait_type)
                else:
                    logger.warning("Could not get valid/required features for trait '%s' on %s image of token %d. Skipping this trait for this image instance.", trait_type, "original feature" if is_original_feature_img else "augmented", token_id)

            if instance_data_added:
                token_data_added_count += 1

        if token_data_added_count > 0:
            self.processed_tokens_count += 1
            self._update_checkpoint_data_collection(token_id)

        self._schedule_next_token_step()

    def _schedule_next_token_step(self):
        """
        Increments index, updates
        progress bar, and schedules
        the next token processing.
        """
        self.progress_bar['value'] = self.current_token_index + 1
        self.current_token_index += 1
        self.master.after(5, self._process_next_token)

    def _run_verification_dialog(self, token_id, image, traits) -> Optional[Dict[str, str]]:
        """
        Runs the modal
        verification dialog.
        """
        dialog = VerificationWindow(self.master, token_id, image, traits, self.trait_schema_order)
        return dialog.wait_for_input()

    def _finish_data_collection(self):
        """
        Logs summary of data
        collection and transitions
        to training classifiers.
        """
        logger.info("--- Data Collection Summary ---")
        logger.info("Total tokens yielding data: %d / %d in range.", self.processed_tokens_count, len(self.token_ids_to_process))
        if ENABLE_DATA_AUGMENTATION:
            logger.info("Generated %d additional augmented data points.", self.augmented_data_count)
        if self.corrections_data:
            logger.info("Applied corrections from feedback file for %d tokens.", self.used_corrections_count)
        if VERIFICATION_ENABLED:
            logger.info("GUI Verification results: %d confirmed correct, %d corrected via GUI, %d discarded via GUI, %d skipped check.", self.verified_correct_count, self.corrected_count, self.discarded_count, self.skipped_verification_count)
        if self.discarded_count > 0:
            logger.warning("Review the log for details on %d tokens discarded during verification.", self.discarded_count)
        if self.corrected_count > 0:
            logger.info("%d tokens had their traits corrected during GUI verification.", self.corrected_count)

        if self.training_data and self.all_trait_types:
            logger.info("Saving collected training data to checkpoint file: %s", COLLECTED_DATA_FILE)
            self.update_status( f"Status: Saving collected data to {os.path.basename(COLLECTED_DATA_FILE)}... This may take a few moments.", show_busy_indicator=True)
            self.master.update_idletasks()

            save_successful = False
            try:
                data_to_save = {
                    'training_data': self.training_data,
                    'all_trait_types': self.all_trait_types,
                    'processed_tokens_count': self.processed_tokens_count,
                    'augmented_data_count': self.augmented_data_count,
                    'used_corrections_count': self.used_corrections_count,
                    'verified_correct_count': self.verified_correct_count,
                    'corrected_count': self.corrected_count,
                    'discarded_count': self.discarded_count,
                    'skipped_verification_count': self.skipped_verification_count
                }
                with gzip.open(COLLECTED_DATA_FILE, "wb") as f:
                    pickle.dump(data_to_save, f)
                logger.info("Successfully saved collected training data.")
                save_successful = True
            except (pickle.PickleError, OSError, AttributeError) as e:
                logger.error("Failed to save collected training data to %s: %s", COLLECTED_DATA_FILE, e)
                self.update_status(f"Status: ERROR saving collected data to {os.path.basename(COLLECTED_DATA_FILE)}. Check logs.", show_busy_indicator=False)

            if save_successful:
                self.update_status("Status: Collected data saved. Preparing for Stage 2...", show_busy_indicator=False)

        elif not self.training_data or not self.all_trait_types:
            logger.warning("No training data or trait types to save to checkpoint. This might be an issue if Stage 1 was expected to produce data.")
            self.update_status("Status: No training data to save. Preparing for Stage 2...", show_busy_indicator=False)

        self.all_trait_types = sorted(list(self.training_data.keys()))
        logger.info("Found data for %d trait types: %s", len(self.all_trait_types), ",".join(self.all_trait_types))

        for trait_type in self.all_trait_types:
            count = len(self.training_data.get(trait_type, []))
            logger.info(" -> Trait '%s': %d data points (incl. augmentations)", trait_type, count)

        if not self.all_trait_types:
            logger.error("No training data collected for any traits. Cannot train classifiers.")
            self.update_status("Status: ERROR No training data collected.")
            self._finalize_process(success=False)
            return

        self.progress_label.config(text=f"Stage 2: Training Classifiers (0 / {len(self.all_trait_types)})")
        self.progress_bar['maximum'] = len(self.all_trait_types)
        self.progress_bar['value'] = 0
        self.current_classifier_index = 0

        last_trained_trait = self._get_checkpoint_classifier_training()
        if last_trained_trait and last_trained_trait in self.all_trait_types:
            try:
                resume_idx = self.all_trait_types.index(last_trained_trait) + 1
                self.current_classifier_index = resume_idx
                logger.info("Resuming classifier training after trait '%s' (index %d).", last_trained_trait, resume_idx - 1)
            except ValueError:
                logger.warning("Could not find checkpointed trait '%s' in current trait list. Starting training from beginning.", last_trained_trait)
                self.current_classifier_index = 0

        self.trained_classifier_count = 0
        self.failed_classifier_count = 0
        logger.info("--- Stage 2: Training Classifiers ---")
        self.master.after(100, self._train_next_classifier)

    def _train_next_classifier(self):
        """
        Trains a single classifier, potentially
        using GridSearchCV for hyperparameter tuning.
        """
        if not self.is_running:
            logger.info("Classifier training loop stopped by user request.")
            self.update_status("Status: Process stopped by user.")
            self.start_button.config(state=tk.NORMAL)
            return

        if self.current_classifier_index >= len(self.all_trait_types):
            self._finish_classifier_training()
            return

        trait_type = self.all_trait_types[self.current_classifier_index]
        self.progress_label.config(text=f"Stage 2: Training Classifiers ({self.current_classifier_index + 1} / {len(self.all_trait_types)}) - '{trait_type}'")
        self.update_status(f"Status: Training classifier for '{trait_type}'...")

        try:
            entries = self.training_data.get(trait_type, [])
            sanitized_trait = trait_type.replace(' ', '_').replace('/', '_')
            model_path = os.path.join(CLASSIFIER_DIR, f"{sanitized_trait}.pkl.gz")

            loaded_successfully = False
            loaded_pipeline = None

            if not ENABLE_HYPERPARAMETER_TUNING:
                if os.path.exists(model_path):
                    logger.info("Attempting to load existing classifier for '%s' from %s", trait_type, model_path)
                    try:
                        with gzip.open(model_path, "rb") as f:
                            loaded_pipeline = pickle.load(f)

                        if isinstance(loaded_pipeline, Pipeline) and hasattr(loaded_pipeline, 'predict'):
                            self.classifiers[trait_type] = loaded_pipeline
                            logger.info("Successfully loaded existing classifier for '%s'.", trait_type)
                            loaded_successfully = True
                            self.trained_classifier_count += 1
                            self._update_checkpoint_classifier_training(trait_type)
                        else:
                            logger.warning("Loaded object for '%s' from %s is not a valid Pipeline. Will retrain.", trait_type, model_path)
                            loaded_pipeline = None
                    except (FileNotFoundError, pickle.PickleError, EOFError, gzip.BadGzipFile, AttributeError, ValueError, ImportError) as e_load:
                        logger.info("Failed to load existing classifier for '%s' from %s: %s. Will attempt to retrain.", trait_type, model_path, e_load)
                        loaded_successfully = False
                        loaded_pipeline = None

            if not loaded_successfully:
                if trait_type in ["One Of One", "Special"] and ENABLE_HYPERPARAMETER_TUNING:
                    logger.warning("Skipping hyperparameter tuning and training for low-sample trait '%s' (%d entries). Manual review/creation of this classifier is recommended if needed. If a valid model exists and ENABLE_HYPERPARAMETER_TUNING is set to False, it might be loaded.", trait_type, len(entries))
                    self.failed_classifier_count += 1
                    self._update_checkpoint_classifier_training(trait_type)
                else:
                    logger.info("Training new classifier for trait type '%s'.", trait_type)
                    if not entries:
                        logger.warning("No training data for '%s'. Skipping training.", trait_type)
                        self.failed_classifier_count += 1
                        self._update_checkpoint_classifier_training(trait_type)
                    else:
                        features_list = [e[0] for e in entries]
                        labels = [e[1] for e in entries]
                        if not features_list:
                            raise ValueError("Empty features list for training.")

                        first_shape = features_list[0].shape
                        if not all(isinstance(f, np.ndarray) and f.shape == first_shape for f in features_list):
                            inconsistent_shapes = [f.shape for f in features_list if not (isinstance(f, np.ndarray) and f.shape == first_shape)]
                            raise ValueError(f"Inconsistent feature shapes detected for trait '{trait_type}'. Expected {first_shape}, found shapes like {inconsistent_shapes[:5]}")

                        feature_array = np.array(features_list)
                        if feature_array.ndim == 1:
                            feature_array = feature_array.reshape(-1, 1)

                        n_samples = feature_array.shape[0]
                        n_features = feature_array.shape[1] if feature_array.ndim > 1 else 0
                        unique_labels = set(labels)

                        if n_samples < 2 or len(unique_labels) < 2:
                            logger.warning("Cannot train '%s': Samples=%d, Unique Labels=%d. Need at least 2 of each. Skipping.", trait_type, n_samples, len(unique_labels))
                            self.failed_classifier_count += 1
                            self._update_checkpoint_classifier_training(trait_type)
                        else:
                            pipeline_steps: List[Tuple[str, Any]] = [('scaler', StandardScaler())]
                            max_pca_components = min(n_samples - 1, n_features) if n_samples > 1 and n_features > 0 else 0

                            if trait_type != "Background" and max_pca_components > 1:
                                pca_n_components = min(50, max_pca_components)
                                logger.info("Adding PCA step for '%s' (n_components: %d, svd_solver: 'randomized', max_pca_components_theoretical: %d)", trait_type, pca_n_components, max_pca_components)
                                pipeline_steps.append(('pca', PCA(n_components=pca_n_components, svd_solver='randomized', random_state=42)))
                            elif trait_type != "Background":
                                logger.info("Skipping PCA for '%s' (max_pca_components_theoretical=%d, not enough for reduction or n_samples/n_features too small)", trait_type, max_pca_components)

                            pipeline_steps.append(
                                ('svm', SVC(probability=True, random_state=42, class_weight='balanced')))

                            pipeline = Pipeline(pipeline_steps)
                            best_pipeline = None

                            if ENABLE_HYPERPARAMETER_TUNING:
                                param_grid = {}
                                if 'pca' in pipeline.named_steps:
                                    candidate_grid_pca_comps = [10, 20, 30]
                                    pca_options = [comp for comp in candidate_grid_pca_comps if 0 < comp <= max_pca_components]
                                    pca_options = sorted(list(set(pca_options)))
                                    if not pca_options and max_pca_components > 0:
                                        opt1 = max(1, max_pca_components // 2)
                                        opt2 = max_pca_components
                                        temp_options = []
                                        if 0 < opt1 <= max_pca_components:
                                            temp_options.append(opt1)
                                        if 0 < opt2 <= max_pca_components and opt2 != opt1:
                                            temp_options.append(opt2)
                                        pca_options = sorted(list(set(temp_options)))
                                    if pca_options:
                                        param_grid['pca__n_components'] = pca_options
                                param_grid['svm__C'] = [0.1, 1, 5]
                                param_grid['svm__gamma'] = ['scale']
                                param_grid['svm__kernel'] = ['rbf', 'linear']

                                min_class_count = min(Counter(labels).values()) if labels and Counter(labels) else 0
                                can_run_grid_search = True
                                cv_folds_for_grid_search = 3

                                if not param_grid.get('pca__n_components') and not param_grid.get('svm__C'):
                                    logger.info("No valid parameters for GridSearchCV for '%s'. Fitting directly.", trait_type)
                                    can_run_grid_search = False
                                elif min_class_count < 2:
                                    logger.warning("Cannot perform GridSearchCV for '%s': least populated class has %d sample(s). Fitting directly.", trait_type, min_class_count)
                                    can_run_grid_search = False
                                elif min_class_count < cv_folds_for_grid_search:
                                    cv_folds_for_grid_search = min_class_count
                                    logger.info("Adjusting GridSearchCV cv to %d for '%s' (min class count: %d).", cv_folds_for_grid_search, trait_type, min_class_count)

                                if not can_run_grid_search:
                                    logger.info("Fitting pipeline for '%s' with %d samples (GridSearchCV skipped)...", trait_type, n_samples)
                                    pipeline.fit(feature_array, labels)
                                    best_pipeline = pipeline
                                else:
                                    search = GridSearchCV(pipeline, param_grid, cv=cv_folds_for_grid_search, n_jobs=1, scoring='accuracy', error_score='raise', verbose=1)
                                    self.update_status(f"Status: GridSearchCV for '{trait_type}'... This may take a few moments.", show_busy_indicator=True)
                                    self.master.update_idletasks()
                                    logger.info("Performing GridSearchCV for '%s' with %d samples (cv=%d)...", trait_type, n_samples, cv_folds_for_grid_search)
                                    search.fit(feature_array, labels)
                                    logger.info("GridSearchCV completed for '%s'. Best params: %s, Best score: %.4f", trait_type, search.best_params_, search.best_score_)
                                    best_pipeline = search.best_estimator_
                                    self.update_status(f"Status: GridSearchCV for '{trait_type}' complete.", show_busy_indicator=False)
                            else:
                                logger.info("Fitting pipeline for '%s' with %d samples (tuning disabled)...", trait_type, n_samples)
                                pipeline.fit(feature_array, labels)
                                best_pipeline = pipeline

                            if best_pipeline is not None:
                                self.classifiers[trait_type] = best_pipeline
                                self.trained_classifier_count += 1
                                logger.info("Saving trained pipeline for '%s' to %s", trait_type, model_path)
                                with gzip.open(model_path, "wb") as f:
                                    pickle.dump(best_pipeline, f)
                                logger.info("Successfully saved pipeline for '%s'.", trait_type)
                                self._update_checkpoint_classifier_training(trait_type)
                            else:
                                self.failed_classifier_count += 1
                                logger.error("Pipeline training FAILED for '%s'. Classifier will not be available.", trait_type)
                                self._update_checkpoint_classifier_training(trait_type)

        except KeyboardInterrupt:
            logger.critical("KeyboardInterrupt occurred while processing trait '%s'. Process will attempt to stop.", trait_type)
            self.update_status(f"Status: Interrupted during processing of '{trait_type}'. Stopping.", show_busy_indicator=False)
            self._update_checkpoint_classifier_training(trait_type)
            self._stop_process()
            return

        except Exception as e:
            logger.error("Unhandled exception processing trait '%s': %s. Skipping to next trait.", trait_type, e, exc_info=True)
            self.failed_classifier_count += 1
            self.update_status(f"Status: Error on trait '{trait_type}'. Skipping.", show_busy_indicator=False)
            self._update_checkpoint_classifier_training(trait_type)

        if self.is_running:
            self._schedule_next_classifier_step()

    def _schedule_next_classifier_step(self):
        """
        Increments index, updates
        progress bar, and schedules
        the next classifier training.
        """
        self.progress_bar['value'] = self.current_classifier_index + 1
        self.current_classifier_index += 1
        self.master.after(5, self._train_next_classifier)

    def _finish_classifier_training(self):
        """
        Logs summary of
        classifier training
        and finalizes the
        process.
        """
        logger.info("--- Classifier Training Complete ---")
        total_possible_traits = len(self.all_trait_types) if self.all_trait_types else 0
        logger.info( "Successfully trained/loaded %d out of %d trait classifiers.", self.trained_classifier_count, total_possible_traits)

        if self.failed_classifier_count > 0 or self.trained_classifier_count < total_possible_traits:
            missing_or_failed = set(self.all_trait_types) - set(self.classifiers.keys())
            if missing_or_failed:
                logger.warning("Failed to train, load, or skipped classifiers for: %s", ", ".join(sorted(list(missing_or_failed))))

        self.update_status("Status: Process Completed Successfully.")

        messagebox.showinfo(
            "Process Complete",
            f"Training process finished.\n\n"
            f"Tokens Processed: {self.processed_tokens_count}\n"
            f"Augmented Data Points Added: {self.augmented_data_count}\n"
            f"Corrections Used (Feedback): {self.used_corrections_count}\n"
            f"Verified/Corrected (GUI): {self.verified_correct_count + self.corrected_count}\n"
            f"Discarded (GUI): {self.discarded_count}\n"
            f"Classifiers Trained/Loaded:{self.trained_classifier_count}\n"
            f"Classifiers Failed/Skipped: {self.failed_classifier_count}\n\n"
            f"Check log file for details: \n{LOG_FILE_PATH}"
        )
        self._finalize_process(success=True)

    def _finalize_process(self, success: bool):
        """
        Resets GUI state after
        process completion or stop.
        """
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        if not success:
            self.update_status("Status: Process Finished with Errors or Stopped.")
        else:
            self.update_status("Status: Idle. Process Completed.")

class GuiLogger(logging.Handler):
    """
    A logging handler that
    directs messages to a
    Tkinter Text widget.
    """

    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.config(state=tk.DISABLED)

    def emit(self, record):
        try:
            msg = self.format(record)
            self.text_widget.config(state=tk.NORMAL)
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.see(tk.END)
            self.text_widget.config(state=tk.DISABLED)
            self.text_widget.update_idletasks()
        except (RuntimeError, tk.TclError):
            self.handleError(record)

if __name__ == "__main__":
    root = tk.Tk()
    app = TraitTrainerApp(root)

    def on_closing():
        """
        Handles pre-destroy tasks like
        removing GUI-specific log handlers.
        """
        if app.gui_log_handler:
            logging.getLogger().removeHandler(app.gui_log_handler)
            app.gui_log_handler = None
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
    logger.info("--- GUI Application Closed ---")
