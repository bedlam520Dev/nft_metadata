#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0301
# pylint: disable=C0302
# pylint: disable=W0718
"""
Part 3: Metadata Verification GUI
Provides a graphical user interface (Tkinter) to randomly sample
and visually verify the correctness of generated NFT metadata
against their corresponding images. Allows users to `Verify`
or `Reject` pairs, logs rejections, and generates an analysis
report highlighting potentially problematic traits or values.
"""
import gzip
import json
import logging
import os
import pickle
import random
import re
import subprocess
import sys
import tkinter as tk
from tkinter import Event, messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageTk, UnidentifiedImageError
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = "."
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
LOG_DIR = os.path.join(BASE_DIR, "logs")
CONFIG_DIR = os.path.join(BASE_DIR, "config")
CLASSIFIER_DIR = os.path.join(CONFIG_DIR, "classifiers")
MAIN_LOG_FILE = os.path.join(LOG_DIR, "verify_metadata.log")
REJECT_LOG_FILE = os.path.join(LOG_DIR, "verify_rejections.log")
ANALYSIS_LOG_FILE = os.path.join(LOG_DIR, "verify_low_confidence_analysis.log")
IMAGE_JSON_PATH = os.path.join(CONFIG_DIR, "images.json")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

TRAIT_PREDICTOR_LOG_FILE = os.path.join(LOG_DIR, "trait_predictor_v2.log")
COLLECTED_DATA_FILE_PATH = os.path.join(CHECKPOINT_DIR, "part1_collected_training_data.pkl.gz")

IMAGE_SIZE = (128, 128)
HISTOGRAM_BINS = 16

IMAGE_DISPLAY_SIZE = (400, 400)
INITIAL_WINDOW_WIDTH = 750
INITIAL_WINDOW_HEIGHT = 800

try:
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)
    os.makedirs(CLASSIFIER_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
except OSError as e:
    print(f"CRITICAL: Error creating directories: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(MAIN_LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger()

def image_to_vector(image: Image.Image) -> np.ndarray:
    """
    Converts an image into a flattened numpy array vector.
    Ensures image is resized to IMAGE_SIZE.
    """
    try:
        if image.size != IMAGE_SIZE:
            image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32).flatten()
    except Exception:
        logger.error("Failed to convert image to vector: %d", exc_info=True)
        return np.array([])

def image_to_color_histogram(image: Image.Image, bins: int = HISTOGRAM_BINS) -> np.ndarray:
    """
    Calculates a flattened, normalized 3D color histogram for an RGB image.
    Ensures image is resized to IMAGE_SIZE.
    """
    try:
        if image.size != IMAGE_SIZE:
            image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        img_array = np.asarray(image)
        hist_r = np.histogram(img_array[:, :, 0], bins=bins, range=(0, 256))[0]
        hist_g = np.histogram(img_array[:, :, 1], bins=bins, range=(0, 256))[0]
        hist_b = np.histogram(img_array[:, :, 2], bins=bins, range=(0, 256))[0]
        total_pixels = image.width * image.height
        if total_pixels == 0:
            logger.error("Cannot calculate histogram for zero-pixel image.")
            return np.array([])
        norm_factor = float(total_pixels)
        return np.concatenate((hist_r / norm_factor, hist_g / norm_factor, hist_b / norm_factor))
    except Exception:
        logger.error("Failed to calculate color histogram: %d", exc_info=True)
        return np.array([])

def get_feature_extractor_and_data(
    trait_type: str,
    image_vector: np.ndarray,
    image_histogram: np.ndarray
) -> Optional[np.ndarray]:
    """
    Selects the appropriate feature vector based on trait type.
    """
    if trait_type == "Background":
        return image_histogram if image_histogram.size > 0 else None
    return image_vector if image_vector.size > 0 else None

def parse_low_confidence_skips(log_file_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parses the trait_predictor_v2.log file to find traits skipped due to low confidence.
    Returns a dictionary where keys are trait types and values are lists of
    dictionaries, each containing 'token_id', 'predicted_value', and 'confidence'.
    """
    low_confidence_data: Dict[str, List[Dict[str, Any]]] = {}
    log_pattern = re.compile(
        r"Token (\d+), Trait '([^']+)': Prediction '([^']*)' has low confidence \((\d\.\d+) < \d\.\d+\)\."
    )
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if "has low confidence" in line:
                    match = log_pattern.search(line)
                    if match:
                        token_id = int(match.group(1))
                        trait_type = match.group(2)
                        predicted_value = match.group(3)
                        confidence = float(match.group(4))

                        if trait_type not in low_confidence_data:
                            low_confidence_data[trait_type] = []
                        low_confidence_data[trait_type].append({
                            "token_id": token_id,
                            "predicted_value": predicted_value,
                            "confidence": confidence
                        })
        logger.info("Parsed low confidence data from {log_file_path}. Found issues for {len(low_confidence_data)} trait types.")
    except FileNotFoundError:
        logger.error("Log file not found: {log_file_path}")
    except Exception:
        logger.error("Error parsing log file {log_file_path}: %d", exc_info=True)
    return low_confidence_data


def load_local_image_original(token_id: int, image_map: Dict[str, str]) -> Optional[Image.Image]:
    """
    Loads an image from the local IMAGES_DIR based on token_id and image_map.
    Returns the original size image. Handles errors.
    Args:
        token_id: The integer ID of the token.
        image_map: A dictionary mapping string token IDs to image filenames.
    Returns:
        A PIL Image object (original size), or None if loading fails.
    """
    filename = image_map.get(str(token_id))
    if not filename:
        logger.error(
            "Filename not found in image_map for token ID %d.", token_id)
        return None

    local_image_path = os.path.join(IMAGES_DIR, filename)
    logger.debug("Attempting to load image for token %d from local path: %s", token_id, local_image_path)

    try:
        if not os.path.exists(local_image_path):
            logger.error("Local image file not found for token %d: %s", token_id, local_image_path)
            return None

        original_full_image = Image.open(local_image_path).convert("RGB")
        logger.debug("Successfully loaded original image for token %d from %s", token_id, local_image_path)
        return original_full_image

    except FileNotFoundError:
        logger.error("Local image file not found (FileNotFoundError) for token %d: %s", token_id, local_image_path)
        return None
    except UnidentifiedImageError:
        logger.error("Failed to load image for token %d from %s: Cannot identify image file", token_id, local_image_path)
        return None
    except (OSError, IOError, ValueError) as e:
        logger.error("Failed processing image for token %d from %s: %s", token_id, local_image_path, str(e))
        return None
    except RuntimeError as e:
        logger.error("Unexpected error loading image for token %d from %s: %s", token_id, local_image_path, e, exc_info=True)
        return None


def load_metadata(token_id: int) -> Optional[Dict]:
    """
    Loads metadata JSON from the metadata
    directory for a given token ID.
    Args:
        token_id: The integer ID of the token to
        load metadata for.
    Returns:
        A dictionary containing the loaded metadata,
        or None if loading fails.
    """
    metadata_path = os.path.join(METADATA_DIR, str(token_id))
    if not os.path.exists(metadata_path):
        logger.error("Metadata file not found for token ID %d at %s", token_id, metadata_path)
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON format in %s for token ID %d: %s", metadata_path, token_id, e)
        return None
    except (OSError, IOError) as e:
        logger.error("OS error reading metadata file %s for token ID %d: %s", metadata_path, token_id, e)
        return None
    except RuntimeError as e:
        logger.error("Unexpected error loading metadata for token ID %d: %s", token_id, e, exc_info=True)
        return None


def format_metadata_display(metadata: Optional[Dict[str, Any]]) -> str:
    """
    Formats the full metadata dictionary into a
    pretty-printed JSON string.
    Args:
        metadata: The dictionary containing the
        token's metadata.
    Returns:
        A formatted string representation of
        the metadata, or an error message.
    """
    if not metadata:
        return "Error loading metadata."
    try:
        if 'attributes' not in metadata:
            metadata['attributes'] = []
        elif not isinstance(metadata['attributes'], list):
            metadata['attributes'] = []

        return json.dumps(metadata, indent=2)
    except (TypeError, ValueError) as e:
        logger.error("Error formatting metadata for display: %s", e)
        return f"Error formatting metadata:\n{str(metadata)}"


def open_file(filepath: str):
    """
    Opens a file using the system's default
    application in a cross-platform way.
    Args:
        filepath: The absolute or relative
        path to the file to open.
    """
    try:
        if not os.path.exists(filepath):
            logger.warning("Cannot open file: File does not exist at %s", filepath)
            messagebox.showwarning("File Not Found",
                                    f"Could not find file:\n{filepath}")
            return
        if sys.platform == "win32":
            os.startfile(os.path.normpath(filepath))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", filepath])
        else:
            subprocess.Popen(["xdg-open", filepath])
        logger.info("Attempted to open file: %s", filepath)
    except (OSError, subprocess.SubprocessError) as e:
        logger.error("Failed to open file %s: %s", filepath, e)
        messagebox.showerror("Error Opening File",
                                f"Could not open file:\n{filepath}\n\nError: {e}")


class CorrectionDialog(tk.Toplevel):
    """
    GUI window for correcting\
    a single trait for a token.
    """
    def __init__(self, master, token_id: int, image: Image.Image,
                    trait_type: str, current_value: str):
        super().__init__(master)
        self.title(f"Correct Trait for Token {token_id}")
        self.grab_set()
        self.transient(master)
        self.token_id = token_id
        self.image = image
        self.trait_type = trait_type
        self.current_value = current_value
        self.corrected_value: Optional[str] = None

        self.style = ttk.Style(self)
        self.style.theme_use('clam')

        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        try:
            display_img = image.copy()
            display_img.thumbnail(IMAGE_DISPLAY_SIZE, Image.Resampling.LANCZOS)
            self.img_tk = ImageTk.PhotoImage(display_img)
            img_label = tk.Label(main_frame, image=self.img_tk, relief="groove", borderwidth=1)
            img_label.pack(pady=10)
        except Exception:
            logger.error("Error displaying image in CorrectionDialog: %d")
            img_label = tk.Label(main_frame, text="Image display error")
            img_label.pack(pady=10)

        ttk.Label(main_frame, text=f"Token ID: {token_id}").pack()
        ttk.Label(main_frame, text=f"Trait Type: {trait_type}").pack()
        ttk.Label(main_frame, text=f"Current (low-confidence) Value: {current_value}").pack()

        ttk.Label(main_frame, text="Enter Correct Value:").pack(pady=(10,0))
        self.entry_value = ttk.Entry(main_frame, width=40)
        self.entry_value.insert(0, current_value)
        self.entry_value.pack(pady=5)
        self.entry_value.focus_set()

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Submit Correction", command=self._submit).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Skip Item", command=self._skip).pack(side=tk.LEFT, padx=5)

        self.bind('<Return>', lambda event=None: self._submit())
        self.bind('<Escape>', lambda event=None: self._skip())
        self.update_idletasks()
        x = master.winfo_x() + (master.winfo_width() // 2) - (self.winfo_width() // 2)
        y = master.winfo_y() + (master.winfo_height() // 2) - (self.winfo_height() // 2)
        self.geometry(f"+{x}+{y}")

    def _submit(self):
        """
        Submit function...
        """
        self.corrected_value = self.entry_value.get().strip()
        if not self.corrected_value and not messagebox.askyesno("Confirm Empty", "Submit empty value (will be treated as 'None')?", parent=self):
            return
        self.corrected_value = self.corrected_value if self.corrected_value else "None"
        logger.info("User submitted correction for token {self.token_id}, trait '{self.trait_type}': '{self.corrected_value}'")
        self.destroy()

    def _skip(self):
        """
        Skips...
        """
        self.corrected_value = None
        logger.info("User skipped correction for token {self.token_id}, trait '{self.trait_type}'.")
        self.destroy()

    def wait_for_input(self) -> Optional[str]:
        """
        Waits...
        """
        self.wait_window(self)
        return self.corrected_value

class MetadataVerifierApp:
    """
    Main application class for the
    Tkinter-based metadata verification GUI.
    Handles UI setup, data loading, user
    interactions (verify/reject), and manages
    the verification workflow
    from start to completion analysis.
    """
    BG_COLOR = "#ECECEC"
    TEXT_COLOR = "#333333"
    HEADER_COLOR = "#1E3A5F"
    BUTTON_COLOR = "#F0F0F0"
    REJECT_COLOR = "#D32F2F"
    VERIFY_COLOR = "#388E3C"

    def __init__(self, master: tk.Tk):
        """
        Initializes the application, sets up styles, frames, and widgets.
        Args:
            master: The main Tkinter root window (often called root).
        """
        self.master = master
        self.master.title("Metadata Verifier")
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        x_coord = int((screen_width / 2) - (INITIAL_WINDOW_WIDTH / 2))
        y_coord = int((screen_height / 2) - (INITIAL_WINDOW_HEIGHT / 2))
        x_coord = max(0, x_coord)
        y_coord = max(0, y_coord)
        self.master.geometry(f"{INITIAL_WINDOW_WIDTH}x{INITIAL_WINDOW_HEIGHT}+{x_coord}+{y_coord}")
        self.master.minsize(600, 650)

        self.low_confidence_data: Dict[str, List[Dict[str, Any]]] = {}
        self.newly_corrected_data: Dict[str, List[Tuple[np.ndarray, str]]] = {}
        self.current_trait_for_correction_queue: List[str] = []
        self.current_items_for_trait_queue: List[Dict[str, Any]] = []
        self.current_trait_being_corrected: Optional[str] = None
        self.current_pil_image_for_correction: Optional[Image.Image] = None
        self.cd_img_tk: Optional[ImageTk.PhotoImage] = None
        self.retraining_report: List[str] = []

        self.image_map: Optional[Dict] = None

        self._load_image_map()
        if self.image_map is None:
            messagebox.showerror("Fatal Error",
                                    f"Could not load image map from:\n{IMAGE_JSON_PATH}\n\n"
                                    "Please check the file exists and is valid JSON.\n"
                                    "See log file for details.")
            logger.critical("Exiting due to failed image map load.")
            self.master.after(100, self.master.destroy)
            return

        self._setup_styles()
        self._setup_frames()
        self._setup_initial_screen()
        self._setup_correction_ui_elements()
        self._setup_complete_widgets()

        self.cd_image_label.bind('<Configure>', self._on_resize_correction_image)

        if self.image_map is not None:
            self.initial_screen_frame.pack(fill=tk.BOTH, expand=True)

    def _load_image_map(self):
        """
            Loads the
            images.json
            file into
            self.image_map.
        """
        logger.info("Attempting to load image map from %s", IMAGE_JSON_PATH)
        try:
            with open(IMAGE_JSON_PATH, 'r', encoding='utf-8') as f:
                self.image_map = json.load(f)
            if not isinstance(self.image_map, dict):
                logger.error("Image map loaded from %s is not a dictionary.", IMAGE_JSON_PATH)
                self.image_map = None
            else:
                logger.info("Successfully loaded image map with %d entries.", len(self.image_map))
        except FileNotFoundError:
            logger.error("Image map file not found: %s", IMAGE_JSON_PATH)
            self.image_map = None
        except json.JSONDecodeError as e:
            logger.error("Error decoding JSON from image map file %s: %s", IMAGE_JSON_PATH, e)
            self.image_map = None
        except (OSError, IOError) as e:
            logger.error("OS error reading image map file %s: %s", IMAGE_JSON_PATH, e)
            self.image_map = None
        except RuntimeError as e:
            logger.error("Unexpected error loading image map %s: %s", IMAGE_JSON_PATH, e, exc_info=True)
            self.image_map = None

    def _setup_styles(self):
        """
        Configures
        ttk styles
        for the
        application.
        """
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.master.configure(bg=self.BG_COLOR)

        self.style.configure("TFrame", background=self.BG_COLOR)
        self.style.configure("TLabel", padding=6, font=('Segoe UI', 10), background=self.BG_COLOR, foreground=self.TEXT_COLOR)
        self.style.configure("Header.TLabel", font=('Segoe UI', 14, 'bold'), foreground=self.HEADER_COLOR, background=self.BG_COLOR)
        self.style.configure("Status.TLabel", font=('Segoe UI', 11), background=self.BG_COLOR, foreground=self.TEXT_COLOR)
        self.style.configure("TButton", padding=(10, 5), font=('Segoe UI', 10, 'bold'), background=self.BUTTON_COLOR, relief="raised")
        self.style.map("TButton", background=[('active', '#E0E0E0')])

        self.style.configure("TProgressbar", thickness=20)

        self.style.configure("Reject.TButton", foreground="white", background=self.REJECT_COLOR)
        self.style.map("Reject.TButton", background=[('active', '#C62828')])

        self.style.configure("Verify.TButton", foreground="white", background=self.VERIFY_COLOR)
        self.style.map("Verify.TButton", background=[('active', '#2E7D32')])

        self.style.configure("Close.TButton", foreground=self.TEXT_COLOR, background="#D0D0D0")
        self.style.map("Close.TButton", background=[('active', '#BDBDBD')])

    def _setup_frames(self):
        """
        Initializes the main frames
        used by the application.
        """
        self.initial_screen_frame = ttk.Frame(self.master, padding="30")
        self.correction_display_frame = ttk.Frame(self.master, padding="15")
        self.complete_frame = ttk.Frame(self.master, padding="30")

    def _setup_initial_screen(self):
        """
        Sets up the initial screen with
        a button to start the analysis.
        """
        ttk.Label(self.initial_screen_frame, text="NFT Metadata Verifier & Classifier Strengthener", style="Header.TLabel").pack(pady=20)
        ttk.Label(self.initial_screen_frame,
                    text="This tool will:\n"
                        "1. Analyze 'trait_predictor_v2.log' for low-confidence predictions.\n"
                        "2. Allow you to manually correct a sample of these.\n"
                        "3. Retrain the affected classifiers with your corrections.\n"
                        "4. Optionally, re-run metadata generation (metagen.py).",
                    justify=tk.LEFT).pack(pady=10)

        self.analyze_button = ttk.Button(self.initial_screen_frame, text="Analyze Logs & Start Verification",
                                            command=self._start_full_process, width=30)
        self.analyze_button.pack(pady=20, ipadx=10, ipady=5)

    def _setup_correction_ui_elements(self):
        """
        Sets up widgets for the main correction
        screen (image, trait info, etc.).
        """
        self.correction_display_frame.columnconfigure(0, weight=1)
        self.correction_display_frame.rowconfigure(1, weight=1)

        self.cd_token_label = ttk.Label(self.correction_display_frame, text="Token ID: -", style="Header.TLabel")
        self.cd_token_label.grid(row=0, column=0, pady=5, sticky="w")

        self.cd_image_label = tk.Label(self.correction_display_frame, text="Loading Image...", anchor=tk.CENTER, background="#DDDDDD", relief="groove", borderwidth=1)
        self.cd_image_label.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.cd_info_text = tk.Text(self.correction_display_frame, wrap=tk.WORD, height=5, state=tk.DISABLED, font=('Consolas', 10))
        self.cd_info_text.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        self.cd_next_trait_button = ttk.Button(self.correction_display_frame, text="Skip to Next Trait Type / Finish This Trait",
                                                command=self._finish_current_trait_correction_or_proceed)
        self.cd_next_trait_button.grid(row=3, column=0, pady=10)

    def _setup_complete_widgets(self):
        """
        Sets up widgets for
        the completion screen.
        """
        self.complete_label = ttk.Label(self.complete_frame, text="Verification Complete!", style="Header.TLabel")
        self.complete_label.pack(pady=(20, 10))
        self.complete_info_label = \
            ttk.Label(self.complete_frame, text="", style="Status.TLabel", justify=tk.CENTER)
        self.complete_info_label.pack(pady=5)

        self.complete_button_frame = ttk.Frame(self.complete_frame)
        self.complete_button_frame.pack(pady=20)

        self.open_logs_button = \
            ttk.Button(self.complete_button_frame,
                        text="Open Log Files", command=lambda: self._open_logs(open_main_log=True), style="TButton", width=20)
        self.final_action_button = \
            ttk.Button(self.complete_button_frame, text="Close Verifier", command=self.master.quit, style="Close.TButton", width=20)
        self.run_metagen_button = ttk.Button(self.complete_button_frame, text="Run Metagen Now", command=self._run_metagen, style="Verify.TButton", width=20)

    def _on_resize_correction_image(self, _event: Optional[Event] = None):
        """
        Handles window resize events to rescale
        the displayed image proportionally
        within the cd_image_label widget.
        Args:
            event: The Tkinter configure event
            object (passed automatically).
        """
        if not self.current_pil_image_for_correction:
            return

        try:
            widget_width = self.cd_image_label.winfo_width()
            widget_height = self.cd_image_label.winfo_height()

            pad_x = 10
            pad_y = 10

            target_width = max(1, widget_width - (2 * pad_x))
            target_height = max(1, widget_height - (2 * pad_y))

            if target_width <= 1 or target_height <= 1:
                return None
            img_width, img_height = self.current_pil_image_for_correction.size
            if img_height == 0:
                logger.warning("Cannot resize image with zero height.")
                return
            img_ratio = img_width / img_height

            widget_ratio = target_width / target_height

            if widget_ratio > img_ratio:
                new_height = target_height
                new_width = int(new_height * img_ratio)
            else:
                new_width = target_width
                new_height = int(new_width / img_ratio)

            new_width = max(1, new_width)
            new_height = max(1, new_height)

            resized_image = self.current_pil_image_for_correction.resize(
                (new_width, new_height), Image.Resampling.LANCZOS
            )

            self.cd_img_tk = ImageTk.PhotoImage(resized_image)
            self.cd_image_label.config(image=self.cd_img_tk)

        except (ValueError, MemoryError) as pil_error:
            logger.error("PIL error during image resize: %s", pil_error)
            self.cd_image_label.config(image='')
        except (tk.TclError, RuntimeError) as e:
            logger.error("Error during image resize: %s", e, exc_info=True)
            self.cd_image_label.config(image='')

    def _start_full_process(self):
        """
        Orchestrates the new workflow:
        parse logs, allow corrections,
        retrain.
        """
        logger.info("Starting full verification and retraining process.")
        self.analyze_button.config(state=tk.DISABLED)
        self.initial_screen_frame.pack_forget()

        self.low_confidence_data = parse_low_confidence_skips(TRAIT_PREDICTOR_LOG_FILE)
        self.generate_skip_analysis_report_local(self.low_confidence_data, ANALYSIS_LOG_FILE)

        if not self.low_confidence_data:
            messagebox.showinfo("No Skips Found", "No low-confidence skips were found in the log file. Nothing to verify or retrain.")
            self._show_completion_screen(message="No low-confidence skips found.")
            return

        self.newly_corrected_data = {}
        self.current_trait_for_correction_queue = sorted(list(self.low_confidence_data.keys()))

        summary_message = "Low-confidence skips found for:\n"
        for tt, items in self.low_confidence_data.items():
            summary_message += f"- {tt}: {len(items)} skips\n"
        summary_message += "\nDo you want to proceed with manual correction for a sample of these?"

        if not messagebox.askyesno("Proceed with Correction?", summary_message, parent=self.master):
            self._show_completion_screen(message="User chose not to proceed with corrections.")
            return

        self.correction_display_frame.pack(fill=tk.BOTH, expand=True)
        self._process_next_trait_type_for_correction()

    def generate_skip_analysis_report_local(self, low_confidence_data: Dict[str, List[Dict[str, Any]]], report_file_path: str):
        """
        Generates and logs an analysis
        report of low-confidence skips.
        """
        logger.info("Generating low-confidence skip analysis report to: {report_file_path}")
        try:
            with open(report_file_path, 'a', encoding='utf-8') as f:
                f.write("\n\n--- Low Confidence Skip Analysis Report ---\n")
                f.write(f"Timestamp: {logging.Formatter('%(asctime)s').formatTime(logging.LogRecord('dummy', 0, '', 0, '', (), None, None), datefmt='%Y-%m-%d %H:%M:%S')}\n")
                if not low_confidence_data:
                    f.write("No low-confidence skips found in the log.\n")
                    logger.info("No low-confidence skips to report.")
                    return

                total_skips = sum(len(items) for items in low_confidence_data.values())
                f.write(f"Total low-confidence skips found: {total_skips}\n\n")

                for trait_type, items in sorted(low_confidence_data.items()):
                    f.write(f"Trait Type: '{trait_type}' - {len(items)} low-confidence skips\n")
                f.write("\n--- End of Report ---\n")
            logger.info("Successfully generated skip analysis report.")
        except Exception:
            logger.error("Failed to generate skip analysis report: %d", exc_info=True)


    def _process_next_trait_type_for_correction(self):
        """
        Moves to the next trait type
        in the queue for user correction.
        """
        if not self.current_trait_for_correction_queue:
            logger.info("All trait types processed for correction.")
            self._start_retraining_phase()
            return

        self.current_trait_being_corrected = self.current_trait_for_correction_queue.pop(0)
        items_to_correct = self.low_confidence_data.get(self.current_trait_being_corrected, [])

        if not items_to_correct:
            logger.info("No items to correct for trait '{self.current_trait_being_corrected}'. Skipping.")
            self.master.after(10, self._process_next_trait_type_for_correction)
            return

        sample_size = max(1, int(len(items_to_correct) * 0.10))
        self.current_items_for_trait_queue = random.sample(items_to_correct, min(sample_size, len(items_to_correct)))

        logger.info("Starting correction for trait '{self.current_trait_being_corrected}'. Will show {len(self.current_items_for_trait_queue)} items.")
        self._display_next_correction_item()

    def _display_next_correction_item(self):
        """
        Displays the next item
        (image and low-confidence
        prediction) for correction.
        """
        if not self.current_items_for_trait_queue:
            logger.info("Finished correcting items for trait '{self.current_trait_being_corrected}'.")
            self.master.after(10, self._process_next_trait_type_for_correction)
            return

        item_data = self.current_items_for_trait_queue.pop(0)
        token_id = item_data["token_id"]
        predicted_value = item_data["predicted_value"]

        trait_to_correct = self.current_trait_being_corrected
        if trait_to_correct is None:
            logger.error("In _display_next_correction_item, current_trait_being_corrected is None. Skipping.")
            self.master.after(10, self._display_next_correction_item)
            return

        if self.image_map is None:
            logger.error("Image map not loaded, cannot display item for correction.")
            messagebox.showerror("Error", "Image map not loaded.", parent=self.master)
            self._finish_current_trait_correction_or_proceed()
            return

        self.cd_token_label.config(text=f"Token ID: {token_id} - Correcting Trait: {trait_to_correct}")

        info_str = (f"Trait: {trait_to_correct}\n"
                    f"Low-Confidence Prediction: '{predicted_value}'\n"
                    f"Remaining for this trait: {len(self.current_items_for_trait_queue)}")
        self.cd_info_text.config(state=tk.NORMAL)
        self.cd_info_text.delete(1.0, tk.END)
        self.cd_info_text.insert(tk.END, info_str)
        self.cd_info_text.config(state=tk.DISABLED)

        self.current_pil_image_for_correction = load_local_image_original(token_id, self.image_map)

        if self.current_pil_image_for_correction:
            self.cd_image_label.config(text="")
            self._on_resize_correction_image()
        else:
            self.cd_image_label.config(image='', text=f"Image for token {token_id} unavailable.")
            if not messagebox.askyesno("Image Unavailable", f"Image for token {token_id} could not be loaded. Still attempt to correct this trait?", parent=self.master):
                self.master.after(10, self._display_next_correction_item)
                return

        dialog = CorrectionDialog(self.master, token_id,
                                    self.current_pil_image_for_correction if self.current_pil_image_for_correction else Image.new("RGB", (100,100), "gray"),
                                    trait_to_correct,
                                    predicted_value)
        corrected_value = dialog.wait_for_input()

        if corrected_value is not None and corrected_value.lower() != "none":
            if self.current_pil_image_for_correction:
                img_vector = image_to_vector(self.current_pil_image_for_correction.copy())
                img_hist = image_to_color_histogram(self.current_pil_image_for_correction.copy())
                features = get_feature_extractor_and_data(trait_to_correct, img_vector, img_hist)

                if features is not None and features.size > 0:
                    if trait_to_correct not in self.newly_corrected_data:
                        self.newly_corrected_data[trait_to_correct] = []
                    self.newly_corrected_data[trait_to_correct].append((features, corrected_value))
                    logger.info("Stored new data for token {token_id}, trait '{trait_to_correct}': '{corrected_value}'")
                else:
                    logger.warning("Could not extract features for token {token_id}. Correction not stored.")
            else:
                logger.warning("No image for token {token_id}. Correction for '{trait_to_correct}' not stored with features.")

        self.master.after(10, self._display_next_correction_item)

    def _finish_current_trait_correction_or_proceed(self):
        """
        Clears current trait queue and moves
        to the next trait type or retraining."
        """
        logger.info("User chose to skip remaining items for trait '{self.current_trait_being_corrected}'.")
        self.current_items_for_trait_queue = []
        self.master.after(10, self._process_next_trait_type_for_correction)

    def _start_retraining_phase(self):
        """
        Initiates the classifier retraining
        process for traits with new data.
        """
        self.correction_display_frame.pack_forget()
        self.complete_frame.pack(fill=tk.BOTH, expand=True)
        self.complete_label.config(text="Retraining Classifiers...")
        self.complete_info_label.config(text="Please wait...")
        self.master.update_idletasks()

        self.retraining_report = []

        if not self.newly_corrected_data:
            logger.info("No new data collected for retraining.")
            self.retraining_report.append("No new data was collected, so no classifiers were retrained.")
            self._show_completion_screen(message="No new data collected for retraining.")
            return

        logger.info("Starting retraining for {len(self.newly_corrected_data)} trait types.")

        original_training_data_collection = {}
        try:
            if os.path.exists(COLLECTED_DATA_FILE_PATH):
                with gzip.open(COLLECTED_DATA_FILE_PATH, "rb") as f:
                    loaded_checkpoint = pickle.load(f)
                    original_training_data_collection = loaded_checkpoint.get('training_data', {})
                logger.info("Loaded original training data from {COLLECTED_DATA_FILE_PATH}")
            else:
                logger.warning("Original training data file {COLLECTED_DATA_FILE_PATH} not found.")
        except Exception:
            logger.error ("Error loading original training data: %d", exc_info=True)

        traits_to_retrain_sorted = sorted(list(self.newly_corrected_data.keys()))
        total_to_retrain = len(traits_to_retrain_sorted)

        for i, trait_type in enumerate(traits_to_retrain_sorted):
            self.complete_info_label.config(text=f"Retraining '{trait_type}' ({i+1}/{total_to_retrain})...")
            self.master.update_idletasks()

            new_points_for_trait = self.newly_corrected_data[trait_type]
            original_points_for_trait = original_training_data_collection.get(trait_type, [])
            combined_data_points = original_points_for_trait + new_points_for_trait

            if not combined_data_points or len(combined_data_points) < 2:
                logger.warning("Insufficient data for trait '{trait_type}'. Skipping retraining.")
                self.retraining_report.append(f"Trait '{trait_type}': Skipped (insufficient data).")
                continue

            features_list = [dp[0] for dp in combined_data_points]
            labels_list = [dp[1] for dp in combined_data_points]

            if len(set(labels_list)) < 2:
                logger.warning("Insufficient class variety for trait '{trait_type}'. Skipping retraining.")
                self.retraining_report.append(f"Trait '{trait_type}': Skipped (insufficient class variety).")
                continue

            feature_array = np.array(features_list)

            pipeline_steps: List[Tuple[str, Any]] = [('scaler', StandardScaler())]
            n_samples, n_features = feature_array.shape[0], feature_array.shape[1]
            max_pca_components = min(n_samples - 1, n_features)

            if trait_type != "Background" and max_pca_components > 1:
                pca_n_components = min(50, max_pca_components)
                pipeline_steps.append(('pca', PCA(n_components=pca_n_components, svd_solver='randomized', random_state=42)))

            pipeline_steps.append(('svm', SVC(probability=True, random_state=42, class_weight='balanced', C=1.0, kernel='rbf', gamma='scale')))
            retrained_pipeline = Pipeline(pipeline_steps)

            try:
                logger.info("Retraining classifier for '{trait_type}' with {len(combined_data_points)} data points.")
                retrained_pipeline.fit(feature_array, labels_list)

                sanitized_trait_filename = trait_type.replace(' ', '_').replace('/', '_')
                model_path = os.path.join(CLASSIFIER_DIR, f"{sanitized_trait_filename}.pkl.gz")
                with gzip.open(model_path, "wb") as f_model:
                    pickle.dump(retrained_pipeline, f_model)
                logger.info("Successfully retrained and saved classifier for '{trait_type}' to {model_path}.")
                self.retraining_report.append(f"Trait '{trait_type}': Retrained with {len(new_points_for_trait)} new points.")
            except Exception as e:
                logger.error("Error retraining classifier for '{trait_type}': {e}", exc_info=True)
                self.retraining_report.append(f"Trait '{trait_type}': FAILED retraining ({e}).")

        self._show_completion_screen(message="Classifier retraining phase complete.")

    def _show_completion_screen(self, message: str = "Process Complete."):
        """
        Displays the final completion
        screen and prompts to run metagen.
        """
        self.correction_display_frame.pack_forget()
        self.initial_screen_frame.pack_forget()
        self.complete_frame.pack(fill=tk.BOTH, expand=True)

        self.complete_label.config(text="Verification & Retraining Complete")

        final_summary = message + "\n\nRetraining Summary:\n" + "\n".join(self.retraining_report)
        self.complete_info_label.config(text=final_summary)

        for widget in self.complete_button_frame.winfo_children():
            widget.destroy()

        self.open_logs_button.pack(side=tk.LEFT, padx=10, ipady=5)
        self.run_metagen_button.pack(side=tk.LEFT, padx=10, ipady=5)
        self.final_action_button.pack(side=tk.LEFT, padx=10, ipady=5)

        logger.info("Verifier process finished. Displaying completion screen.")

    def _run_metagen(self):
        """
        Attempts to run metagen.py as a subprocess.
        """
        metagen_script_path = os.path.join(BASE_DIR, "metagen.py")
        if not os.path.exists(metagen_script_path):
            messagebox.showerror("Error", f"metagen.py not found at {metagen_script_path}", parent=self.master)
            logger.error("metagen.py not found at {metagen_script_path}")
            return

        try:
            logger.info("Attempting to launch metagen.py...")
            subprocess.Popen([sys.executable, metagen_script_path])
            messagebox.showinfo("Metagen Launched", "metagen.py has been launched.\nThis verifier application will now close.", parent=self.master)
            self.master.quit()
        except Exception as e:
            messagebox.showerror("Error Launching Metagen", f"Could not launch metagen.py: {e}", parent=self.master)
            logger.error("Failed to launch metagen.py: {e}", exc_info=True)

    def _open_logs(self, open_main_log: bool = True):
        """
        Opens the relevant log files
        using the default system application.
        Args:
            open_main_log: Whether to open
            the main verification log.
        """
        files_to_open = []
        if open_main_log and os.path.exists(MAIN_LOG_FILE):
            files_to_open.append(MAIN_LOG_FILE)
        if os.path.exists(ANALYSIS_LOG_FILE):
            files_to_open.append(ANALYSIS_LOG_FILE)

        opened_count = 0
        if not files_to_open:
            messagebox.showinfo("No Logs Found",
                                "Could not find any relevant log files to open.")
            return

        for log_file in files_to_open:
            open_file(log_file)
            opened_count += 1

        if opened_count == 0:
            messagebox.showinfo("No Logs Opened",
                                "Could not open any log files.")

if __name__ == "__main__":
    root = tk.Tk()
    app = MetadataVerifierApp(root)
    if hasattr(app, 'image_map') and app.image_map is not None:
        root.mainloop()
        logger.info("Application closed normally.")
    else:
        logger.info("Application did not start due to initialization errors.")
        try:
            root.destroy()
        except tk.TclError:
            pass
