#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0301
# pylint: disable=C0302
# pylint: disable=W0718
"""
Part 2 - Trait Prediction & Metadata Generation for Unrevealed Tokens (Improved)
--------------------------------------------------------------------------------
This script uses pretrained classifier pipelines from Part 1 (Improved) to analyze
remaining unrevealed NFT images, predict their traits, and generate corresponding
metadata files. It performs the following tasks:
1. Loads pre-trained classifier pipelines from the specified directory.
2. Fetches images based on image mappings in `images.json` (handling various formats).
3. Extracts appropriate features (vector or histogram) and predicts traits using pipelines.
4. Calculates prediction confidence and optionally filters low-confidence traits (adjustable).
5. Generates standardized metadata files per token ID, for the remaining unfinished Tokens.
6. Saves output to `./metadata/` (no .json extension).
7. Also saves a combined metadata file `metadata.json` in the same directory.
8. Entire process is complete with checkpoints and logging.
Free to use, no API keys or external paid services required.
"""
import gzip
import json
import logging
import os
import pickle
import sys
from time import sleep
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

BASE_DIR = "."
LOG_DIR = os.path.join(BASE_DIR, "logs")
CONFIG_DIR = os.path.join(BASE_DIR, "config")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
CLASSIFIER_DIR = os.path.join(CONFIG_DIR, "classifiers")
IMAGE_JSON_PATH = os.path.join(CONFIG_DIR, "images.json")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "part2_checkpoint.json")
SCHEMA_FILE_PATH = os.path.join(CONFIG_DIR, "schema.json")
LOG_FILE = os.path.join(LOG_DIR, "trait_predictor_v2.log")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(CLASSIFIER_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

IMAGE_SIZE = (128, 128)
HISTOGRAM_BINS = 16
TOKEN_START_PREDICT = 1295
TOKEN_END_PREDICT = 10000
PREDICTION_CONFIDENCE_THRESHOLD = 0.70
EXCLUDE_LOW_CONFIDENCE_TRAITS = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

def load_trait_schema(file_path: str) -> List[str]:
    """
    Loads the trait order schema from a JSON file.
    Args:
        file_path: Path to the schema.json file.
    Returns:
        A list of trait types in the specified order, or an empty list on error.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        if not isinstance(schema, list) or not all(isinstance(item, str) for item in schema):
            logger.error("Schema file %s is not a valid list of strings.", file_path)
            return []
        logger.info("Successfully loaded trait schema with %d traits from %s.", len(schema), file_path)
        return schema
    except FileNotFoundError:
        logger.error("Trait schema file not found: %s", file_path)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Error reading or parsing trait schema file %s: %s", file_path, e)
    return []

def load_pickle_pipeline(file_path: str) -> Optional[Pipeline]:
    """
    Load a gzip-compressed
    pickle object (sklearn
    Pipeline) from file.
    Includes basic validation checks.
    """
    logger.debug("Attempting to load pipeline from: %s", file_path)
    pipeline: Optional[Pipeline] = None
    try:
        with gzip.open(file_path, 'rb') as f:
            pipeline = pickle.load(f)

        if not isinstance(pipeline, Pipeline) or not hasattr(pipeline, 'predict_proba'):
            logger.error("Loaded object from %s is not a valid scikit-learn Pipeline with predict_proba.", file_path)
            return None

        try:
            _ = getattr(pipeline[-1], "classes_", None)

            if hasattr(pipeline.steps[0][1], 'n_features_in_') and pipeline.steps[0][1].n_features_in_ > 0:
                dummy_data = np.zeros((1, pipeline.steps[0][1].n_features_in_))
                pipeline.predict_proba(dummy_data)
            elif hasattr(pipeline.steps[-1], 'n_features_in_') and pipeline.steps[-1].n_features_in_ > 0:

                dummy_data = np.zeros((1, pipeline.steps[-1].n_features_in_))

                logger.warning("Could not fully validate pipeline dimensions for %s, relying on runtime checks.", file_path)
            else:
                logger.warning("Could not determine expected feature dimension for %s during loading validation.", file_path)

            logger.debug("Pipeline loaded and passed initial validation: %s", file_path)
            return pipeline

        except NotFittedError:
            logger.error("Loaded pipeline from %s is not fitted.", file_path)
            return None
        except AttributeError as ae:
            logger.warning("Attribute error during pipeline validation for %s: %s. Proceeding cautiously.", file_path, ae)
            return pipeline
        except ValueError as ve:
            logger.warning("ValueError during pipeline validation prediction for %s: %s. Proceeding cautiously.", file_path, ve)
            return pipeline

    except FileNotFoundError:
        logger.error("Pipeline file not found at %s.", file_path)
        return None
    except (pickle.PickleError, EOFError, gzip.BadGzipFile, OSError, ValueError) as e:
        logger.error("Failed to load or validate pipeline from %s: %s", file_path, e)
        return None
    except RuntimeError as e:
        logger.error("Unexpected error loading pipeline from %s: %s", file_path, e, exc_info=True)
        return None

def load_pipelines() -> Dict[str, Optional[Pipeline]]:
    """
    Load all classifier pipelines
    from the specified CLASSIFIER_DIR.
    Determines trait types based on
    filenames. Returns a dictionary mapping
    trait type (string) to the loaded
    Pipeline object or None if loading failed.
    """
    pipelines: Dict[str, Optional[Pipeline]] = {}
    trait_types: List[str] = []

    if not os.path.isdir(CLASSIFIER_DIR):
        logger.error("Classifier directory not found: %s. Cannot load models.", CLASSIFIER_DIR)
        return {}

    logger.info("Scanning for classifier models in: %s", CLASSIFIER_DIR)
    try:
        model_files = [f for f in os.listdir(CLASSIFIER_DIR) if f.endswith('.pkl.gz')]
        trait_types = sorted([f.replace('.pkl.gz', '').replace('_', ' ') \
            for f in model_files])
    except OSError as e:
        logger.error("Error reading classifier directory %s: %s", CLASSIFIER_DIR, e)
        return {}

    if not trait_types:
        logger.error("No classifier files (.pkl.gz) found in %s.", CLASSIFIER_DIR)
        return {}

    logger.info("Found %d potential trait types: %s", len(trait_types), ", ".join(trait_types))

    loaded_count = 0
    for trait in trait_types:
        sanitized_trait = trait.replace(' ', '_').replace('/', '_')
        model_path = os.path.join(CLASSIFIER_DIR, f"{sanitized_trait}.pkl.gz")

        logger.info("Attempting to load pipeline for trait '%s'...", trait)
        pipeline = load_pickle_pipeline(model_path)
        pipelines[trait] = pipeline
        if pipeline is not None:
            logger.info("Successfully loaded pipeline for trait '%s'.", trait)
            loaded_count += 1
        else:
            logger.error("Failed to load pipeline for trait '%s' from %s.", trait, model_path)

    logger.info("Finished loading pipelines. Successfully loaded %d out of %d.", loaded_count, len(trait_types))
    if loaded_count == 0:
        logger.critical("CRITICAL: No pipelines were loaded successfully. Cannot proceed.")
    return pipelines

def get_checkpoint() -> int:
    """
    Return last processed
    token ID from
    checkpoint file.
    """ # Fix: Dedent the code block
    default_start_token = TOKEN_START_PREDICT - 1
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                last_processed = data.get("last_processed", default_start_token)
                if not isinstance(last_processed, int) or last_processed < default_start_token:
                    logger.warning("Invalid checkpoint value '%s' found. Resetting to %d.", last_processed, default_start_token)
                    return default_start_token

                logger.info("Checkpoint found. Last processed token: %d", last_processed)
                return last_processed

        except (json.JSONDecodeError, OSError) as e:
            logger.error("Error reading checkpoint file %s: %s. Starting from %d.", CHECKPOINT_FILE, e, default_start_token)
            return default_start_token

        except RuntimeError as e:
            logger.error("Unexpected error reading checkpoint file %s: %s. Starting from %d.", CHECKPOINT_FILE, e, default_start_token, exc_info=True)
        return default_start_token

    logger.info("No checkpoint file found. Starting from token %d.", default_start_token + 1)
    return default_start_token

def update_checkpoint(token_id: int):
    """
    Save the last successfully
    processed token ID to
    the checkpoint file.
    """
    try:
        os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
        with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump({"last_processed": token_id}, f, indent=2)
        logger.debug("Checkpoint updated to token %d", token_id)
    except OSError as e:
        logger.error("Failed to update checkpoint file %s: %s", CHECKPOINT_FILE, e)
    except RuntimeError as e:
        logger.error("Unexpected error updating checkpoint file %s: %s", CHECKPOINT_FILE, e, exc_info=True)

def load_image(filename: str) -> Optional[Image.Image]:
    """
    Loads an image from the local IMAGES_DIR.
    Returns the image resized to
    IMAGE_SIZE. Handles errors.
    Args:
        filename: The name of the image file (e.g., "998.jpeg").
    Returns:
        A PIL Image object resized to IMAGE_SIZE, or None if loading fails.
    """
    if not filename or not isinstance(filename, str):
        logger.error("Invalid filename provided to load_image: %s", filename)
        return None

    local_image_path = os.path.join(IMAGES_DIR, filename)
    logger.debug("Attempting to load image from local path: %s", local_image_path)

    img: Optional[Image.Image] = None
    try:
        if not os.path.exists(local_image_path):
            logger.error("Local image file not found: %s", local_image_path)
            return None

        img = Image.open(local_image_path).convert("RGB")

        if img.size != IMAGE_SIZE:
            logger.debug("Resizing image from %s to %s", img.size, IMAGE_SIZE)
            img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

        logger.debug("Successfully loaded and processed image from %s", local_image_path)
        return img

    except FileNotFoundError:
        logger.error("Local image file not found (FileNotFoundError): %s", local_image_path)
        return None
    except UnidentifiedImageError:
        logger.error("Failed to load image from %s: Cannot identify image file (PIL)", local_image_path)
        return None
    except (OSError, IOError, ValueError) as e:
        logger.error("Failed processing image from %s: %s", local_image_path, str(e))
        return None
    except RuntimeError as e:
        logger.error("Unexpected error loading image from %s: %s", local_image_path, e, exc_info=True)
        return None

def image_to_vector(image: Image.Image) -> Optional[np.ndarray]:
    """
    Converts an image (assumed to be correct IMAGE_SIZE)
    into a flattened numpy array vector. Returns None on failure.
    """
    if image is None:
        logger.error("image_to_vector received None image.")
        return None
    try:
        if image.size != IMAGE_SIZE:
            logger.warning("Image provided to image_to_vector is not the expected size %s. Resizing.", IMAGE_SIZE)
            image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        vector = np.asarray(image).flatten()
        logger.debug("Image converted to vector shape: %s", vector.shape)
        return vector
    except RuntimeError as e:
        logger.error("Failed to convert image to vector: %s", e, exc_info=True)
        return None

def image_to_color_histogram(image: Image.Image, bins: int = HISTOGRAM_BINS) -> Optional[np.ndarray]:
    """
    Calculates a flattened, normalized
    3D color histogram for an RGB image
    (assumed to be correct IMAGE_SIZE).
    Returns None on failure.
    """
    if image is None:
        logger.error("image_to_color_histogram received None image.")
        return None

    try:
        if image.size != IMAGE_SIZE:
            logger.warning("Image provided to image_to_color_histogram is not the expected size %s. Resizing.", IMAGE_SIZE)
            image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

        img_array = np.asarray(image)
        hist_r = np.histogram(img_array[:, :, 0], bins=bins, range=(0, 256))[0]
        hist_g = np.histogram(img_array[:, :, 1], bins=bins, range=(0, 256))[0]
        hist_b = np.histogram(img_array[:, :, 2], bins=bins, range=(0, 256))[0]

        total_pixels = image.width * image.height
        if total_pixels == 0:
            logger.error("Cannot calculate histogram for zero-pixel image.")
            return None

        norm_factor = float(total_pixels)
        hist_r = hist_r / norm_factor
        hist_g = hist_g / norm_factor
        hist_b = hist_b / norm_factor

        histogram_vector = np.concatenate((hist_r, hist_g, hist_b))
        logger.debug("Image converted to histogram shape: %s", histogram_vector.shape)
        return histogram_vector
    except AttributeError as e:
        logger.error("Failed to calculate color histogram: %s", e, exc_info=True)
        return None

def predict_trait_with_confidence(
    pipeline: Pipeline,
    features: Optional[np.ndarray],
    feature_type: str,
    token_id: int,
    trait_type: str
) -> Tuple[Optional[Any], Optional[float]]:
    """
    Predicts a trait using the pipeline
    and returns the prediction and its confidence.
    Args:
        pipeline: The scikit-learn pipeline for the trait.
        features: The input feature array (vector or histogram).
        feature_type: A string indicating the type
        of features ('vector' or 'histogram').
        token_id: The token ID being processed (for logging).
        trait_type: The name of the trait being predicted (for logging).
    Returns:
        A tuple (predicted_value, confidence_score).
        Returns (None, None) if prediction fails.
        Confidence score is None if predict_proba fails but predict succeeds.
    """
    if features is None:
        logger.warning("Skipping trait '%s' for token %d: Required %s features are None.", trait_type, token_id, feature_type)
        return None, None

    try:
        features_reshaped = features.reshape(1, -1)
        logger.debug("Predicting trait '%s' using %s features (shape: %s)...", trait_type, feature_type, features_reshaped.shape)

        probabilities = pipeline.predict_proba(features_reshaped)[0]
        predicted_index = np.argmax(probabilities)
        predicted_value = pipeline.classes_[predicted_index]
        confidence = probabilities[predicted_index]

        logger.info("Token %d, Trait '%s': Predicted -> '%s' (Confidence: %.4f)", token_id, trait_type, predicted_value, confidence)
        return predicted_value, confidence

    except NotFittedError:
        logger.error("CRITICAL: Pipeline for trait '%s' is not fitted! Skipping prediction for token %d.", trait_type, token_id)
        return None, None
    except ValueError as ve:
        expected_features = "N/A"
        try:
            if hasattr(pipeline.steps[0][1], 'n_features_in_'):
                expected_features = pipeline.steps[0][1].n_features_in_
            elif hasattr(pipeline.steps[-1], 'n_features_in_'):
                expected_features = pipeline.steps[-1].n_features_in_
        except (AttributeError, IndexError, TypeError):
            pass

        logger.error("Prediction failed for trait '%s', token %d: %s. Check feature dimensions (Expected: %s, Got shape: %s).", trait_type, token_id, ve, expected_features, str(features.shape if features is not None else "N/A"))
        return None, None
    except AttributeError as ae:
        logger.error("AttributeError during prediction for trait '%s', token %d: %s. Does pipeline support predict_proba?", trait_type, token_id, ae)
        return None, None
    except KeyError as e:
        logger.error("Unexpected error predicting trait '%s' for token %d: %s", trait_type, token_id, e, exc_info=True)
        return None, None

def generate_metadata(token_id: int, predicted_traits_map: Dict[str, Any], image_filename_in_metadata: str, schema_trait_order: List[str]) -> Dict[str, Any]:
    """
    Create the metadata JSON structure for a token, ordering traits correctly.
    Ensures all main traits from schema_trait_order are present.
    Args:
        token_id: The ID of the token.
        traits: A dictionary of predicted trait_type: value pairs. Values can be None.
        image_filename_in_metadata: The filename (e.g., "998.jpeg") for the token's image
                               to be stored *inside* the metadata file.
        trait_order: A list of trait types in the desired output order.
    Returns:
        A dictionary representing the token's metadata.
    """
    attributes = []
    one_of_one_trait_value = predicted_traits_map.get("One Of One")
    special_trait_value = predicted_traits_map.get("Special")

    is_one_of_one = one_of_one_trait_value is not None and str(one_of_one_trait_value).strip().lower() not in ["none", ""]
    is_special = special_trait_value is not None and str(special_trait_value).strip().lower() not in ["none", ""]

    if is_one_of_one:
        logger.info("Token %d identified as One Of One: '%s'", token_id, one_of_one_trait_value)
        attributes = [{"trait_type": "One Of One", "value": str(one_of_one_trait_value)}]
    elif is_special:
        logger.info("Token %d identified as Special: '%s'", token_id, special_trait_value)
        attributes = [{"trait_type": "Special", "value": str(special_trait_value)}]
    else:
        logger.debug("Generating standard attributes for token %d based on schema order: %s", token_id, schema_trait_order)
        for trait_type_from_schema in schema_trait_order:
            if trait_type_from_schema in ["One Of One", "Special"]: # These are handled above
                continue

            # Get the predicted value. It might be None if it was low confidence and excluded,
            # or if no model existed for this schema trait.
            predicted_value = predicted_traits_map.get(trait_type_from_schema)
            output_value: str

            if predicted_value is None or str(predicted_value).strip().lower() == "none" or str(predicted_value).strip() == "":
                output_value = "None"
            else:
                output_value = str(predicted_value)

            attributes.append({"trait_type": trait_type_from_schema, "value": output_value})
            if output_value == "None" and predicted_value is not None and str(predicted_value).strip().lower() != "none":
                logger.debug("Trait '%s' for token %d included as 'None' (original prediction was '%s' but might have been empty or 'none').", trait_type_from_schema, token_id, predicted_value)
            elif output_value == "None":
                logger.debug("Trait '%s' for token %d included as 'None' (no prediction or explicitly None).", trait_type_from_schema, token_id)

    if not attributes:
        logger.warning("No valid traits included for token %d. Metadata will have empty attributes list.", token_id)

    metadata = {
        "name": f"Fren #{token_id}",
        "description": "Frens staying Low Key",
        "attributes": attributes,
        "image": image_filename_in_metadata,
        "animation_url": "",
        "background_color": "",
        "youtube_url": "",
        "external_url": "https://lowkey.fun",
        "seller_fee_basis_points": 500,
        "fee_recipient": "0xb6FFA8792476B0188811137536090da3cfEA70B0"
    }

    logger.debug("Generated metadata structure for token %d", token_id)
    return metadata

def main():
    """
    Loads pipelines, processes tokens,
    predicts traits, and saves metadata.
    """
    logger.info("--- Starting Part 2: Trait Prediction & Metadata Generation (v2) ---")
    logger.info("Confidence Threshold: %s", PREDICTION_CONFIDENCE_THRESHOLD)
    logger.info("Exclude Low Confidence Traits: %s", EXCLUDE_LOW_CONFIDENCE_TRAITS)

    schema_trait_order = load_trait_schema(SCHEMA_FILE_PATH)
    if not schema_trait_order:
        logger.critical("CRITICAL: Trait schema could not be loaded from %s. Cannot ensure correct trait order or inclusion. Exiting.", SCHEMA_FILE_PATH)
        sys.exit(1)

    pipelines = load_pipelines()
    if not pipelines or all(p is None for p in pipelines.values()):
        logger.critical("CRITICAL: No valid classifier pipelines were loaded. Cannot proceed. Ensure Part 1 ran successfully and models exist in %s.", CLASSIFIER_DIR)
        sys.exit(1)

    # These are the traits for which we have successfully loaded models
    available_model_traits = sorted([trait for trait, p in pipelines.items() if p is not None])
    if not available_model_traits:
        logger.critical("CRITICAL: No pipelines loaded successfully. Cannot proceed.")
        sys.exit(1)

    missing_model_for_schema_traits = sorted(list(set(schema_trait_order) - set(available_model_traits) - {"One Of One", "Special"}))
    if missing_model_for_schema_traits:
        logger.warning("Models for the following schema traits could not be loaded (will output as 'None' if main traits): %s", ", ".join(missing_model_for_schema_traits))
    logger.info("Will attempt to predict for traits with loaded models: %s", ", ".join(available_model_traits))

    try:
        logger.info("Loading image map from %s", IMAGE_JSON_PATH)
        with open(IMAGE_JSON_PATH, 'r', encoding='utf-8') as f:
            image_map = json.load(f)
        logger.info("Image map loaded successfully (%d entries).", len(image_map))
    except FileNotFoundError:
        logger.critical("CRITICAL: Image map file not found: %s. Cannot proceed.", IMAGE_JSON_PATH)
        sys.exit(1)
    except (json.JSONDecodeError, OSError) as e:
        logger.critical("CRITICAL: Error reading image map file %s: %s. Cannot proceed.", IMAGE_JSON_PATH, e)
        sys.exit(1)
    except RuntimeError as e:
        logger.critical("CRITICAL: Unexpected error loading image map %s: %s. Cannot proceed.", IMAGE_JSON_PATH, e, exc_info=True)
        sys.exit(1)

    if not isinstance(image_map, dict):
        logger.critical("CRITICAL: Image map loaded from %s is not a dictionary. Cannot proceed.", IMAGE_JSON_PATH)
        sys.exit(1)
    if not image_map:
        logger.critical("CRITICAL: Image map loaded from %s is empty. Cannot proceed.", IMAGE_JSON_PATH)
        sys.exit(1)

    last_processed_token = get_checkpoint()
    start_token = last_processed_token + 1
    end_token = TOKEN_END_PREDICT
    logger.info("Starting prediction from token ID %d up to %d (inclusive).", start_token, end_token)

    if start_token > end_token:
        logger.info("All tokens up to %d already processed based on checkpoint. Nothing to do.", end_token)
        logger.info("--- Trait prediction script finished ---")
        return

    tokens_processed_this_run = 0
    tokens_failed_this_run = 0
    low_confidence_trait_count = 0

    for token_id in range(start_token, end_token + 1):
        logger.info("----- Processing Token ID: %d -----", token_id)
        image_filename = image_map.get(str(token_id))

        if not image_filename or not isinstance(image_filename, str):
            logger.warning("No valid image filename found for token %d (key '%s') in %s. Skipping.", token_id, str(token_id), IMAGE_JSON_PATH)
            tokens_failed_this_run += 1
            continue

        logger.debug("Image filename from map (for metadata and loading): %s", image_filename)

        try:
            image = load_image(image_filename)
            if image is None:
                logger.warning("Skipping token %d due to image loading failure.", token_id)
                tokens_failed_this_run += 1
                sleep(0.5)
                continue

            vector_features = None
            histogram_features = None
            needs_vector = any(t != "Background" for t in available_model_traits)
            needs_histogram = "Background" in available_model_traits

            if needs_vector:
                vector_features = image_to_vector(image)
                if vector_features is None:
                    logger.warning("Vector feature extraction failed for token %d.", token_id)

            if needs_histogram:
                histogram_features = image_to_color_histogram(image)
                if histogram_features is None:
                    logger.warning("Histogram feature extraction failed for token %d.", token_id)

            if vector_features is None and histogram_features is None and (needs_vector or needs_histogram):
                logger.warning("Skipping token %d due to all required feature extraction failing.", token_id)
                tokens_failed_this_run += 1
                continue

            predicted_traits: Dict[str, Any] = {}
            prediction_successful_for_any_trait = False

            for trait_type in available_model_traits: # Predict only for traits where models exist
                pipeline_optional = pipelines[trait_type]
                if pipeline_optional is None:
                    logger.error("Logic error: Pipeline for validated trait '%s' is None. Skipping.", trait_type)
                    continue

                pipeline = pipeline_optional
                features: Optional[np.ndarray] = None
                feature_type: str = ""

                if trait_type == "Background":
                    features = histogram_features
                    feature_type = "histogram"
                else:
                    features = vector_features
                    feature_type = "vector"

                predicted_value, confidence = predict_trait_with_confidence(pipeline, features, feature_type, token_id, trait_type)

                if predicted_value is not None:
                    prediction_successful_for_any_trait = True
                    if confidence is not None and confidence >= PREDICTION_CONFIDENCE_THRESHOLD:
                        predicted_traits[trait_type] = predicted_value
                    elif confidence is None:
                        logger.warning("Token %d, Trait '%s': Prediction '%s' obtained, but confidence score is missing.", token_id, trait_type, predicted_value)
                        predicted_traits[trait_type] = predicted_value
                    else:
                        low_confidence_trait_count += 1
                        logger.warning("Token %d, Trait '%s': Prediction '%s' has low confidence (%.4f < %.4f).", token_id, trait_type, predicted_value, confidence, PREDICTION_CONFIDENCE_THRESHOLD)
                        if not EXCLUDE_LOW_CONFIDENCE_TRAITS:
                            predicted_traits[trait_type] = predicted_value
                        else:
                            predicted_traits[trait_type] = None

            if not prediction_successful_for_any_trait:
                logger.warning("No traits were successfully predicted for token %d. Skipping metadata generation.", token_id)
                tokens_failed_this_run += 1
                continue

            metadata = generate_metadata(token_id, predicted_traits, image_filename, schema_trait_order)
            output_path = os.path.join(METADATA_DIR, str(token_id))

            try:
                with open(output_path, 'w', encoding='utf-8') as f: # Changed to 'w' to overwrite if re-running
                    json.dump(metadata, f, indent=2)
                logger.info("Successfully generated and saved metadata for token %d to %s", token_id, output_path)
                update_checkpoint(token_id)
                tokens_processed_this_run += 1

            except OSError as e:
                logger.error("Failed to write metadata for token %d to %s: %s", token_id, output_path, e)
                tokens_failed_this_run += 1
            except TypeError as e:
                logger.error("Failed to serialize metadata for token %d: %s", token_id, e)
                tokens_failed_this_run += 1
            except KeyError as e:
                logger.error("Unexpected error saving metadata for token %d: %s", token_id, e, exc_info=True)
                tokens_failed_this_run += 1

        except RuntimeError as e:
            logger.error("Unhandled error processing token %d: %s", token_id, e, exc_info=True)
            tokens_failed_this_run += 1
            sleep(1)

    logger.info("--- Trait prediction script finished ---")
    logger.info("Summary: Processed %d tokens successfully, Failed/Skipped %d tokens in this run.", tokens_processed_this_run, tokens_failed_this_run)
    if low_confidence_trait_count > 0:
        logger.info("Encountered %d low-confidence trait predictions during this run.", low_confidence_trait_count)
        if EXCLUDE_LOW_CONFIDENCE_TRAITS:
            logger.info("Low-confidence traits were excluded from the metadata.")
        else:
            logger.info("Low-confidence traits were included in the metadata (logged warnings).")

if __name__ == "__main__":
    main()
