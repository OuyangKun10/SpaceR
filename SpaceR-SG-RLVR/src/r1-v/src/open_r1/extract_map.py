import os
import json
import jsonlines # Ensure jsonlines is imported if used
import pickle
import pandas as pd
import csv # Ensure csv is imported
import re
import ast
from collections import defaultdict
import logging
from typing import List, Dict, Union, Any, Set, Tuple, TypeAlias # For type hints if used more broadly

# Define type aliases for clarity in docstrings, assuming these structures
Coordinate: TypeAlias = List[int] # Typically [x, y]
CoordinatesList: TypeAlias = List[Coordinate]
ObjectMap: TypeAlias = Dict[str, CoordinatesList] # e.g., {"object_name": [[x1,y1], [x2,y2]]}
ResponseMap: TypeAlias = ObjectMap # For calculate_prediction_score
SolutionMap: TypeAlias = ObjectMap # For calculate_prediction_score


def read_data(file_path: str, file_format: str = None) -> Union[List[Dict[str, Any]], List[Any], None]:
    """
    Reads data from a file based on its format.

    Automatically detects file format from extension if not provided.
    Supports JSON, JSONL, PKL/PICKLE, Parquet, CSV, and TSV.

    Args:
        file_path (str): The path to the data file.
        file_format (str, optional): The format of the file (e.g., 'json', 'jsonl', 'csv').
                                     If None, it's inferred from the file extension. Defaults to None.

    Returns:
        Union[List[Dict[str, Any]], List[Any], None]: A list of records (typically dictionaries)
        read from the file. Returns None if the file is not found, the format is unsupported,
        or a read error occurs. Returns an empty list if the file is empty.
    """
    data = []
    try:
        if file_format is None:
            _, file_extension = os.path.splitext(file_path)
            file_format = file_extension.lstrip('.').lower()

        if file_format == 'json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list): # If the JSON root is not a list
                    try:
                        f.seek(0) # Rewind to try reading line by line
                        data = [json.loads(line) for line in f]
                    except json.JSONDecodeError:
                        # Warning: JSON file content is not a list or a JSON object per line,
                        # attempted to read line by line but parsing failed.
                        print(f"Warning: JSON file content is not a list or a JSON object per line, attempted to read line by line but parsing failed.")
                        return None


        elif file_format == 'jsonl':
            with jsonlines.open(file_path, 'r') as reader:
                data = list(reader)

        elif file_format == 'pkl' or file_format == 'pickle':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                if not isinstance(data, list): # If pickle content is not a list, try to convert
                    try:
                        data = list(data) # Attempt to convert to a list
                    except Exception as e:
                        # Warning: Pickle file content is not a list or iterable,
                        # failed to convert to list.
                        print(f"Warning: Pickle file content is not a list or iterable, failed to convert to list: {e}")
                        return None


        elif file_format == 'parquet':
            df = pd.read_parquet(file_path)
            data = df.to_dict('records')

        elif file_format == 'csv':
            with open(file_path, 'r', encoding='utf-8', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                data = list(reader)

        elif file_format == 'tsv':
            with open(file_path, 'r', encoding='utf-8', newline='') as tsvfile:
                reader = csv.DictReader(tsvfile, delimiter='\t')
                data = list(reader)


        else:
            print(f"Unsupported file format: {file_format}")
            return None

        if not data:
            # Warning: File {file_path} is empty, cannot extract samples.
            print(f"Warning: File {file_path} is empty, cannot extract samples.")
            return []


        return data


    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
# Configure basic logging (INFO level shows fallback triggering)
# Set level=DEBUG to see more detailed logs like coordinate parsing steps and name matching
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(filename)s:%(lineno)d: %(message)s')

# --- Coordinate Parsing and Validation Helpers (Enhanced Robustness) ---

def _parse_and_validate_coord_pair(coord_input: Any, context_name: str) -> Union[List[int], None]:
    """
    Validates if the input is a coordinate pair [x, y] or (x, y) containing
    numeric values or number strings. Handles various messy formats.
    Tries to convert to integers.

    Args:
        coord_input (Any): The input to parse as a coordinate pair.
                           Can be a list, tuple, or string.
        context_name (str): A string describing the context (e.g., object name)
                            for logging purposes.

    Returns:
        Union[List[int], None]: A list [int, int] representing the coordinate pair
                                if successfully parsed and validated, otherwise None.
    """
    logging.debug(f"Attempting to parse as coord pair for '{context_name}': {coord_input} (type: {type(coord_input)})")

    pair_elements = None

    # 1. Handle list/tuple input directly
    if isinstance(coord_input, (list, tuple)):
        if len(coord_input) == 2:
            pair_elements = list(coord_input) # Convert tuple to list for consistency
        elif len(coord_input) == 1:
             # Handle incomplete like [8,] or maybe nested like [[5,0]] passed mistakenly
             single_element = coord_input[0]
             logging.debug(f"Input for '{context_name}' is single-element list/tuple: {coord_input}. Processing element: {single_element}")
             # Try parsing the inner element IF it looks like a pair itself
             if isinstance(single_element, (list, tuple)) and len(single_element) == 2:
                  pair_elements = list(single_element)
                  logging.debug(f"Used inner list/tuple as pair elements: {pair_elements}")
             else:
                  logging.warning(f"Single element '{single_element}' in list/tuple for '{context_name}' is not a pair. Skipping.")
                  return None
        else:
             logging.debug(f"Input list/tuple for '{context_name}' does not have 1 or 2 elements: {coord_input}")
             return None

    # 2. Handle string input - try to extract numbers
    elif isinstance(coord_input, str):
        logging.debug(f"Input for '{context_name}' is a string '{coord_input}', trying to extract numbers.")
        # Find potential numbers (int/float, possibly negative) within the string
        # Example: "<center>5, 4</center>" -> ['5', '4']
        # Example: "6, 5" -> ['6', '5']
        numbers_in_string = re.findall(r"-?\d+(?:\.\d+)?", coord_input)
        if len(numbers_in_string) >= 2:
            # Take the first two numbers found
            pair_elements = numbers_in_string[:2]
            logging.debug(f"Extracted number strings for '{context_name}': {pair_elements} from '{coord_input}'")
        else:
            logging.warning(f"Could not extract at least two numbers from string '{coord_input}' for '{context_name}'. Found: {numbers_in_string}. Skipping.")
            return None

    # 3. If not list/tuple/string, unlikely to be a valid coordinate pair representation
    else:
        logging.debug(f"Input type {type(coord_input)} not directly processable as coord pair for '{context_name}': {coord_input}")
        return None

    # If we didn't get pair_elements (e.g., list/tuple wasn't length 1 or 2)
    if pair_elements is None:
         logging.debug(f"Could not derive pair elements for validation for '{context_name}' from input: {coord_input}")
         return None

    # Now, validate and convert the extracted/provided elements
    coord_pair_numeric = []
    all_numeric_convertible = True
    for n in pair_elements:
        num_val = None
        original_n = n # Keep original for logging
        try:
            if isinstance(n, (int, float)):
                num_val = n # Keep as is for now, final int conversion later
            elif isinstance(n, str):
                 processed_n = n.strip()
                 # More robust cleaning: remove common non-numeric prefixes/suffixes ONLY
                 # Regex aims to capture an optional negative sign ('-?') followed by digits.
                 cleaned_n_match = re.match(r"^[<\[(]*(-?\d+(?:\.\d+)?)[>\])]*$", processed_n)
                 if cleaned_n_match:
                     processed_n = cleaned_n_match.group(1)
                     logging.debug(f"Cleaned string element '{n}' to '{processed_n}' for '{context_name}'")

                 # Try converting cleaned string to float first (handles int and float strings)
                 num_val = float(processed_n)
            else: # Not int, float, or str
                 logging.warning(f"Non-numeric type '{type(n)}' found in elements for '{context_name}': {original_n}")
                 all_numeric_convertible = False
                 break

            if num_val is not None:
                 coord_pair_numeric.append(num_val) # Add the numeric value (int or float)
            else:
                 # Should not happen if logic is correct
                 all_numeric_convertible = False
                 break

        except (ValueError, TypeError) as e:
             logging.warning(f"Could not convert element '{original_n}' to number for '{context_name}': {e}")
             all_numeric_convertible = False
             break

    if not all_numeric_convertible:
        return None

    # Final check: Should have 2 numeric values. Try converting ALL to int.
    if len(coord_pair_numeric) == 2:
        try:
            # Attempt conversion to int for both elements
            final_pair = [int(x) for x in coord_pair_numeric]
            # Check if conversion resulted in loss of precision for floats
            for original, integer in zip(coord_pair_numeric, final_pair):
                 if isinstance(original, float) and not original.is_integer():
                      logging.warning(f"Non-integer float {original} converted to int {integer} for '{context_name}'. Original elements: {pair_elements}")
            logging.debug(f"Successfully validated and converted to int pair for '{context_name}': {final_pair}")
            return final_pair
        except (ValueError, TypeError) as e:
             logging.warning(f"Could not convert numeric coordinates strictly to integers for '{context_name}': {coord_pair_numeric}. Original elements: {pair_elements}. Error: {e}. Skipping pair.")
             return None
    else:
         # This might happen if initial input was list/tuple of len 2 but elements failed conversion
         logging.warning(f"Coordinate pair validation failed (numeric conversion step) for '{context_name}': Original elements: {pair_elements} -> Numeric: {coord_pair_numeric}")
         return None


def _parse_and_validate_coord_list(coord_list_input: Any, context_name: str) -> List[List[int]]:
    """
    Validates if the input (typically from parsed dict value or extracted region)
    is a list where each item can be validated as a coordinate pair.
    Handles nested lists like [[x,y], [x,y]].

    Args:
        coord_list_input (Any): The input to parse as a list of coordinate pairs.
        context_name (str): A string describing the context (e.g., object name)
                            for logging purposes.

    Returns:
        List[List[int]]: A list of valid coordinate pairs (each pair being a list of two integers).
                         Returns an empty list if no valid pairs are found.
    """
    if not isinstance(coord_list_input, list):
         logging.debug(f"Expected a list for coordinate list parsing for '{context_name}', got {type(coord_list_input)}. Value: {str(coord_list_input)[:100]}")
         # If it's not a list, maybe it's a single pair? Try parsing it.
         single_pair = _parse_and_validate_coord_pair(coord_list_input, f"{context_name} (as single item)")
         return [single_pair] if single_pair else []

    valid_coords = []
    for item in coord_list_input:
        # Pass context_name down for better logging
        validated_pair = _parse_and_validate_coord_pair(item, context_name)
        if validated_pair is not None:
            valid_coords.append(validated_pair)
        else:
            logging.debug(f"Item '{item}' in list for '{context_name}' was not a valid coordinate pair.")
    return valid_coords


# --- Helper for Processing Pre-parsed Dictionary (Handles cleaned keys) ---
def _process_preformatted_dict(
    parsed_dict: Dict[str, Any],
    object_set: Set[str]
) -> Dict[str, List[List[int]]]:
    """
    Processes a dictionary (presumably parsed from a string) to extract
    object names and their associated coordinates. Keys are cleaned and
    matched against the `object_set`. Values are parsed as coordinate lists.

    Args:
        parsed_dict (Dict[str, Any]): The dictionary parsed from input.
                                      Keys are expected to be object names (possibly messy),
                                      and values are expected to be coordinate data.
        object_set (Set[str]): A set of lowercase target object names to look for.

    Returns:
        Dict[str, List[List[int]]]: A dictionary mapping cleaned, matched object names
                                    to a list of their valid coordinate pairs.
    """
    result_map = defaultdict(list)
    if not isinstance(parsed_dict, dict):
        logging.error("Internal Error: _process_preformatted_dict called with non-dict input.")
        return {} # Should be an empty dict
    logging.info("Processing parsed dictionary...")
    for key, value in parsed_dict.items():
        # Convert key to string and clean it
        key_str = str(key).lower().strip()
        # Clean keys like '<armchair}:' -> 'armchair', "'table'": -> 'table'
        # Matches word characters or spaces, surrounded by optional non-word chars
        key_cleaned_match = re.match(r"^[^\w\s]*([\w\s]+)[^\w\s]*$", key_str)
        if key_cleaned_match:
            key_cleaned = key_cleaned_match.group(1).strip()
            logging.debug(f"Cleaned dictionary key '{key_str}' to '{key_cleaned}'")
            key_lower = key_cleaned # Already lowercased and stripped
        else:
             logging.warning(f"Could not cleanly extract name from dict key '{key_str}', using lowercased version as is.")
             key_lower = key_str # Use the lowercased string version

        if key_lower in object_set:
            # Use the coordinate list validator helper on the dictionary value
            validated_coords = _parse_and_validate_coord_list(value, key_lower)
            if validated_coords:
                 result_map[key_lower].extend(validated_coords)
                 logging.debug(f"Dict processing: Added {len(validated_coords)} pairs for '{key_lower}'.")
            else:
                 logging.debug(f"Dict processing: No valid coords for '{key_lower}' from value: {value}")
        else:
            logging.debug(f"Dict processing: Skipping key '{key_lower}' (not in object_set).")
    final_map = dict(result_map)
    logging.info(f"Finished processing dictionary. Result: {final_map}")
    return final_map


# --- Helper for Robust String Extraction (REVISED - Non-overlapping Name Finding) ---

def _extract_from_string_robustly(
    map_string: str,
    object_set: Set[str],
    object_list: List[str] # Original list for reference if needed
) -> Dict[str, List[List[int]]]:
    """
    Extracts object names and their coordinates directly from a string when
    dictionary parsing fails. It finds non-overlapping occurrences of object names
    (preferring longest matches in case of overlap) and then extracts coordinate
    pairs from the text region following each name.

    Args:
        map_string (str): The raw string potentially containing object names and coordinates.
        object_set (Set[str]): A set of lowercase target object names to search for.
        object_list (List[str]): The original list of object names (case-sensitive)
                                 used for pattern matching.

    Returns:
        Dict[str, List[List[int]]]: A dictionary mapping found object names
                                    to a list of their extracted coordinate pairs.
    """
    logging.info("Running robust string extraction (non-overlapping, longest match fallback)...")
    result_map = defaultdict(list)
    processed_text_regions: Set[Tuple[int, int]] = set() # Avoid double processing text spans

    # 1. Find ALL potential occurrences using the whole-word regex
    all_occurrences = []
    for obj_name in object_list: # Iterate through original list to preserve case for regex if needed, though pattern is IGNORECASE
        if not obj_name or not isinstance(obj_name, str):
             continue
        obj_name_lower = obj_name.lower() # For matching against object_set and as key
        if obj_name_lower in object_set:
             try:
                 # Whole-word match, case-insensitive
                 pattern_str = r'(?<![a-zA-Z])' + re.escape(obj_name) + r'(?![a-zA-Z])'
                 pattern = re.compile(pattern_str, re.IGNORECASE)
                 logging.debug(f"Searching for pattern: {pattern_str}")
                 for match in pattern.finditer(map_string):
                     logging.debug(f"Found potential match for '{obj_name_lower}': span=({match.start()}, {match.end()}), text='{match.group()}'")
                     all_occurrences.append({
                         "name": obj_name_lower, # Use lowercase name as key
                         "match_obj": match, # Store match object for start/end
                         "start": match.start(),
                         "end": match.end()
                     })
             except re.error as e:
                  logging.error(f"Regex error compiling/using pattern for '{obj_name}': {e}")
             except Exception as e:
                 logging.error(f"Unexpected error finding matches for '{obj_name}': {e}")

    if not all_occurrences:
        logging.info("Robust extraction: No occurrences of target object names found.")
        return {} # Return an empty dict

    # 2. Sort occurrences primarily by start position, secondarily by END position (descending)
    # Sorting by descending end helps ensure that if two matches start at the same place,
    # the longer one comes first, simplifying the filtering logic slightly.
    all_occurrences.sort(key=lambda x: (x["start"], -x["end"]))
    logging.debug(f"Found {len(all_occurrences)} potential matches (sorted by start, then desc end): {[(o['name'], o['start'], o['end']) for o in all_occurrences]}")

    # 3. Filter out overlapping matches, keeping the longest one
    filtered_occurrences = []
    if not all_occurrences: # Should be redundant given the check above, but safe
        return {} # Return an empty dict

    # Always add the first match
    filtered_occurrences.append(all_occurrences[0])

    for i in range(1, len(all_occurrences)):
        current_match = all_occurrences[i]
        last_accepted_match = filtered_occurrences[-1]

        # Check for overlap: current starts before the last one ends
        if current_match["start"] < last_accepted_match["end"]:
            # Overlap detected!
            # current_len = current_match["end"] - current_match["start"] # Not directly used in this logic
            # last_len = last_accepted_match["end"] - last_accepted_match["start"] # Not directly used

            # --- KEY DECISION LOGIC ---
            # If current match is strictly longer than the last accepted one AND they overlap
            # (This happens if the sorting put a shorter match first somehow, or complex overlaps)
            # OR if the current match starts AT OR AFTER the last one started but ends later
            # (meaning it encompasses the last one or is simply longer starting near the same spot)
            # Basically, if the current one provides *more coverage* or is the same coverage but potentially
            # preferred due to being later in the initial scan (less critical now with sorting).
            # The primary case is: is current one contained within last one?
            if current_match["end"] <= last_accepted_match["end"]:
                 # Current match is fully contained within or ends at the same place as the last accepted match.
                 # Since we sorted by start then descending end, the last_accepted should be the longest/preferred one already.
                 logging.debug(f"Filtering overlap: Discarding '{current_match['name']}' ({current_match['start']}-{current_match['end']}) because it's covered by '{last_accepted_match['name']}' ({last_accepted_match['start']}-{last_accepted_match['end']})")
                 continue # Discard current_match
            else:
                 # Current match starts within the last one but extends further.
                 # This implies the *last* one was the shorter, contained match.
                 # Replace the last accepted match with the current, longer one.
                 logging.debug(f"Filtering overlap: Replacing '{last_accepted_match['name']}' ({last_accepted_match['start']}-{last_accepted_match['end']}) with '{current_match['name']}' ({current_match['start']}-{current_match['end']}) as it is longer/extends further.")
                 filtered_occurrences[-1] = current_match
        else:
            # No overlap, simply add the current match
            filtered_occurrences.append(current_match)

    logging.info(f"Filtered down to {len(filtered_occurrences)} non-overlapping longest matches.")
    logging.debug(f"Final occurrences for region processing: {[(o['name'], o['start'], o['end']) for o in filtered_occurrences]}")


    # 4. Iterate through the *filtered* occurrences to define regions and extract numbers
    num_occurrences = len(filtered_occurrences) # Use filtered list count
    for i, current_occurrence in enumerate(filtered_occurrences): # Iterate over FILTERED list
        current_name = current_occurrence["name"]
        region_start = current_occurrence["end"] # Start searching *after* the name ends

        # Determine end of search region for numbers
        region_end = len(map_string) # Default: search until the end of the map string
        if i + 1 < num_occurrences:
            # Use the start of the NEXT occurrence in the FILTERED list
            next_occurrence_start = filtered_occurrences[i+1]["start"]
            region_end = next_occurrence_start
        else:
             # Last item, search till end of string
             pass

        # --- Region processing (extracting numbers) remains the same ---
        if region_start >= region_end:
            logging.debug(f"Skipping empty or invalid region after '{current_name}' at {current_occurrence['start']} (region: [{region_start}:{region_end}]).")
            continue

        # Check against processed regions (secondary check for complex cases)
        # This might become less critical with filtering but kept for safety
        # is_processed = False # This variable is not used
        adjusted_region_start = region_start
        for p_start, p_end in processed_text_regions:
            if adjusted_region_start >= p_start and adjusted_region_start < p_end: # If current region starts within an already processed one
                new_start = p_end # Adjust start to be after the already processed region
                logging.debug(f"Adjusting region start for '{current_name}' from {adjusted_region_start} to {new_start} due to overlap with processed region [{p_start}, {p_end}].")
                adjusted_region_start = new_start

        if adjusted_region_start >= region_end: # If adjustment makes region invalid
            logging.debug(f"Skipping region for '{current_name}' as adjusted start ({adjusted_region_start}) meets or exceeds end ({region_end}) due to processed region overlap.")
            continue

        region_start = adjusted_region_start # Use the adjusted start
        search_region_text = map_string[region_start:region_end]
        logging.debug(f"Processing region for '{current_name}' (found at {current_occurrence['start']}): Final Region=[{region_start}:{region_end}] -> Text='{search_region_text[:100].replace(chr(10), ' ')}...'")

        # Extract all sequences of digits (potentially with signs/decimals) from the region
        number_strings = re.findall(r"-?\d+(?:\.\d+)?", search_region_text)
        logging.debug(f"Found number strings in region for '{current_name}': {number_strings}")

        found_coords_in_region = []
        if len(number_strings) >= 2: # Need at least two numbers for a pair
            for j in range(0, len(number_strings) - 1, 2): # Iterate in steps of 2
                pair_str_tuple = (number_strings[j], number_strings[j+1])
                validated_pair = _parse_and_validate_coord_pair(pair_str_tuple, f"{current_name} (robust)")
                if validated_pair:
                    found_coords_in_region.append(validated_pair)
                else:
                    # Logging for failed pair validation is done inside _parse_and_validate_coord_pair
                    pass # Continue to next potential pair

        if found_coords_in_region:
            logging.info(f"Robust extraction success: Found {len(found_coords_in_region)} pairs for '{current_name}': {found_coords_in_region}")
            result_map[current_name].extend(found_coords_in_region)
            processed_text_regions.add((region_start, region_end)) # Mark this text span as processed
        else:
            logging.debug(f"No valid coordinate pairs extracted from numbers for '{current_name}' in region [{region_start}:{region_end}]")
            processed_text_regions.add((region_start, region_end)) # Mark as processed even if no coords found to avoid re-evaluation


    final_map = dict(result_map)
    logging.info(f"Finished robust extraction (non-overlapping, longest match). Result: {final_map}")
    return final_map

# --- Main Extraction Function (Structure Unchanged, uses revised helpers) ---
def extract_map_data(map_string: str, object_list: List[str]) -> Dict[str, List[List[int]]]:
    """
    Extracts structured map data (object names to coordinate lists) from a string.

    It first attempts to parse the string as a Python dictionary literal.
    If that fails, it falls back to a robust positional extraction method,
    searching for object names from `object_list` in the string and then
    extracting subsequent numbers as coordinates.

    Args:
        map_string (str): The input string suspected to contain map data.
        object_list (List[str]): A list of target object names (case-sensitive,
                                 but matching is generally case-insensitive).

    Returns:
        Dict[str, List[List[int]]]: A dictionary mapping object names (lowercase)
                                    to a list of their coordinate pairs.
                                    Returns an empty dictionary if input is invalid
                                    or no data can be extracted.
    """
    if not isinstance(map_string, str) or not map_string:
        logging.warning("Input map_string must be a non-empty string.")
        return {}
    if not isinstance(object_list, list):
        logging.warning("Input object_list must be a list of strings.")
        return {}

    # Filter object_list for valid strings and create lowercase set for matching
    valid_object_names = [name for name in object_list if isinstance(name, str) and name]
    object_set = {name.lower() for name in valid_object_names} # For quick lookups and as keys in result

    if not object_set:
         logging.warning("Provided object_list is empty or contains no valid strings.")
         return {}

    map_string_stripped = map_string.strip()
    map_string_cleaned = map_string_stripped # Start with stripped version

    logging.info(f"Processing input string (length {len(map_string_stripped)}): '{map_string_stripped[:100]}...'")
    logging.debug(f"Target object set: {object_set}")

    # --- Step 0: Preprocessing (Optional 'str{...}' removal) ---
    # Handles cases where the string might be wrapped like 'str{"key": "value"}'
    if map_string_cleaned.startswith('str{') and map_string_cleaned.endswith('}'):
         potential_dict_str = map_string_cleaned[4:-1].strip() # Remove 'str{' and '}', then strip whitespace
         if potential_dict_str.startswith('{') and potential_dict_str.endswith('}'): # Check if inner content looks like a dict
              map_string_cleaned = potential_dict_str
              logging.info("Removed 'str{...}' wrapper and stripped.")
         else:
              logging.info("Found 'str{...}' but inner content doesn't look like dict. Using original string for ast.")
              map_string_cleaned = map_string_stripped # Revert to original stripped string if inner part is not a dict


    # --- Step 1: Try parsing as a dictionary literal ---
    parsed_successfully = False
    if map_string_cleaned.startswith('{') and map_string_cleaned.endswith('}'):
        logging.info("Attempting ast.literal_eval on potentially cleaned string...")
        try:
            # ast.literal_eval is safer than eval() for parsing Python literals.
            # It's strict and may not handle all string formats models might produce.
            parsed_data = ast.literal_eval(map_string_cleaned)
            if isinstance(parsed_data, dict):
                logging.info("Successfully parsed as dictionary via literal_eval.")
                # Process the parsed dictionary to standardize keys and validate coordinates
                return _process_preformatted_dict(parsed_data, object_set)
                # parsed_successfully = True # This line would be unreachable due to return
            else:
                logging.info(f"Parsed via literal_eval, but not a dict (type: {type(parsed_data)}). Falling back.")
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError) as e:
            # These are common errors if the string is not a valid Python literal.
            logging.info(f"Not a valid standard dictionary literal ({type(e).__name__}: {e}). Falling back to robust extraction.")
        except Exception as e:
             # Catch any other unexpected errors during parsing.
             logging.warning(f"Unexpected error during dict parsing ({type(e).__name__}: {e}). Falling back.")
    else:
         logging.info("Input (after potential cleaning) doesn't look like dict literal. Proceeding directly to robust extraction.")

    # --- Step 2: Fallback to Robust Positional Extraction ---
    # This step is reached if parsing_successfully is still False
    # (i.e., ast.literal_eval failed or was skipped).
    if not parsed_successfully: # This condition will always be true if we reach here due to the return in the success case
        # Use the original stripped string and the validated object list
        return _extract_from_string_robustly(map_string_stripped, object_set, valid_object_names)
    else:
         # This path should ideally not be taken if the logic above is correct (due to returns).
         # Included for logical completeness or if refactoring changes flow.
         logging.error("Internal logic error: Parsed successfully but did not return from Step 1.")
         return {} # Return empty dict in case of unexpected flow
import numpy as np
from scipy.optimize import linear_sum_assignment # Used for optimal assignment, though current code uses greedy
import math

def calculate_distance(p1: Coordinate, p2: Coordinate) -> float:
    """
    Calculates the Euclidean distance between two 2D points.
    This is a wrapper around np.linalg.norm.

    Args:
        p1 (Coordinate): The first point, e.g., [x1, y1].
        p2 (Coordinate): The second point, e.g., [x2, y2].

    Returns:
        float: The Euclidean distance between p1 and p2.
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))

def euclidean_distance(p1: Coordinate, p2: Coordinate) -> float:
    """
    Calculates the Euclidean distance between two 2D points.

    Args:
        p1 (Coordinate): The first point, as a list or tuple [x1, y1].
        p2 (Coordinate): The second point, as a list or tuple [x2, y2].

    Returns:
        float: The Euclidean distance between the two points.

    Raises:
        ValueError: If the coordinates are not 2-dimensional points.
    """
    if len(p1) != 2 or len(p2) != 2:
        raise ValueError("Coordinates must be 2-dimensional points (e.g., [x, y])")
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_prediction_score(response: ResponseMap, solution: SolutionMap, grid_size_N: int) -> float:
    """
    Calculates a prediction score for object localization based on comparing
    a response map to a solution map within a grid of size N x N.

    The score is a weighted average of accuracies for each object type.
    For each object type:
    1.  Points in `response` are matched to points in `solution` using a greedy
        approach (closest pairs first).
    2.  The accuracy for a matched pair is (1 - normalized_distance), where
        normalized_distance is the Euclidean distance divided by the grid diagonal.
    3.  The object type's accuracy is the sum of these matched point accuracies,
        divided by max(num_response_points, num_solution_points) for that type.
        This penalizes predicting too many or too few instances.
    4.  The final score is the sum of (object_type_accuracy * num_solution_points_for_type)
        divided by the total number of solution points across all types.

    Args:
        response (ResponseMap): A dictionary mapping object type (str) to a list
                                of predicted coordinates (List[Coordinate]).
                                Example: {"chair": [[1,2], [3,4]], "table": [[5,5]]}
        solution (SolutionMap): A dictionary mapping object type (str) to a list
                                of ground truth coordinates (List[Coordinate]).
                                Same format as `response`.
        grid_size_N (int): The size of one dimension of the square grid (e.g., if grid is N x N).
                           Used for normalizing distances.

    Returns:
        float: The final prediction score, ranging from 0.0 to 1.0.
               Returns 1.0 if both response and solution are empty.
               Returns 0.0 if solution is not empty but response is, or vice-versa in some cases.

    Raises:
        ValueError: If grid_size_N is not positive.
    """
    if grid_size_N <= 0:
        raise ValueError("Grid size N must be positive.")

    # Calculate the maximum possible distance (diagonal length) for normalization
    # Use a small epsilon to prevent division by zero if N is extremely small or 1
    max_distance = grid_size_N * math.sqrt(2.0)
    if max_distance < 1e-9: # Avoid division by zero or near-zero
         max_distance = 1e-9 # Use a tiny value to avoid literal zero division

    # Consider all object types present in either map
    all_object_types = set(response.keys()) | set(solution.keys())

    weighted_accuracy_sum = 0.0
    total_weight_sum = 0 # Sum of weights (number of objects in solution)

    # Handle the case where both maps are empty initially
    if not all_object_types:
        # If both response and solution are empty (no keys), it's a perfect match.
        return 1.0

    # Iterate through each object type found
    for obj_type in all_object_types:
        res_coords = response.get(obj_type, []) # Get predicted coordinates for the type, or empty list
        sol_coords = solution.get(obj_type, []) # Get ground truth coordinates for the type, or empty list

        n_res = len(res_coords) # Number of predicted instances of this object type
        n_sol = len(sol_coords) # Number of ground truth instances (this is the weight for this object type)

        # Add the number of ground truth objects (weight) to the total weight sum
        total_weight_sum += n_sol

        # --- Calculate the accuracy for this specific object type ---
        # The denominator for the object's accuracy calculation considers the max count
        # to penalize for predicting too many or too few instances.
        num_points_for_average = max(n_res, n_sol)

        object_accuracy = 0.0 # Default accuracy for this object type is 0

        if num_points_for_average == 0:
            # This type exists as a key but has no coordinates in either map (e.g. {"table": []} in both).
            # Consider this a perfect match for "predicting nothing correctly".
            # Its individual accuracy is 1.0.
            object_accuracy = 1.0
            # Note: Since n_sol is 0, its contribution to the weighted sum will be 0 * 1.0 = 0 anyway.
        elif n_res == 0 or n_sol == 0:
            # One side has coordinates, the other doesn't (e.g. response has chairs, solution has no chairs).
            # All points are considered unmatched relative to the max count.
            # The accuracy for this type is 0.
            object_accuracy = 0.0
            # Weighted contribution will be 0 * n_sol = 0 if n_sol > 0, or 0 if n_sol == 0.
        else:
            # Both response and solution have coordinates for this object type.
            # Proceed with matching.
            # --- Matching Strategy: Greedy Best Match ---
            # Create a list of all possible pairings between response and solution points with their distances.
            distances: List[Tuple[float, int, int]] = [] # Stores (distance, res_idx, sol_idx)
            for r_idx, r_coord in enumerate(res_coords):
                for s_idx, s_coord in enumerate(sol_coords):
                    dist = euclidean_distance(r_coord, s_coord)
                    distances.append((dist, r_idx, s_idx))

            # Sort pairs by distance, ascending, to process closest pairs first
            distances.sort()

            matched_res_indices = set() # Keep track of response points already matched
            matched_sol_indices = set() # Keep track of solution points already matched
            num_matched_pairs = 0
            # Accumulator for the sum of accuracies of matched points *within* this object type
            current_type_accuracy_sum = 0.0

            # Greedily match the closest available pairs
            for dist, r_idx, s_idx in distances:
                # If both points in the current pair are currently unmatched
                if r_idx not in matched_res_indices and s_idx not in matched_sol_indices:
                    # Calculate relative distance based on grid diagonal
                    relative_distance = dist / max_distance

                    # Calculate accuracy for this point pair (1 - relative distance, clamped to [0, 1])
                    point_accuracy = max(0.0, 1.0 - relative_distance)
                    current_type_accuracy_sum += point_accuracy

                    # Mark these points as matched
                    matched_res_indices.add(r_idx)
                    matched_sol_indices.add(s_idx)
                    num_matched_pairs += 1

                    # Optimization: If we've matched all possible pairs (limited by the smaller count of points), stop.
                    if num_matched_pairs == min(n_res, n_sol):
                        break


            # Calculate the accuracy for this object type
            if num_points_for_average > 0: # Avoid division by zero if somehow n_res and n_sol were 0 but this block was reached
                 object_accuracy = current_type_accuracy_sum / num_points_for_average
            else:
                 # This case implies n_res=0, n_sol=0. If num_points_for_average was 0,
                 # it should have been caught by the `if num_points_for_average == 0:` block above.
                 # However, as a safeguard:
                 object_accuracy = 1.0 # No points to mismatch.

        # --- Weighted Sum Calculation ---
        # Add the (accuracy of this object type * its weight) to the total sum.
        weighted_accuracy_sum += object_accuracy * n_sol

    # --- Final Score Calculation ---
    if total_weight_sum == 0:
        # This means the solution map was empty (or all its object lists were empty).
        if any(response.values() and any(coords_list for coords_list in response.values())): # Check if there are any actual coordinates predicted in response
             # Solution is empty, but response predicted something. Score is 0.
             return 0.0
        else:
             # Both response and solution are effectively empty (no objects with coordinates).
             # This is considered a perfect match.
             return 1.0
    else:
        # Calculate the weighted average accuracy across all object types.
        final_score = weighted_accuracy_sum / total_weight_sum
        return final_score