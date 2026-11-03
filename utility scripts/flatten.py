import os
import shutil
import zipfile
from pathlib import Path


def extract_without_zip_folders():
  """Finds zip files in 'source_dataset', extracts their contents directly

  into category folders (EARLYBLT, HEALTHY, LATEBLT) without creating
  zip-named subfolders.
  """
  # Get the directory where this script is currently sitting
  parent_dir = Path(__file__).resolve().parent

  # Define the source folder containing the zip files
  source_dir = parent_dir
  # Define categories
  categories = ["EARLYBLT", "HEALTHY", "LATEBLT"]

  # 1. Create the 3 destination category folders in the parent directory
  for category in categories:
    folder_path = parent_dir / category
    folder_path.mkdir(parents=True, exist_ok=True)
    print(f"Created/Verified folder: {folder_path.name}/")

  # 2. Check if the source directory exists
  if not source_dir.exists():
    print(f"\nError: Could not find source directory at: {source_dir}")
    print(
        "Please make sure you have a folder named 'source_dataset' containing"
        " your zip files next to this script."
    )
    return

  extracted_counts = {cat: 0 for cat in categories}

  # 3. Iterate through zip files and extract cleanly
  print(f"\nProcessing zip files from '{source_dir.name}/'...")
  for file_item in source_dir.iterdir():
    if not file_item.is_file() or not file_item.name.endswith(".zip"):
      continue

    filename = file_item.name
    zip_stem = file_item.stem  # e.g., 'EARLYBLT_1' from 'EARLYBLT_1.zip'

    # Match prefix to determine target category folder
    matched_category = None
    for category in categories:
      if filename.startswith(category):
        matched_category = category
        break

    if matched_category:
      dest_folder = parent_dir / matched_category
      print(f"Extracting contents of: {filename} -> {matched_category}/")

      try:
        with zipfile.ZipFile(file_item, "r") as zip_ref:
          for member in zip_ref.namelist():
            # Skip directory header entries themselves
            if member.endswith("/"):
              continue

            # Parse path parts inside the zip
            parts = Path(member).parts

            # If the top-level folder inside the zip matches the zip filename,
            # strip it so we don't recreate a folder named after the zip.
            if len(parts) > 1 and parts[0] == zip_stem:
              relative_path = Path(*parts[1:])
            else:
              relative_path = Path(member)

            # Construct the final destination path for this individual file
            target_file_path = dest_folder / relative_path
            target_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Extract the file safely
            with zip_ref.open(member) as source, open(
                target_file_path, "wb"
            ) as target:
              shutil.copyfileobj(source, target)

        extracted_counts[matched_category] += 1
      except zipfile.BadZipFile:
        print(f"  [Error] '{filename}' is corrupted or not a valid zip file.")
    else:
      print(f"Skipped (unrecognized pattern): {filename}")

  print("\n--- Extraction Summary ---")
  for cat, count in extracted_counts.items():
    print(
        f"Successfully extracted {count} zip archives directly into"
        f" '{cat}/'"
    )


if __name__ == "__main__":
  extract_without_zip_folders()