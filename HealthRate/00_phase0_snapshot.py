import argparse
import shutil
from pathlib import Path


# ----------------------------
# CONFIG
# ----------------------------

SNAPSHOT_FOLDER_NAME = "Phase 00_ Snapshot"

PRESERVED_FOLDER_NAME = "01_Preserved_Folder_Structure"
FLAT_FOLDER_NAME = "02_All_Files_Flat"


# ----------------------------
# HELPERS
# ----------------------------

def iter_files_recursive(input_path: Path) -> list[Path]:
    """
    Find all files inside the input path.

    If input_path is a single file:
        return just that file.

    If input_path is a folder:
        search through the folder and all subfolders,
        then return every file found.
    """

    if input_path.is_file():
        return [input_path]

    files = []

    for path in input_path.rglob("*"):
        if path.is_file():
            files.append(path)

    return sorted(files, key=lambda x: str(x).lower())


def copy_file_preserve_tree(src: Path, src_root: Path, preserved_root: Path) -> Path:
    """
    Copy one file into the preserved folder while keeping its folder structure.

    Example:

    Original:
        Data/Data Room/file.pdf

    Preserved output:
        Phase 00_ Snapshot/01_Preserved_Folder_Structure/Data Room/file.pdf
    """

    relative_path = src.relative_to(src_root)

    destination_path = preserved_root / relative_path

    destination_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src, destination_path)

    return destination_path


def get_unique_flat_destination(src: Path, flat_root: Path) -> Path:
    """
    Create a safe destination path for the flat folder.

    The flat folder removes folder structure, so duplicate filenames can happen.

    Example:

    Original files:
        Company A/deck.pdf
        Company B/deck.pdf

    Flat output:
        deck.pdf
        deck__copy_001.pdf
    """

    destination_path = flat_root / src.name

    if not destination_path.exists():
        return destination_path

    counter = 1

    while True:
        new_name = f"{src.stem}__copy_{counter:03d}{src.suffix}"
        new_destination_path = flat_root / new_name

        if not new_destination_path.exists():
            return new_destination_path

        counter += 1


def copy_file_flat(src: Path, flat_root: Path) -> Path:
    """
    Copy one file into the flat folder.

    This does not preserve folder structure.

    Example:

    Original:
        Data/Data Room/file.pdf

    Flat output:
        Phase 00_ Snapshot/02_All_Files_Flat/file.pdf
    """

    flat_root.mkdir(parents=True, exist_ok=True)

    destination_path = get_unique_flat_destination(src, flat_root)

    shutil.copy2(src, destination_path)

    return destination_path


# ----------------------------
# MAIN SNAPSHOT LOGIC
# ----------------------------

def run_phase0(input_path: Path, output_root: Path) -> Path:
    """
    Run Phase 0.

    Main goal:
    Copy all files from the input file/folder into two snapshot folders:

        Phase 00_ Snapshot/
            01_Preserved_Folder_Structure/
            02_All_Files_Flat/

    This phase does not parse files.
    This phase does not hash files.
    This phase does not create JSON manifests.

    It only creates clean copied snapshots.
    """

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    snapshot_root = output_root / SNAPSHOT_FOLDER_NAME

    preserved_root = snapshot_root / PRESERVED_FOLDER_NAME
    flat_root = snapshot_root / FLAT_FOLDER_NAME

    preserved_root.mkdir(parents=True, exist_ok=True)
    flat_root.mkdir(parents=True, exist_ok=True)

    files = iter_files_recursive(input_path)

    preserved_copied_count = 0
    flat_copied_count = 0
    error_count = 0

    if input_path.is_file():
        src_root = input_path.parent
    else:
        src_root = input_path

    for src in files:
        try:
            copy_file_preserve_tree(
                src=src,
                src_root=src_root,
                preserved_root=preserved_root,
            )

            preserved_copied_count += 1

            copy_file_flat(
                src=src,
                flat_root=flat_root,
            )

            flat_copied_count += 1

        except Exception as e:
            error_count += 1
            print(f"ERROR copying file: {src}")
            print(f"Reason: {e}")
            print("-" * 80)

    print("=" * 80)
    print("PHASE 0 SNAPSHOT COMPLETE")
    print("=" * 80)
    print(f"Input path:                  {input_path.resolve()}")
    print(f"Snapshot folder:             {snapshot_root.resolve()}")
    print(f"Preserved folder:            {preserved_root.resolve()}")
    print(f"Flat folder:                 {flat_root.resolve()}")
    print(f"Files found:                 {len(files)}")
    print(f"Preserved copies created:    {preserved_copied_count}")
    print(f"Flat copies created:         {flat_copied_count}")
    print(f"Copy errors:                 {error_count}")
    print("=" * 80)

    return snapshot_root


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    """
    Read command-line arguments.

    Example:

    python 00_phase0_snapshot.py "C:\\Users\\amenx\\Desktop\\Data"

    Optional:

    python 00_phase0_snapshot.py "C:\\Users\\amenx\\Desktop\\Data" --output-root "C:\\Users\\amenx\\Desktop\\all phases re-engineered"
    """

    parser = argparse.ArgumentParser(
        description="Phase 0: copy a source file or folder into Phase 00_ Snapshot."
    )

    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the source folder or single file.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=".",
        help="Where the Phase 00_ Snapshot folder will be created. Default is current folder.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_path).expanduser()
    output_root = Path(args.output_root).expanduser()

    output_root.mkdir(parents=True, exist_ok=True)

    run_phase0(
        input_path=input_path,
        output_root=output_root,
    )


if __name__ == "__main__":
    main()