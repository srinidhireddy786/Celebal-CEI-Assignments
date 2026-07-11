from pathlib import Path
from PyPDF2 import PdfReader


def load_documents(data_folder="data"):
    """
    Loads all PDF and TXT files from the given folder.

    Args:
        data_folder (str): Path to the folder containing documents.

    Returns:
        list: List of dictionaries with filename and extracted text.
    """

    documents = []

    folder = Path(data_folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder '{data_folder}' not found.")

    for file in folder.iterdir():

        # -------- PDF Files --------
        if file.suffix.lower() == ".pdf":
            reader = PdfReader(file)

            text = ""

            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

            documents.append({
                "filename": file.name,
                "text": text
            })

        # -------- TXT Files --------
        elif file.suffix.lower() == ".txt":
            with open(file, "r", encoding="utf-8") as f:
                text = f.read()

            documents.append({
                "filename": file.name,
                "text": text
            })

    return documents