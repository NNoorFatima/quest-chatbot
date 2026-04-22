import shutil
from pdf2image import convert_from_path

def get_poppler_path():
    if shutil.which("pdftoppm"):
        return None  # already in PATH
    return r"C:\path\to\poppler\Library\bin"

file_path=r"C:\Users\Noor\Downloads\22i-1036_J_Tbw.pdf"
images = convert_from_path(
    file_path,
    dpi=300,
    poppler_path=get_poppler_path()
)