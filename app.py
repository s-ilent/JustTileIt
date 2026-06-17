import sys

# Import Qt with fallbacks for maximum compatibility
try:
    from PyQt6 import QtWidgets
except ImportError:
    try:
        from PySide6 import QtWidgets
    except ImportError:
        from PyQt5 import QtWidgets

from utils import get_device
from gui import MainWindow

def main():
    app = QtWidgets.QApplication(sys.argv)
    device = get_device()
    window = MainWindow(device)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
