import hashlib
import os
import sys
import time

import librosa
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog,
                             QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                             QMainWindow, QMessageBox, QPushButton,
                             QSpacerItem, QSplitter, QTextEdit, QVBoxLayout,
                             QWidget)

import sqlite_db
import utils
from maxima_pairing_algorithm import MaximaPairingAlgorithm


class RegistrationWorker(QThread):
    """Thread for registering audio files by generating their fingerprints and storing them in a database."""
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, data_dir, db_path, clear_db, exts, params):
        super().__init__()
        self.data_dir = data_dir
        self.db_path = db_path
        self.clear_db = clear_db
        self.exts = exts
        self.params = params

    def run(self):
        self.log_signal.emit("Starting registration...")
        fingerprint_generator = MaximaPairingAlgorithm(
            sr=self.params['sample_rate'],
            n_fft=self.params['n_fft'],
            hop_length=self.params['hop_length'],
            neighborhood_size=self.params['peak_neighborhood_size'],
            min_amplitude=self.params['min_amplitude'],
            target_t_min=self.params['target_t_min'],
            target_t_max=self.params['target_t_max'],
            target_f_max_delta=self.params['target_f_max_delta'],
            hash_algorithm=hashlib.sha1 if self.params['hash_algorithm'] == "sha1" else hashlib.sha256
        )
        db = sqlite_db.SQLiteDB(db_path=self.db_path)
        if self.clear_db:
            self.log_signal.emit(f"Clearing database at {self.db_path}...")
            db.clear_db()

        audio_files = []
        for root, dirs, files in os.walk(self.data_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in self.exts:
                    full_path = os.path.join(root, f)
                    if os.access(full_path, os.R_OK):
                        audio_files.append(full_path)

        self.log_signal.emit(f"Found {len(audio_files)} audio files to register.")
        register_count = 0
        for file in audio_files:
            start = time.time()
            if db.file_already_registered(file):
                self.log_signal.emit(f"Skipping registration: '{os.path.basename(file)}' already registered.")
                continue
            
            fingerprints = fingerprint_generator.generate_fingerprints(file)
            if not fingerprints:
                self.log_signal.emit(f"Could not generate fingerprints for {file}. Skipping registration.")
                continue

            audio_id = db.register_audio(file, {"filename": os.path.basename(file)}, fingerprints)
            duration = time.time() - start
            if audio_id is not None:
                register_count += 1
                self.log_signal.emit(f"Registered '{os.path.basename(file)}' (ID: {audio_id}) in {duration:.2f}s. [Fingerprints: {len(fingerprints)}]")
            else:
                self.log_signal.emit(f"Failed to register '{os.path.basename(file)}' in {duration:.2f}s.")
        db.close()
        self.log_signal.emit(f"Registration completed. Total files registered: {register_count}")
        self.finished_signal.emit()


class MatchingWorker(QThread):
    """Thread for matching audio fingerprints."""
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, query_file, db_path, params, start_time, end_time):
        super().__init__()
        self.query_file = query_file
        self.db_path = db_path
        self.params = params
        self.start_time = start_time        # Start time for audio cut
        self.end_time = end_time            # End time for audio cut

    def run(self):
        self.log_signal.emit("Starting matching process...")
        if not os.path.exists(self.query_file):
            self.log_signal.emit(f"Error: Query file not found: {self.query_file}")
            self.finished_signal.emit()
            return

        fingerprint_generator = MaximaPairingAlgorithm(
            sr=self.params['sample_rate'],
            n_fft=self.params['n_fft'],
            hop_length=self.params['hop_length'],
            neighborhood_size=self.params['peak_neighborhood_size'],
            min_amplitude=self.params['min_amplitude'],
            target_t_min=self.params['target_t_min'],
            target_t_max=self.params['target_t_max'],
            target_f_max_delta=self.params['target_f_max_delta'],
            hash_algorithm=hashlib.sha1 if self.params['hash_algorithm'] == "sha1" else hashlib.sha256
        )

        db = sqlite_db.SQLiteDB(db_path=self.db_path)
        start = time.time()
        query_fingerprints = fingerprint_generator.generate_fingerprints(self.query_file, self.start_time, self.end_time)
        if not query_fingerprints:
            self.log_signal.emit("No fingerprints generated for the query file.")
            db.close()
            self.finished_signal.emit()
            return

        self.log_signal.emit(f"Generated {len(query_fingerprints)} fingerprints for the query file.")
        best_match_audio_id, message = db.find_match(query_fingerprints)
        elapsed = time.time() - start
        self.log_signal.emit(f"{message}\nMatching took {elapsed:.2f}s.")
        db.close()
        self.finished_signal.emit()


class MainWindow(QMainWindow):
    """Main window has a Registration and a Matching sections, side by side."""
    def __init__(self):
        self.HEIGHT = 700
        self.WIDTH = int(self.HEIGHT * 1.5)
        super().__init__()
        self.setWindowTitle("Audio Fingerprinting UI")
        self.resize(self.WIDTH, self.HEIGHT)
        self.setMinimumSize(800, 600)
        self.setGeometry(100, 100, self.WIDTH, self.HEIGHT)
        self._init_ui()
        self._center_window()

    def _center_window(self):
        screen = QApplication.primaryScreen()
        screen_center = screen.availableGeometry().center()
        frame_geom = self.frameGeometry()
        frame_geom.moveCenter(screen_center)
        self.move(frame_geom.topLeft())

    def _init_ui(self):
        splitter = QSplitter(Qt.Horizontal)

        # Left registration widget
        self.register_widget = QWidget()
        self._setup_registration_widget(self.register_widget)
        splitter.addWidget(self.register_widget)

        # Right matching widget
        self.match_widget = QWidget()
        self._setup_matching_widget(self.match_widget)
        splitter.addWidget(self.match_widget)

        splitter.setSizes([450, 450])

        self.setCentralWidget(splitter)

    def _setup_registration_widget(self, widget):
        layout = QVBoxLayout()
        form = QFormLayout()

        # Data directory field
        self.data_dir_input = QLineEdit("data/")
        self.data_dir_input.setPlaceholderText("Select folder containing your audio files")
        form.addRow("Data Directory:", self.data_dir_input)
        self.data_dir_browse = QPushButton("Browse")
        self.data_dir_browse.clicked.connect(self._browse_data_dir)
        form.addRow("", self.data_dir_browse)

        # DB path field
        self.db_path_input = QLineEdit("fingerprints.db")
        form.addRow("Database Path:", self.db_path_input)
        self.db_path_browse = QPushButton("Browse")
        self.db_path_browse.clicked.connect(self._browse_db_file)
        form.addRow("", self.db_path_browse)

        # Clear DB option
        self.clear_db_checkbox = QCheckBox("Clear DB before registering")
        form.addRow("", self.clear_db_checkbox)

        # Audio extensions field
        self.exts_input = QLineEdit(".wav,.mp3,.flac,.ogg,.m4a")
        form.addRow("Audio Extensions:", self.exts_input)

        # Fingerprint generator parameters
        self.sample_rate_input = QLineEdit("22050")
        form.addRow("Sample Rate:", self.sample_rate_input)

        self.n_fft_input = QLineEdit("2048")
        form.addRow("FFT Window Size:", self.n_fft_input)

        self.hop_length_input = QLineEdit("512")
        form.addRow("Hop Length:", self.hop_length_input)

        self.peak_neighborhood_input = QLineEdit("20")
        form.addRow("Peak Neighborhood Size:", self.peak_neighborhood_input)

        self.min_amplitude_input = QLineEdit("10")
        form.addRow("Minimum Amplitude:", self.min_amplitude_input)

        self.target_t_min_input = QLineEdit("5")
        form.addRow("Target T Min:", self.target_t_min_input)

        self.target_t_max_input = QLineEdit("100")
        form.addRow("Target T Max:", self.target_t_max_input)

        self.target_f_max_delta_input = QLineEdit("100")
        form.addRow("Target F Max Delta:", self.target_f_max_delta_input)

        self.hash_algorithm_combo = QComboBox()
        self.hash_algorithm_combo.addItems(["sha1", "sha256"])
        form.addRow("Hash Algorithm:", self.hash_algorithm_combo)

        layout.addLayout(form)

        # Registration button and log area
        self.register_btn = QPushButton("Register Audio Files")
        self.register_btn.clicked.connect(self._start_registration)
        layout.addWidget(self.register_btn)

        self.reg_log = QTextEdit()
        self.reg_log.setReadOnly(True)
        layout.addWidget(self.reg_log)

        widget.setLayout(layout)

    def _setup_matching_widget(self, widget):
        layout = QVBoxLayout()
        form = QFormLayout()

        # Query file field
        self.query_file_input = QLineEdit("data/fma/music-fma-0002.wav")
        self.query_file_input.setPlaceholderText("Select the query audio file")
        form.addRow("Query File:", self.query_file_input)
        hbox = QHBoxLayout()
        self.query_file_browse = QPushButton("Browse")
        self.query_file_browse.clicked.connect(self._browse_query_file)
        hbox.addWidget(self.query_file_browse)
        form.addRow("", hbox)

        self.segment_start_input = QLineEdit("00:00")
        form.addRow("Start Time (mm:ss):", self.segment_start_input)
        self.segment_end_input = QLineEdit("")
        form.addRow("End Time (mm:ss):", self.segment_end_input)
        self._set_file_duration(self.query_file_input.text().strip())

        # Database path for matching
        self.match_db_path_input = QLineEdit("fingerprints.db")
        form.addRow("Database Path:", self.match_db_path_input)
        self.match_db_browse = QPushButton("Browse")
        self.match_db_browse.clicked.connect(self._browse_db_file_match)
        form.addRow("", self.match_db_browse)

        # Matching fingerprint generator parameters
        self.match_sample_rate_input = QLineEdit("22050")
        form.addRow("Sample Rate:", self.match_sample_rate_input)

        self.match_n_fft_input = QLineEdit("2048")
        form.addRow("FFT Window Size:", self.match_n_fft_input)

        self.match_hop_length_input = QLineEdit("512")
        form.addRow("Hop Length:", self.match_hop_length_input)

        self.match_peak_neighborhood_input = QLineEdit("20")
        form.addRow("Peak Neighborhood Size:", self.match_peak_neighborhood_input)

        self.match_min_amplitude_input = QLineEdit("10")
        form.addRow("Minimum Amplitude:", self.match_min_amplitude_input)

        self.match_target_t_min_input = QLineEdit("5")
        form.addRow("Target T Min:", self.match_target_t_min_input)

        self.match_target_t_max_input = QLineEdit("100")
        form.addRow("Target T Max:", self.match_target_t_max_input)

        self.match_target_f_max_delta_input = QLineEdit("100")
        form.addRow("Target F Max Delta:", self.match_target_f_max_delta_input)

        self.match_hash_algorithm_combo = QComboBox()
        self.match_hash_algorithm_combo.addItems(["sha1", "sha256"])    # TODO use only one of them, no reason to have two options
        form.addRow("Hash Algorithm:", self.match_hash_algorithm_combo)

        layout.addLayout(form)

        # Match button and log area
        self.match_btn = QPushButton("Find Match")
        self.match_btn.clicked.connect(self._start_matching)
        layout.addWidget(self.match_btn)

        self.match_log = QTextEdit()
        self.match_log.setReadOnly(True)
        layout.addWidget(self.match_log)

        widget.setLayout(layout)

    def _browse_data_dir(self):
        folder = QFileDialog.getExistingDirectory(self, 
                                                "Select Data Directory")

        if folder:
            self.data_dir_input.setText(folder)

    def _browse_db_file(self):
        file_path, _ = QFileDialog.getSaveFileName(self, 
                                                "Select Database File", "", 
                                                "SQLite DB (*.db);;All Files (*)")
        
        if file_path:
            self.db_path_input.setText(file_path)

    def _browse_db_file_match(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 
                                                "Select Database File", "", 
                                                "SQLite DB (*.db);;All Files (*)")

        if file_path:
            self.match_db_path_input.setText(file_path)
            
    def _set_file_duration(self, file_path):
        try:
            y, sr = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            self.segment_start_input.setText("00:00")
            self.segment_end_input.setText(f"{minutes:02d}:{seconds:02d}")
        except Exception as e:
            self.match_log.append(f"Error loading file duration: {e}")

    def _browse_query_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 
                                                "Select Query Audio File", "", 
                                                "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a);;All Files (*)")

        if file_path:
            self.query_file_input.setText(file_path)
            self._set_file_duration(file_path)


    def _grab_common_params(self, prefix=""):
        params = {}
        params['sample_rate'] = int(getattr(self, f"{prefix}sample_rate_input").text())
        params['n_fft'] = int(getattr(self, f"{prefix}n_fft_input").text())
        params['hop_length'] = int(getattr(self, f"{prefix}hop_length_input").text())
        params['peak_neighborhood_size'] = int(getattr(self, f"{prefix}peak_neighborhood_input").text())
        params['min_amplitude'] = float(getattr(self, f"{prefix}min_amplitude_input").text())
        params['target_t_min'] = int(getattr(self, f"{prefix}target_t_min_input").text())
        params['target_t_max'] = int(getattr(self, f"{prefix}target_t_max_input").text())
        params['target_f_max_delta'] = int(getattr(self, f"{prefix}target_f_max_delta_input").text())
        if prefix == "":
            params['hash_algorithm'] = self.hash_algorithm_combo.currentText()
        else:
            params['hash_algorithm'] = self.match_hash_algorithm_combo.currentText()
        return params

    def _start_registration(self):
        data_dir = self.data_dir_input.text().strip()
        db_path = self.db_path_input.text().strip()
        
        if not data_dir or not os.path.isdir(data_dir):
            QMessageBox.warning(self, "Input Error", "Please select a valid data directory.")
            return
        if not db_path:
            QMessageBox.warning(self, "Input Error", "Please specify a valid database path.")
            return
        
        exts = set(ext.strip().lower() for ext in self.exts_input.text().split(",") if ext.strip())
        params = self._grab_common_params(prefix="")
        clear_db = self.clear_db_checkbox.isChecked()
        self.reg_log.clear()
        self.register_btn.setEnabled(False)

        self.registration_worker = RegistrationWorker(data_dir, db_path, clear_db, exts, params)
        self.registration_worker.log_signal.connect(self._append_reg_log)
        self.registration_worker.finished_signal.connect(self._registration_finished)
        self.registration_worker.start()

    def _append_reg_log(self, text):
        self.reg_log.append(text)

    def _registration_finished(self):
        self.register_btn.setEnabled(True)
        self._append_reg_log("Registration process finished.")
        self._append_reg_log("-" * 50)
        

    def _start_matching(self):
        query_file = self.query_file_input.text().strip()
        db_path = self.match_db_path_input.text().strip()
        if not query_file or not os.path.isfile(query_file):
            QMessageBox.warning(self, "Input Error", "Please select a valid query audio file.")
            return
        if not db_path or not os.path.isfile(db_path):
            QMessageBox.warning(self, "Input Error", "Please select a valid database file.")
            return
        
        segment_start = utils.mmss_to_seconds(self.segment_start_input.text().strip())
        segment_end = utils.mmss_to_seconds(self.segment_end_input.text().strip())
        
        params = self._grab_common_params(prefix="match_")
        self.match_log.clear()
        self.match_btn.setEnabled(False)

        self.matching_worker = MatchingWorker(query_file, db_path, params, segment_start, segment_end)
        self.matching_worker.log_signal.connect(self._append_match_log)
        self.matching_worker.finished_signal.connect(self._matching_finished)
        self.matching_worker.start()

    def _append_match_log(self, text):
        self.match_log.append(text)

    def _matching_finished(self):
        self.match_btn.setEnabled(True)
        self._append_match_log("Matching proccess finished.")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())