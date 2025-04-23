import hashlib
import os
import sys
import time

import librosa
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog,
                             QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                             QLineEdit, QMainWindow, QMessageBox, QPushButton,
                             QSpacerItem, QSplitter, QStackedWidget, QTextEdit,
                             QVBoxLayout, QWidget)

import sqlite_db
from chroma_algorithm import ChromaAlgorithm
from maxima_pairing_algorithm import MaximaPairingAlgorithm
from spectral_patch_algorithm import SpectralPatchAlgorithm


def mmss_to_seconds(mmss):
    """Converts a time string in mm:ss format to seconds."""
    try:
        parts = mmss.strip().split(":")
        if len(parts) != 2:
            print(f"Invalid time format: '{mmss}'. Expected format is mm:ss.")
            return None
        
        
        minutes = float(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
        
    except Exception as e:
        print(f"Error converting time '{mmss}': {e}")
        return None


class RegistrationWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, data_dir, db_path, clear_db, exts, algo_name, params):
        super().__init__()
        self.data_dir = data_dir
        self.db_path = db_path
        self.clear_db = clear_db
        self.exts = exts
        self.algo_name = algo_name
        self.params = params

    def run(self):
        self.log_signal.emit("Starting registration...")
        if self.algo_name == "MaximaPairingAlgorithm":
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
        elif self.algo_name == "SpectralPatchAlgorithm":
            fingerprint_generator = SpectralPatchAlgorithm(
                sr=self.params['sample_rate'],
                n_fft=self.params['n_fft'],
                hop_length=self.params['hop_length'],
                patch_size=self.params['patch_size'],
                min_patch_energy=self.params['min_patch_energy'],
                hash_algorithm=hashlib.sha1 if self.params['hash_algorithm'] == "sha1" else hashlib.sha256
            )
        elif self.algo_name == "ChromaAlgorithm":
            fingerprint_generator = ChromaAlgorithm(
                sr=self.params['sample_rate'],
                n_fft=self.params['n_fft'],
                hop_length=self.params['hop_length'],
                threshold=self.params['threshold'],
                hash_algorithm=hashlib.sha1 if self.params['hash_algorithm'] == "sha1" else hashlib.sha256
            )            
        else:
            self.log_signal.emit("Unsupported algorithm selected.")
            self.finished_signal.emit()
            return

        db = sqlite_db.SQLiteDB(db_path=self.db_path, clear_db=self.clear_db)

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
            if db.fingerprint_already_registered(file, fingerprint_generator.name):
                self.log_signal.emit(f"Skipping: '{os.path.basename(file)}' already registered.")
                continue

            fingerprints = fingerprint_generator.generate_fingerprints(file)
            if not fingerprints:
                self.log_signal.emit(f"Could not generate fingerprints for {file}. Skipping.")
                continue

            audio_id = db.register_audio(file, {"filename": os.path.basename(file)}, fingerprints, fingerprint_generator.name)
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
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, query_file, db_path, algo_name, params, start_time, end_time):
        super().__init__()
        self.query_file = query_file
        self.db_path = db_path
        self.algo_name = algo_name
        self.params = params
        self.start_time = start_time
        self.end_time = end_time

    def run(self):
        self.log_signal.emit("Starting matching process...")
        if not os.path.exists(self.query_file):
            self.log_signal.emit(f"Error: Query file not found: {self.query_file}")
            self.finished_signal.emit()
            return

        if self.algo_name == "MaximaPairingAlgorithm":
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
        elif self.algo_name == "SpectralPatchAlgorithm":
            fingerprint_generator = SpectralPatchAlgorithm(
                sr=self.params['sample_rate'],
                n_fft=self.params['n_fft'],
                hop_length=self.params['hop_length'],
                patch_size=self.params['patch_size'],
                min_patch_energy=self.params['min_patch_energy'],
                hash_algorithm=hashlib.sha1 if self.params['hash_algorithm'] == "sha1" else hashlib.sha256
            )
        elif self.algo_name == "ChromaAlgorithm":
            from chroma_algorithm import ChromaAlgorithm
            fingerprint_generator = ChromaAlgorithm(
                sr=self.params['sample_rate'],
                n_fft=self.params['n_fft'],
                hop_length=self.params['hop_length'],
                threshold=self.params['threshold'],
                hash_algorithm=hashlib.sha1 if self.params['hash_algorithm'] == "sha1" else hashlib.sha256
            )            
        else:
            self.log_signal.emit("Unsupported algorithm selected.")
            self.finished_signal.emit()
            return

        db = sqlite_db.SQLiteDB(db_path=self.db_path)
        start = time.time()
        query_fingerprints = fingerprint_generator.generate_fingerprints(self.query_file, self.start_time, self.end_time)
        if not query_fingerprints:
            self.log_signal.emit("No fingerprints generated for the query file.")
            db.close()
            self.finished_signal.emit()
            return

        self.log_signal.emit(f"Generated {len(query_fingerprints)} fingerprints for the query file.")
        best_match_audio_id, message = fingerprint_generator.find_match(query_fingerprints, db)
        elapsed = time.time() - start
        self.log_signal.emit(f"{message}\nMatching took {elapsed:.2f}s.")
        db.close()
        self.finished_signal.emit()


class MainWindow(QMainWindow):
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
        main_layout = QVBoxLayout()
        
        algo_params_container = QWidget()
        self._setup_algo_params_container(algo_params_container)
        main_layout.addWidget(algo_params_container, 1)
        
        # Create a splitter for the middle part (registration|matching controls)
        middle_splitter = QSplitter(Qt.Horizontal)
        
        # Registration controls
        self.register_widget = QWidget()
        self._setup_registration_widget(self.register_widget)
        middle_splitter.addWidget(self.register_widget)
        
        # Matching controls
        self.match_widget = QWidget()
        self._setup_matching_widget(self.match_widget)
        middle_splitter.addWidget(self.match_widget)
        
        middle_splitter.setSizes([450, 450])
        main_layout.addWidget(middle_splitter, 1)
        
        # Create a splitter for the bottom part (logs)
        bottom_splitter = QSplitter(Qt.Horizontal)
        
        # Registration log
        self.reg_log = QTextEdit()
        self.reg_log.setReadOnly(True)
        bottom_splitter.addWidget(self.reg_log)
        
        # Matching log
        self.match_log = QTextEdit()
        self.match_log.setReadOnly(True)
        bottom_splitter.addWidget(self.match_log)
        
        bottom_splitter.setSizes([450, 450])
        main_layout.addWidget(bottom_splitter, 2)
        
        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        

    def _setup_registration_widget(self, widget):
        layout = QVBoxLayout()
        form = QFormLayout()
        
        heading = QLabel("Registration")
        heading.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(heading)

        # Data directory input
        self.data_dir_input = QLineEdit("data/")
        self.data_dir_input.setPlaceholderText("Select folder containing your audio files")
        form.addRow("Data Directory:", self.data_dir_input)
        self.data_dir_browse = QPushButton("Browse")
        self.data_dir_browse.clicked.connect(self._browse_data_dir)
        form.addRow("", self.data_dir_browse)

        # Database Path input
        self.db_path_input = QLineEdit("fingerprints.db")
        form.addRow("Database Path:", self.db_path_input)
        self.db_path_browse = QPushButton("Browse")
        self.db_path_browse.clicked.connect(self._browse_db_file)
        form.addRow("", self.db_path_browse)

        self.clear_db_checkbox = QCheckBox("Clear DB before registering")
        form.addRow("", self.clear_db_checkbox)

        # Audio extensions
        self.exts_input = QLineEdit(".wav,.mp3,.flac,.ogg,.m4a")
        form.addRow("Audio Extensions:", self.exts_input)

        self.register_btn = QPushButton("Register Audio Files")
        self.register_btn.clicked.connect(self._start_registration)
        layout.addLayout(form)
        layout.addWidget(self.register_btn)
        
        widget.setLayout(layout)

    def _setup_matching_widget(self, widget):
        layout = QVBoxLayout()
        form = QFormLayout()
        
        # Add a heading label
        heading = QLabel("Matching")
        heading.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(heading)

        self.query_file_input = QLineEdit("data/music-fma-0005.wav")
        self.query_file_input.setPlaceholderText("Select the query audio file")
        form.addRow("Query File:", self.query_file_input)
        self.query_file_browse = QPushButton("Browse")
        self.query_file_browse.clicked.connect(self._browse_query_file)
        form.addRow("", self.query_file_browse)

        self.match_db_path_input = QLineEdit("fingerprints.db")
        form.addRow("Database Path:", self.match_db_path_input)
        self.match_db_browse = QPushButton("Browse")
        self.match_db_browse.clicked.connect(self._browse_db_file_match)
        form.addRow("", self.match_db_browse)

        self.segment_start_input = QLineEdit("00:00")
        form.addRow("Start Time (mm:ss):", self.segment_start_input)
        self.segment_end_input = QLineEdit("")
        form.addRow("End Time (mm:ss):", self.segment_end_input)


        self.match_btn = QPushButton("Find Match")
        self.match_btn.clicked.connect(self._start_matching)
        layout.addLayout(form)
        layout.addWidget(self.match_btn)
        
        widget.setLayout(layout)

    def _setup_algo_params_container(self, widget):
        layout = QVBoxLayout()
        
        # Add a heading label with center alignment
        heading = QLabel("Algorithm Parameters")
        heading.setStyleSheet("font-size: 14pt; font-weight: bold;")
        heading.setAlignment(Qt.AlignCenter)  # Center the heading text
        layout.addWidget(heading)
        
        
        # Algorithm selection in a horizontal layout with centering
        algo_selection_layout = QHBoxLayout()
        algo_selection_layout.addStretch(1)   # Center the content
        
        algo_label = QLabel("Fingerprint Algorithm:")
        algo_selection_layout.addWidget(algo_label)
        
        # Dropdown menu with algorithm options
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["MaximaPairingAlgorithm", "SpectralPatchAlgorithm", "ChromaAlgorithm"])        
        self.algorithm_combo.currentIndexChanged.connect(self._update_algo_params_stack)
        algo_selection_layout.addWidget(self.algorithm_combo)
        
        algo_selection_layout.addStretch(1)     # Center the content
        
        layout.addLayout(algo_selection_layout)
        
        self.algo_params_stack = QStackedWidget()      # Algorithm parameters selection
        
        # MaximaPairingAlgorithm parameters
        page_maxima = QWidget()
        maxima_layout = QHBoxLayout()
        
        # Left column parameters for maxima
        left_form = QFormLayout()
        self.sample_rate_input = QLineEdit("22050")
        left_form.addRow("Sample Rate:", self.sample_rate_input)
        self.n_fft_input = QLineEdit("1024")
        left_form.addRow("FFT Window Size:", self.n_fft_input)
        self.hop_length_input = QLineEdit("512")
        left_form.addRow("Hop Length:", self.hop_length_input)
        self.peak_neighborhood_input = QLineEdit("20")
        left_form.addRow("Peak Neighborhood Size:", self.peak_neighborhood_input)
        
        # Right column parameters for maxima
        right_form = QFormLayout()
        self.min_amplitude_input = QLineEdit("-20")
        right_form.addRow("Minimum Amplitude:", self.min_amplitude_input)
        self.target_t_min_input = QLineEdit("5")
        right_form.addRow("Target T Min:", self.target_t_min_input)
        self.target_t_max_input = QLineEdit("40")
        right_form.addRow("Target T Max:", self.target_t_max_input)
        self.target_f_max_delta_input = QLineEdit("100")
        right_form.addRow("Target F Max Delta:", self.target_f_max_delta_input)
        self.hash_algorithm_combo = QComboBox()
        self.hash_algorithm_combo.addItems(["sha1", "sha256"])
        right_form.addRow("Hash Algorithm:", self.hash_algorithm_combo)
        
        # Add forms to maxima layout
        left_group = QGroupBox("")
        left_group.setLayout(left_form)
        right_group = QGroupBox("")
        right_group.setLayout(right_form)
        
        maxima_layout.addWidget(left_group)
        maxima_layout.addWidget(right_group)
        page_maxima.setLayout(maxima_layout)
        self.algo_params_stack.addWidget(page_maxima)
        
        # SpectralPatchAlgorithm parameters
        page_spectral = QWidget()
        spectral_layout = QHBoxLayout()
        
        # Left column parameters for spectral
        left_spectral_form = QFormLayout()
        self.spectral_sample_rate_input = QLineEdit("22050")
        left_spectral_form.addRow("Sample Rate:", self.spectral_sample_rate_input)
        self.spectral_n_fft_input = QLineEdit("2048")
        left_spectral_form.addRow("FFT Window Size:", self.spectral_n_fft_input)
        self.spectral_hop_length_input = QLineEdit("512")
        left_spectral_form.addRow("Hop Length:", self.spectral_hop_length_input)
        
        # Right column parameters for spectral
        right_spectral_form = QFormLayout()
        self.patch_size_input = QLineEdit("5")
        right_spectral_form.addRow("Patch Size:", self.patch_size_input)
        self.min_patch_energy_input = QLineEdit("50")
        right_spectral_form.addRow("Min Patch Energy:", self.min_patch_energy_input)
        self.spectral_hash_algorithm_combo = QComboBox()
        self.spectral_hash_algorithm_combo.addItems(["sha1", "sha256"])
        right_spectral_form.addRow("Hash Algorithm:", self.spectral_hash_algorithm_combo)
        
        # Add forms to spectral layout
        left_spectral_group = QGroupBox("")
        left_spectral_group.setLayout(left_spectral_form)
        right_spectral_group = QGroupBox("")
        right_spectral_group.setLayout(right_spectral_form)
        
        spectral_layout.addWidget(left_spectral_group)
        spectral_layout.addWidget(right_spectral_group)
        page_spectral.setLayout(spectral_layout)
        self.algo_params_stack.addWidget(page_spectral)
        
        # ChromaAlgorithm parameters
        page_chroma = QWidget()
        chroma_layout = QHBoxLayout()
        
        left_chroma_form = QFormLayout()
        self.sample_rate_input_chroma = QLineEdit("22050")
        left_chroma_form.addRow("Sample Rate:", self.sample_rate_input_chroma)
        self.n_fft_input_chroma = QLineEdit("2048")
        left_chroma_form.addRow("FFT Window Size:", self.n_fft_input_chroma)
        self.hop_length_input_chroma = QLineEdit("512")
        left_chroma_form.addRow("Hop Length:", self.hop_length_input_chroma)
        
        right_chroma_form = QFormLayout()
        self.threshold_input = QLineEdit("0.5")
        right_chroma_form.addRow("Chroma Threshold:", self.threshold_input)
        self.chroma_hash_algorithm_combo = QComboBox()
        self.chroma_hash_algorithm_combo.addItems(["sha1", "sha256"])
        right_chroma_form.addRow("Hash Algorithm:", self.chroma_hash_algorithm_combo)
        
        left_chroma_group = QGroupBox("")
        left_chroma_group.setLayout(left_chroma_form)
        right_chroma_group = QGroupBox("")
        right_chroma_group.setLayout(right_chroma_form)
        
        chroma_layout.addWidget(left_chroma_group)
        chroma_layout.addWidget(right_chroma_group)
        page_chroma.setLayout(chroma_layout)
        self.algo_params_stack.addWidget(page_chroma)
        
        layout.addWidget(self.algo_params_stack)
        widget.setLayout(layout)

    def _update_algo_params_stack(self):
        index = self.algorithm_combo.currentIndex()
        self.algo_params_stack.setCurrentIndex(index)

    def _browse_data_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if folder:
            self.data_dir_input.setText(folder)

    def _browse_db_file(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Database File", "", "SQLite DB (*.db);;All Files (*)")
        if file_path:
            self.db_path_input.setText(file_path)

    def _browse_db_file_match(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Database File", "", "SQLite DB (*.db);;All Files (*)")
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
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Query Audio File", "", "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a);;All Files (*)")
        if file_path:
            self.query_file_input.setText(file_path)
            self._set_file_duration(file_path)

    def _get_current_algorithm_params(self):
        algo_name = self.algorithm_combo.currentText()
        algo_page = self.algorithm_combo.currentIndex()
        
        params = {}
        is_maxima_pairing_algo = algo_page == 0
        is_spectral_patch_algo = algo_page == 1
        is_chroma_algo = algo_page == 2
        
        if is_maxima_pairing_algo:
            params['sample_rate'] = int(self.sample_rate_input.text())
            params['n_fft'] = int(self.n_fft_input.text())
            params['hop_length'] = int(self.hop_length_input.text())
            params['peak_neighborhood_size'] = int(self.peak_neighborhood_input.text())
            params['min_amplitude'] = float(self.min_amplitude_input.text())
            params['target_t_min'] = int(self.target_t_min_input.text())
            params['target_t_max'] = int(self.target_t_max_input.text())
            params['target_f_max_delta'] = int(self.target_f_max_delta_input.text())
            params['hash_algorithm'] = self.hash_algorithm_combo.currentText()
            
        elif is_spectral_patch_algo:
            params['sample_rate'] = int(self.spectral_sample_rate_input.text())
            params['n_fft'] = int(self.spectral_n_fft_input.text())
            params['hop_length'] = int(self.spectral_hop_length_input.text())
            params['patch_size'] = int(self.patch_size_input.text())
            params['min_patch_energy'] = float(self.min_patch_energy_input.text())
            params['hash_algorithm'] = self.spectral_hash_algorithm_combo.currentText()
        
        elif is_chroma_algo:
            params['sample_rate'] = int(self.sample_rate_input_chroma.text())
            params['n_fft'] = int(self.n_fft_input_chroma.text())
            params['hop_length'] = int(self.hop_length_input_chroma.text())
            params['threshold'] = float(self.threshold_input.text())
            params['hash_algorithm'] = self.chroma_hash_algorithm_combo.currentText()
        
        else:
            raise ValueError(f"Unknown algorithm selected: {algo_name}")
            
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
        algo_name = self.algorithm_combo.currentText()
        params = self._get_current_algorithm_params()
        clear_db = self.clear_db_checkbox.isChecked()
        self.reg_log.clear()
        self.register_btn.setEnabled(False)

        self.registration_worker = RegistrationWorker(data_dir, db_path, clear_db, exts, algo_name, params)
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

        segment_start = mmss_to_seconds(self.segment_start_input.text().strip())
        segment_end = mmss_to_seconds(self.segment_end_input.text().strip())
        algo_name = self.algorithm_combo.currentText()
        params = self._get_current_algorithm_params()
        self.match_log.clear()
        self.match_btn.setEnabled(False)

        self.matching_worker = MatchingWorker(query_file, db_path, algo_name, params, segment_start, segment_end)
        self.matching_worker.log_signal.connect(self._append_match_log)
        self.matching_worker.finished_signal.connect(self._matching_finished)
        self.matching_worker.start()

    def _append_match_log(self, text):
        self.match_log.append(text)

    def _matching_finished(self):
        self.match_btn.setEnabled(True)
        self._append_match_log("Matching process finished.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())