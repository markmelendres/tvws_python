import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import config as cfg
from parameters import SimulationParameters, validate_parameters
from simulation import run_simulation


class BasicWindow(QMainWindow):
    """Provide the initial Qt window used to set up and launch a simulation.

    The widget contains the smallest useful user interface for the OFDM port:
    it exposes the key parameters, validates them before the run begins, and
    hands control to the shared simulation logic once the user clicks start.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TVWS OFDM")
        self.resize(480, 520)

        self.receive = self._spin(1, 64, cfg.DEFAULT_RX)
        self.transmit = self._spin(1, 64, cfg.DEFAULT_TX)
        self.taps = self._spin(1, 32, cfg.DEFAULT_TAPS)
        self.cp = self._spin(0, 1024, cfg.DEFAULT_CP)
        self.fft = self._spin(8, 4096, cfg.DEFAULT_FFT)
        self.symbols = self._spin(1, 10000, cfg.DEFAULT_OFDM_SYMBOLS)
        self.trials = self._spin(1, 100000, cfg.DEFAULT_TRIALS)
        self.jammer = QDoubleSpinBox()
        self.jammer.setRange(-100, 100)
        self.jammer.setValue(cfg.DEFAULT_JAMMER_DB)
        self.modulation = QComboBox()
        self.modulation.addItems(["BPSK", "QPSK", "16QAM", "64QAM"])
        self.modulation.setCurrentText(cfg.DEFAULT_MODULATION)
        self.progress = QProgressBar()
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.run_button = QPushButton("Run simulation")
        self.run_button.clicked.connect(self.run_simulation)

        form = QFormLayout()
        for label, control in (
            ("Receive antennas", self.receive),
            ("Transmit antennas", self.transmit),
            ("Channel taps", self.taps),
            ("Cyclic prefix", self.cp),
            ("FFT size", self.fft),
            ("OFDM symbols", self.symbols),
            ("Monte-Carlo trials", self.trials),
            ("Jammer power (dB)", self.jammer),
            ("Modulation", self.modulation),
        ):
            form.addRow(label, control)

        settings = QGroupBox("Simulation parameters")
        settings.setLayout(form)
        layout = QVBoxLayout()
        layout.addWidget(settings)
        layout.addWidget(self.run_button)
        layout.addWidget(self.progress)
        layout.addWidget(self.output, 1)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    @staticmethod
    def _spin(minimum, maximum, value):
        control = QSpinBox()
        control.setRange(minimum, maximum)
        control.setValue(value)
        return control

    def run_simulation(self):
        params = SimulationParameters(
            num_receive_antennas=self.receive.value(),
            num_transmit_antennas=self.transmit.value(),
            num_channel_taps=self.taps.value(),
            cyclic_prefix_length=self.cp.value(),
            num_subcarriers=self.fft.value(),
            num_ofdm_symbols=self.symbols.value(),
            jammer_power_dB=self.jammer.value(),
            modulation_scheme=self.modulation.currentText(),
            random_jammer_power_flag=False,
            num_trials=self.trials.value(),
            snr_dB_list=np.arange(-5.0, 31.0, 1.0),
        )
        try:
            validate_parameters(params)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid parameters", str(exc))
            return

        self.run_button.setEnabled(False)
        self.progress.setRange(
            0, params.num_trials * len(params.snr_dB_list) * params.num_ofdm_symbols
        )
        self.progress.setValue(0)
        self.output.clear()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            result = run_simulation(
                params,
                progress_callback=self._update_progress,
                message_callback=self.output.appendPlainText,
            )
            self.output.appendPlainText("")
            self.output.appendPlainText("Simulation complete.")
            self.output.appendPlainText(
                f"Seed: {result['seed']} | "
                f"Trials: {params.num_trials} | "
                f"SNR points: {len(params.snr_dB_list)}"
            )
            self.output.appendPlainText(
                "CP correlation (compliant): "
                f"{result['threat_detection']['CP correlation - OFDM compliant']:.4f}"
            )
            self.output.appendPlainText(
                "CP correlation (CP-violating): "
                f"{result['threat_detection']['CP correlation - CP violating']:.4f}"
            )
            self.progress.setValue(self.progress.maximum())
        except Exception as exc:
            QMessageBox.critical(
                self, "Simulation failed", f"{type(exc).__name__}: {exc}"
            )
        finally:
            QApplication.restoreOverrideCursor()
            self.run_button.setEnabled(True)

    def _update_progress(self, current, total):
        self.progress.setMaximum(total)
        self.progress.setValue(current)
        QApplication.processEvents()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BasicWindow()
    window.show()
    sys.exit(app.exec())
