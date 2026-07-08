"""Recorder manager and recorder-term configuration options."""

import os
import tempfile

from eden.options.options import ConfigurableOptions
from eden.options.managers.base import ManagerOptions
from eden.constants import DatasetExportMode
from eden.options.file_handler import FileHandlerOptions
from eden.utils.file_handler.npz_file_handler import NPZFileHandler


class RecorderTermOptions(ConfigurableOptions):
    """Recorder term specification."""

    ...


class RecorderManagerOptions(ManagerOptions[RecorderTermOptions]):
    """
    Recorder manager options.

    Parameters
    ----------
    file_handler_options: FileHandlerOptions
        The options for the file handler.
    dataset_export_dir_path: str
        The directory path where the recorded datasets are exported.
    dataset_filename: str
        Dataset file name without file extension.
    dataset_export_mode: DatasetExportMode
        The mode to handle episode exports.
    <recorder_term_name>: RecorderTermOptions
        The recorder terms configuration to be used.
    """

    file_handler_options: FileHandlerOptions = NPZFileHandler.configure()
    # Use the platform tempdir so the default is valid on Windows (no ``/tmp``).
    # Resolves to ``/tmp/eden/logs`` on Linux/Mac and ``%TEMP%\eden\logs`` on Windows.
    dataset_export_dir_path: str = os.path.join(tempfile.gettempdir(), "eden", "logs")
    dataset_filename: str = "dataset"
    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_ALL
    override: bool = False
