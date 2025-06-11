import os
import glob
import re
import numpy as np
import napari
import SimpleITK as sitk
from PyQt5.QtWidgets import (QPushButton, QVBoxLayout, QWidget, QHBoxLayout, 
                            QComboBox, QLabel, QMessageBox, QSlider, QSpinBox,
                            QProgressBar, QDialog, QLineEdit, QDialogButtonBox,
                            QApplication)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont
from scipy import ndimage
from skimage import morphology
from skimage.filters import gaussian
import argparse
import sys

# ADD THIS FUNCTION at the top, before your existing classes
def parse_args():
    """Parse command line arguments for web interface integration"""
    parser = argparse.ArgumentParser(description="HECKTOR Viewer")
    parser.add_argument('--data', type=str, default="./test/", help="Data folder path")
    parser.add_argument('--patient', type=str, help="Specific patient to load")
    parser.add_argument('--annotator', type=str, help="Annotator ID")
    return parser.parse_args()

class AnnotatorLoginDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.annotator_id = None
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("HECKTOR Annotator Login")
        self.setModal(True)
        self.setFixedSize(400, 200)
        
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("HECKTOR Segmentation Tool")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Instructions
        instruction_label = QLabel("Please enter your annotator ID to continue:")
        instruction_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(instruction_label)
        
        # ID input
        id_layout = QHBoxLayout()
        id_layout.addWidget(QLabel("Annotator ID:"))
        
        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("e.g., 01, initials, etc.")
        self.id_input.textChanged.connect(self.validate_input)
        self.id_input.returnPressed.connect(self.accept_login)
        id_layout.addWidget(self.id_input)
        
        layout.addLayout(id_layout)
        
        # Guidelines
        guidelines_label = QLabel(
            "Guidelines:\n"
            "• Use a unique, identifiable ID\n"
            "• Avoid spaces (use underscores instead)\n"
        )
        guidelines_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(guidelines_label)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.ok_button = button_box.button(QDialogButtonBox.Ok)
        self.ok_button.setText("Start Annotation")
        self.ok_button.setEnabled(False)
        
        button_box.accepted.connect(self.accept_login)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        
        # Focus on input field
        self.id_input.setFocus()
    
    def validate_input(self):
        """Enable OK button only if valid ID is entered"""
        text = self.id_input.text().strip()
        # Basic validation: not empty, no spaces, reasonable length
        is_valid = (len(text) >= 1 and 
                   ' ' not in text and 
                   len(text) <= 50 and
                   text.replace('_', '').replace('-', '').isalnum())
        
        self.ok_button.setEnabled(is_valid)
        
        # Visual feedback
        if text and not is_valid:
            self.id_input.setStyleSheet("border: 2px solid red;")
        else:
            self.id_input.setStyleSheet("")
    
    def accept_login(self):
        """Accept the login if ID is valid"""
        if self.ok_button.isEnabled():
            self.annotator_id = self.id_input.text().strip()
            self.accept()
    
    def get_annotator_id(self):
        """Get the entered annotator ID"""
        return self.annotator_id


class HECKTORViewer:
    def __init__(self, data_folder, annotator_id, finals_folder=None, logo_path=None):
        """
        Initialize the HECKTOR dataset viewer
        
        Parameters:
        -----------
        data_folder : str
            Path to the folder containing all scans
        annotator_id : str
            ID of the current annotator
        finals_folder : str, optional
            Path to save the edited masks
        logo_path : str, optional
            Path to logo image file
        """
        self.data_folder = data_folder
        self.labels_folder = os.path.join(data_folder, "labels")
        self.logo_path = logo_path
        self.annotator_id = annotator_id
        
        # Create finals folder if it doesn't exist
        if finals_folder is None:
            self.finals_folder = os.path.join(os.path.dirname(data_folder), "finals")
        else:
            self.finals_folder = finals_folder
            
        if not os.path.exists(self.finals_folder):
            os.makedirs(self.finals_folder)
        
        # Get patient IDs
        self.patients = self._get_patients()
        self.current_patient_idx = -1  # No patient loaded initially
        
        # Track completed patients (by this annotator)
        self.completed_patients = self._get_completed_patients()
        
        # Initialize napari viewer with bottom controls always visible
        self.viewer = napari.Viewer(title=f"HECKTOR Segmentation Editor - Annotator: {self.annotator_id}")
        
        # Ensure bottom controls (slice slider) remain visible in fullscreen
        self._ensure_controls_visible()
        
        # Add custom UI widgets
        self._create_ui()
        
        # Load the first patient if available
        if self.patients:
            self.current_patient_idx = 0
            self.load_patient(self.patients[0])
        else:
            # Don't load any patient initially if none found
            self._update_patient_info()
            
        # Show instructions
        print("=" * 80)
        print(f"HECKTOR SEGMENTATION - ANNOTATOR: {self.annotator_id}")
        print("=" * 80)
        print("LABEL 1 - PRIMARY TUMOR (GTVp):")
        print("• Include entire morphologic anomaly (CT) + hypermetabolic volume (PET)")
        print("• EXCLUDE hypermetabolic activity outside physical tumor limits")
        print("• EXCLUDE nearby FDG-avid lymph nodes")
        print("• Check for tonsillectomy/extensive biopsy history")
        print("\nLABEL 2 - METASTATIC LYMPH NODES (GTVn):")
        print("• Criteria: SUV>2.5 OR diameter ≥1cm OR pathologically confirmed")
        print("• Include morphologic lymphadenopathy + hypermetabolic volume")
        print("• EXCLUDE activity on bony/muscular/vascular structures")
        print("• GTVp and GTVn must be SEPARATED")
        print("\nWORKFLOW:")
        print("1. Review CT morphology + PET uptake")
        print("2. Segment key slices → Smart Interpolate → Clean Up → Save")
        print(f"\nSaved files will include your ID: patient_id_{self.annotator_id}.nii.gz")
        print("=" * 80)

    def _ensure_controls_visible(self):
        """Ensure napari's built-in controls (including slice slider) remain visible in fullscreen"""
        # Access the Qt window
        qt_window = self.viewer.window._qt_window
        
        # Force the dims widget (which contains the slice slider) to always be visible
        dims_widget = self.viewer.window._qt_viewer.dims
        dims_widget.setVisible(True)
        
        # Set minimum height for the main window to ensure bottom controls don't get cut off
        qt_window.setMinimumHeight(600)
        
        # Ensure the status bar is visible (contains slice info)
        if hasattr(qt_window, 'statusBar'):
            qt_window.statusBar().setVisible(True)
        
        # Force layout update
        qt_window.adjustSize()
        
        print("Napari controls visibility ensured for fullscreen mode")
    
    def _get_patients(self):
        """Get list of patient IDs from the data folder, ordered by completion status"""
        # Find all CT files and extract patient IDs
        ct_files = glob.glob(os.path.join(self.data_folder, "*__CT.nii.gz"))
        
        # Extract patient IDs from filenames
        patient_ids = []
        pattern = r"(.+)__CT\.nii\.gz"
        
        for ct_file in ct_files:
            match = re.search(pattern, os.path.basename(ct_file))
            if match:
                patient_id = match.group(1)
                # Only add if matching PT file exists
                pt_file = os.path.join(self.data_folder, f"{patient_id}__PT.nii.gz")
                if os.path.exists(pt_file):
                    patient_ids.append(patient_id)
        
        # Get completed patients (by this annotator)
        completed_patients = self._get_completed_patients_static()
        
        # Separate completed and incomplete patients
        incomplete_patients = [pid for pid in patient_ids if pid not in completed_patients]
        complete_patients = [pid for pid in patient_ids if pid in completed_patients]
        
        # Sort each group separately, then combine (incomplete first)
        incomplete_patients.sort()
        complete_patients.sort()
        
        return incomplete_patients + complete_patients
    
    def _get_completed_patients_static(self):
        """Static method to get completed patients by this annotator"""
        finals_folder = os.path.join(os.path.dirname(self.data_folder), "finals")
        if not os.path.exists(finals_folder):
            return set()
        
        # Look for files with this annotator's ID
        pattern = f"*_{self.annotator_id}.nii.gz"
        completed_files = glob.glob(os.path.join(finals_folder, pattern))
        completed_ids = set()
        
        for file_path in completed_files:
            # Extract patient ID from filename (remove _annotator_id.nii.gz)
            filename = os.path.basename(file_path)
            patient_id = filename.replace(f'_{self.annotator_id}.nii.gz', '')
            completed_ids.add(patient_id)
        
        return completed_ids
    
    def _get_completed_patients(self):
        """Get list of completed patients by this annotator"""
        if not os.path.exists(self.finals_folder):
            return set()
        
        # Look for files with this annotator's ID
        pattern = f"*_{self.annotator_id}.nii.gz"
        completed_files = glob.glob(os.path.join(self.finals_folder, pattern))
        completed_ids = set()
        
        for file_path in completed_files:
            # Extract patient ID from filename (remove _annotator_id.nii.gz)
            filename = os.path.basename(file_path)
            patient_id = filename.replace(f'_{self.annotator_id}.nii.gz', '')
            completed_ids.add(patient_id)
        
        return completed_ids
    
    def _update_progress_bar(self):
        """Update the progress bar based on completed patients by this annotator"""
        if not self.patients:
            self.progress_bar.setValue(0)
            self.progress_label.setText(f"Progress ({self.annotator_id}): 0/0 (0%)")
            return
        
        completed_count = len(self.completed_patients)
        total_count = len(self.patients)
        percentage = int((completed_count / total_count) * 100) if total_count > 0 else 0
        
        self.progress_bar.setValue(percentage)
        self.progress_label.setText(f"Progress ({self.annotator_id}): {completed_count}/{total_count} ({percentage}%)")
    
    def _has_unsaved_changes(self):
        """Check if there are unsaved changes in the current patient"""
        if not hasattr(self, 'mask_layer') or not hasattr(self, 'original_mask'):
            return False
        
        current_mask = self.mask_layer.data
        return not np.array_equal(current_mask, self.original_mask)
    
    def _show_unsaved_changes_dialog(self, action_description="continue"):
        """Show dialog warning about unsaved changes and return user choice"""
        if not self._has_unsaved_changes():
            return "continue"  # No changes, safe to continue
        
        msg_box = QMessageBox(self.viewer.window._qt_window)
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle("⚠️ Unsaved Changes Detected")
        
        msg_box.setText(
            f"<b>You have unsaved changes for patient: {self.current_patient_id}</b>"
        )
        
        msg_box.setInformativeText(
            f"What would you like to do before {action_description}?\n\n"
            "• Save & Continue: Save current work and proceed\n"
            "• Discard & Continue: Lose all changes and proceed\n"
            "• Cancel: Stay with current patient to save manually"
        )
        
        # Create custom buttons
        save_button = msg_box.addButton("💾 Save & Continue", QMessageBox.AcceptRole)
        discard_button = msg_box.addButton("🗑️ Discard & Continue", QMessageBox.DestructiveRole)
        cancel_button = msg_box.addButton("❌ Cancel", QMessageBox.RejectRole)
        
        # Set default button to Save (safest option)
        msg_box.setDefaultButton(save_button)
        
        # Execute dialog
        msg_box.exec_()
        clicked_button = msg_box.clickedButton()
        
        if clicked_button == save_button:
            # Save current work and continue
            self._save_mask()
            return "continue"
        elif clicked_button == discard_button:
            # Show additional confirmation for destructive action
            confirm_discard = QMessageBox.question(
                self.viewer.window._qt_window,
                "Confirm Discard Changes",
                f"⚠️ Are you sure you want to discard all changes for patient {self.current_patient_id}?\n\n"
                "This action cannot be undone!",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if confirm_discard == QMessageBox.Yes:
                return "continue"
            else:
                return "cancel"
        else:
            # Cancel - stay with current patient
            return "cancel"
    
    def _create_ui(self):
        """Create custom UI widgets"""
        # Create a container widget
        container = QWidget()
        layout = QVBoxLayout(container)
        
        # Annotator info section
        annotator_info = QLabel(f"<b>Annotator: {self.annotator_id}</b>")
        annotator_info.setStyleSheet("""
            QLabel {
                background-color: #e3f2fd;
                color: #1976d2;
                padding: 4px;
                border: 2px solid #1976d2;
                border-radius: 5px;
                font-size: 10px;
            }
        """)
        annotator_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(annotator_info)
        
        # Progress Section
        layout.addWidget(QLabel("<b>Progress:</b>"))
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 20px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Progress label
        self.progress_label = QLabel(f"Progress ({self.annotator_id}): 0/0 (0%)")
        self.progress_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_label)
        
        # Update progress bar
        self._update_progress_bar()
        
        # Patient Navigation
        layout.addWidget(QLabel("<b>Patient Navigation:</b>"))
        
        # Patient selection
        patient_select_layout = QHBoxLayout()
        patient_select_layout.addWidget(QLabel("Select Patient:"))
        
        self.patient_combo = QComboBox()
        if self.patients:
            self.patient_combo.addItems(self.patients)
            self.patient_combo.currentIndexChanged.connect(self._on_patient_selected)
        else:
            self.patient_combo.addItem("No patients found")
            self.patient_combo.setEnabled(False)
        
        patient_select_layout.addWidget(self.patient_combo)
        layout.addLayout(patient_select_layout)
        
        # Info label for current patient
        self.patient_label = QLabel("No patient loaded")
        layout.addWidget(self.patient_label)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous Patient")
        self.prev_btn.clicked.connect(self._prev_patient)
        self.prev_btn.setEnabled(False)
        
        self.next_btn = QPushButton("Next Patient")
        self.next_btn.clicked.connect(self._next_patient)
        self.next_btn.setEnabled(len(self.patients) > 0)
        
        self.save_btn = QPushButton("Save Segmentation")
        self.save_btn.clicked.connect(self._save_mask)
        self.save_btn.setEnabled(False)
        
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        nav_layout.addWidget(self.save_btn)
        layout.addLayout(nav_layout)
        

        
        # Drawing tools section
        layout.addWidget(QLabel("<b>Drawing Tools:</b>"))
        
        # Tool selection
        tools_layout = QHBoxLayout()
        
        # Tool selection
        tool_label = QLabel("Edit Tool:")
        self.tool_combo = QComboBox()
        self.tool_combo.addItems(["Paint", "Erase", "Fill"])
        self.tool_combo.currentTextChanged.connect(self._change_tool)
        
        # Brush size
        brush_label = QLabel("Brush Size:")
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setMinimum(1)
        self.brush_slider.setMaximum(50)
        self.brush_slider.setValue(10)
        self.brush_slider.valueChanged.connect(self._change_brush_size)
        
        tools_layout.addWidget(tool_label)
        tools_layout.addWidget(self.tool_combo)
        tools_layout.addWidget(brush_label)
        tools_layout.addWidget(self.brush_slider)
        layout.addLayout(tools_layout)
        
        # Smart Interpolation section
        layout.addWidget(QLabel("<b>Smart Interpolation:</b>"))
        
        # Smart interpolation button
        smart_interp_layout = QHBoxLayout()
        self.smart_interpolate_btn = QPushButton("Smart Interpolate")
        self.smart_interpolate_btn.clicked.connect(self._smart_interpolate)
        self.smart_interpolate_btn.setEnabled(False)
        self.smart_interpolate_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 3px; }")
        
        # Morphological cleanup
        self.cleanup_btn = QPushButton("Clean Up")
        self.cleanup_btn.clicked.connect(self._cleanup_segmentation)
        self.cleanup_btn.setEnabled(False)
        
        smart_interp_layout.addWidget(self.smart_interpolate_btn)
        smart_interp_layout.addWidget(self.cleanup_btn)
        layout.addLayout(smart_interp_layout)

        # Reset/Clear section
        layout.addWidget(QLabel("<b>Reset Options:</b>"))
        
        reset_layout = QHBoxLayout()
        
        # Reload original mask button
        self.reload_original_btn = QPushButton("Reload Original")
        self.reload_original_btn.clicked.connect(self._reload_original_mask)
        self.reload_original_btn.setEnabled(False)
        self.reload_original_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 3px; }")
        
        # Clear button (with enhanced warning)
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self._clear_segmentation)
        self.clear_btn.setEnabled(False)
        self.clear_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 3px; }")
        
        reset_layout.addWidget(self.reload_original_btn)
        reset_layout.addWidget(self.clear_btn)
        layout.addLayout(reset_layout)

        layout.addWidget(QLabel("<b>HECKTOR SEGMENTATION PROTOCOL:</b>"))
        
        # Instructions with detailed protocol guidelines
        instructions_text = (
            "LABEL 1 - Primary Tumor (GTVp):\n"
            "• Include ENTIRE edges of morphologic anomaly on CT\n"
            "• Include corresponding hypermetabolic volume on PET\n"
            "• Use PET/CT fusion for accurate delineation\n"
            "• EXCLUDE hypermetabolic activity outside physical limits\n"
            "• EXCLUDE nearby FDG-avid lymph nodes\n"
            "• Check for tonsillectomy/biopsy history (exclude if extensive)\n\n"
            
            "LABEL 2 - Metastatic Lymph Nodes (GTVn):\n"
            "• Include entire morphologic lymphadenopathy on CT\n"
            "• Include corresponding hypermetabolic volume on PET\n"
            "• Criteria: SUV>2.5 OR diameter ≥1cm OR pathologically confirmed\n"
            "• EXCLUDE activity on bony/muscular/vascular structures\n"
            "• If nodes touch/merge: keep as one structure\n"
            "• GTVn and GTVp must be SEPARATED\n\n"
            
            "WORKFLOW:\n"
            "1. Review CT for morphologic changes\n"
            "2. Review PET for hypermetabolic areas\n"
            "3. Edit segmentation masks if needed\n"
            "4. Use 'Smart Interpolate' to fill gaps\n"
            "5. Use 'Clean Up' to refine results\n"
            "6. Verify separation between GTVp and GTVn\n"
            f"7. Save final segmentation (saves as: patient_{self.annotator_id}.nii.gz)"
        )
        
        instructions_label = QLabel(instructions_text)
        instructions_label.setWordWrap(True)
        instructions_label.setStyleSheet(
            "QLabel { "
            "background-color: #f0f8ff; "
            "color: #333333; "
            "padding: 6px; "
            "border: 1px solid #cccccc; "
            "border-radius: 5px; "
            "font-size: 11px; "
            "line-height: 1.2; "
            "}"
        )
        layout.addWidget(instructions_label)
        
        # Add spacer to push logo to bottom
        layout.addStretch()
        
        # Logo section at the very bottom (if logo path provided)
        if self.logo_path and os.path.exists(self.logo_path):
            logo_label = QLabel()
            logo_pixmap = QPixmap(self.logo_path)
            # Scale logo to reasonable size (max 150px width for bottom placement)
            if logo_pixmap.width() > 250:
                logo_pixmap = logo_pixmap.scaledToWidth(250, Qt.SmoothTransformation)
            logo_label.setPixmap(logo_pixmap)
            logo_label.setAlignment(Qt.AlignCenter)
            logo_label.setStyleSheet(
                "QLabel { "
                "margin-top: 5px; "
                "margin-bottom: 5px; "
                "border-top: 1px solid #cccccc; "
                "padding-top: 8px; "
                "}"
            )
            layout.addWidget(logo_label)
        
        # Add the container to the viewer
        self.viewer.window.add_dock_widget(container, name="HECKTOR Tools", area="right")
    

    
    def _on_patient_selected(self, index):
        """Handle patient selection from dropdown with unsaved changes check"""
        if index >= 0 and index < len(self.patients) and index != self.current_patient_idx:
            # Check for unsaved changes before switching
            choice = self._show_unsaved_changes_dialog("switching to selected patient")
            
            if choice == "continue":
                self.current_patient_idx = index
                self.load_patient(self.patients[index])
            else:
                # Revert dropdown selection if user cancelled
                self.patient_combo.blockSignals(True)
                self.patient_combo.setCurrentIndex(self.current_patient_idx)
                self.patient_combo.blockSignals(False)
    
    def _update_patient_info(self):
        """Update UI based on currently loaded patient"""
        if self.current_patient_idx >= 0 and self.current_patient_idx < len(self.patients):
            patient_id = self.patients[self.current_patient_idx]
            
            # Check if patient is completed by this annotator
            completion_status = " ✓" if patient_id in self.completed_patients else ""
            self.patient_label.setText(f"Current Patient: {patient_id}{completion_status}")
            
            # Update dropdown selection
            self.patient_combo.blockSignals(True)
            self.patient_combo.setCurrentIndex(self.current_patient_idx)
            self.patient_combo.blockSignals(False)
            
            # Enable buttons
            self.prev_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.smart_interpolate_btn.setEnabled(True)
            self.cleanup_btn.setEnabled(True)
            self.clear_btn.setEnabled(True)
            self.reload_original_btn.setEnabled(True)
        else:
            self.patient_label.setText("No patient loaded")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(len(self.patients) > 0)
            self.save_btn.setEnabled(False)
            self.smart_interpolate_btn.setEnabled(False)
            self.cleanup_btn.setEnabled(False)
            self.clear_btn.setEnabled(False)
            self.reload_original_btn.setEnabled(False)
    
    def _smart_interpolate(self):
        """Advanced interpolation using morphological operations and PET guidance for multi-class labels"""
        if not hasattr(self, 'mask_layer'):
            return
            
        # Get current mask data
        mask_data = self.mask_layer.data.copy()
        
        # Find slices that have segmentation
        segmented_slices = []
        for i in range(mask_data.shape[0]):
            if np.any(mask_data[i] > 0):
                segmented_slices.append(i)
        
        if len(segmented_slices) < 2:
            QMessageBox.warning(
                self.viewer.window._qt_window,
                "Insufficient Data",
                "Need at least 2 slices with segmentation to interpolate.\n"
                "Draw segmentation on a few key slices first."
            )
            return
        
        # Get unique labels
        unique_labels = np.unique(mask_data[mask_data > 0])
        print(f"Smart interpolating between slices: {segmented_slices}")
        print(f"Found label classes: {unique_labels}")
        
        # Get the PT image for guidance
        pt_data = self.pt_layer.data
        
        # Create interpolated mask
        interpolated_mask = mask_data.copy()
        
        # Process each gap between segmented slices
        for i in range(len(segmented_slices) - 1):
            start_slice = segmented_slices[i]
            end_slice = segmented_slices[i + 1]
            
            if end_slice - start_slice > 1:
                print(f"Processing gap: slice {start_slice} to {end_slice}")
                
                # Interpolate each label class separately
                for label_value in unique_labels:
                    if label_value == 0:  # Skip background
                        continue
                    
                    # Extract label-specific masks
                    start_label_mask = (mask_data[start_slice] == label_value)
                    end_label_mask = (mask_data[end_slice] == label_value)
                    
                    # Only interpolate if this label exists in both slices
                    if np.any(start_label_mask) and np.any(end_label_mask):
                        # Use advanced interpolation method for this label
                        interpolated_slices = self._advanced_interpolate_gap_multiclass(
                            start_label_mask, 
                            end_label_mask,
                            pt_data[start_slice:end_slice+1],
                            end_slice - start_slice - 1,
                            label_value
                        )
                        
                        # Insert interpolated slices for this label
                        for j, interp_slice in enumerate(interpolated_slices):
                            slice_idx = start_slice + 1 + j
                            # Add this label to the slice (preserving other labels)
                            interpolated_mask[slice_idx][interp_slice] = label_value
                    
                    # Handle cases where label exists in only one slice
                    elif np.any(start_label_mask) or np.any(end_label_mask):
                        print(f"Label {label_value} exists in only one slice - using simpler interpolation")
                        # Use distance-based fade out/in
                        active_mask = start_label_mask if np.any(start_label_mask) else end_label_mask
                        for j in range(end_slice - start_slice - 1):
                            slice_idx = start_slice + 1 + j
                            alpha = (j + 1) / (end_slice - start_slice)
                            
                            if np.any(start_label_mask):
                                # Fade out from start
                                fade_mask = (gaussian(active_mask.astype(float), sigma=1.0) > (0.3 + 0.4 * alpha))
                            else:
                                # Fade in to end
                                fade_mask = (gaussian(active_mask.astype(float), sigma=1.0) > (0.7 - 0.4 * alpha))
                            
                            interpolated_mask[slice_idx][fade_mask] = label_value
        
        # Update the mask layer
        self.mask_layer.data = interpolated_mask
        
        print("Smart interpolation complete!")
        QMessageBox.information(
            self.viewer.window._qt_window,
            "Smart Interpolation Complete",
            f"Interpolated between {len(segmented_slices)} slices for {len(unique_labels)} label classes\n"
            f"Classes: {list(unique_labels)}\n"
            "Use 'Clean Up' to refine the results."
        )
    
    def _advanced_interpolate_gap_multiclass(self, start_mask, end_mask, pt_slices, num_slices, label_value):
        """Advanced interpolation for a specific label class using multiple techniques"""
        interpolated_slices = []
        
        # Convert to binary float for this specific label
        start_binary = start_mask.astype(float)
        end_binary = end_mask.astype(float)
        
        for i in range(num_slices):
            alpha = (i + 1) / (num_slices + 1)
            
            # Method 1: Distance transform interpolation
            start_dist = ndimage.distance_transform_edt(start_binary == 0) - \
                        ndimage.distance_transform_edt(start_binary > 0)
            end_dist = ndimage.distance_transform_edt(end_binary == 0) - \
                      ndimage.distance_transform_edt(end_binary > 0)
            
            interp_dist = (1 - alpha) * start_dist + alpha * end_dist
            distance_mask = (interp_dist <= 0).astype(float)
            
            # Method 2: Morphological interpolation
            start_smooth = gaussian(start_binary, sigma=2.0)
            end_smooth = gaussian(end_binary, sigma=2.0)
            morph_mask = ((1 - alpha) * start_smooth + alpha * end_smooth > 0.3).astype(float)
            
            # Method 3: PET-guided interpolation (adapt threshold based on label)
            current_pt = pt_slices[i + 1]
            # Use different PET thresholds for different labels
            if label_value == 1:
                # Primary tumor - higher PET uptake
                pt_threshold = np.percentile(current_pt[current_pt > 0], 80) if np.any(current_pt > 0) else 0
            else:
                # Secondary structures - moderate PET uptake
                pt_threshold = np.percentile(current_pt[current_pt > 0], 60) if np.any(current_pt > 0) else 0
            
            pt_mask = (current_pt > pt_threshold).astype(float) if pt_threshold > 0 else np.zeros_like(current_pt)
            
            # Combine methods with weights (adjust based on label)
            if label_value == 1:
                # For primary tumor, rely more on distance and PET
                combined_mask = 0.4 * distance_mask + 0.2 * morph_mask + 0.4 * pt_mask
                threshold = 0.4
            else:
                # For other structures, rely more on morphological interpolation
                combined_mask = 0.5 * distance_mask + 0.4 * morph_mask + 0.1 * pt_mask
                threshold = 0.3
            
            # Apply threshold
            final_mask = (combined_mask > threshold).astype(bool)
            
            # Clean up with morphological operations (gentle for multi-class)
            if np.any(final_mask):
                final_mask = morphology.binary_opening(final_mask, morphology.disk(1))
                final_mask = morphology.binary_closing(final_mask, morphology.disk(2))
                final_mask = morphology.remove_small_objects(final_mask, min_size=25)
            
            interpolated_slices.append(final_mask)
        
        return interpolated_slices
    
    def _cleanup_segmentation(self):
        """Clean up segmentation using morphological operations while preserving label classes"""
        if not hasattr(self, 'mask_layer'):
            return
            
        mask_data = self.mask_layer.data.copy()
        cleaned_mask = np.zeros_like(mask_data)
        
        print("Cleaning up segmentation while preserving label classes...")
        
        # Get unique labels (excluding background)
        unique_labels = np.unique(mask_data[mask_data > 0])
        print(f"Found label classes: {unique_labels}")
        
        for i in range(mask_data.shape[0]):
            slice_mask = mask_data[i]
            
            if np.any(slice_mask > 0):
                slice_cleaned = np.zeros_like(slice_mask)
                
                # Process each label class separately
                for label_value in unique_labels:
                    if label_value == 0:  # Skip background
                        continue
                        
                    # Extract current label
                    label_mask = (slice_mask == label_value)
                    
                    if np.any(label_mask):
                        # Apply morphological operations to this label class
                        # Remove small objects
                        label_mask = morphology.remove_small_objects(
                            label_mask, 
                            min_size=20  # Reduced from 30 to preserve smaller structures
                        )
                        
                        # Fill small holes
                        label_mask = morphology.remove_small_holes(
                            label_mask, 
                            area_threshold=50  # Reduced from 100
                        )
                        
                        # Smooth boundaries (gentler operations)
                        label_mask = morphology.binary_opening(label_mask, morphology.disk(1))
                        label_mask = morphology.binary_closing(label_mask, morphology.disk(2))
                        
                        # Add cleaned label back to slice with original label value
                        slice_cleaned[label_mask] = label_value
                
                cleaned_mask[i] = slice_cleaned
        
        self.mask_layer.data = cleaned_mask
        print(f"Cleanup complete! Preserved {len(unique_labels)} label classes: {unique_labels}")
    
    def _reload_original_mask(self):
        """Reload the original mask from when the patient was first loaded"""
        if not hasattr(self, 'mask_layer') or not hasattr(self, 'original_mask'):
            QMessageBox.warning(
                self.viewer.window._qt_window,
                "No Original Mask",
                "No original mask available to reload."
            )
            return
            
        # Show confirmation dialog
        reply = QMessageBox.question(
            self.viewer.window._qt_window,
            "Reload Original Mask",
            "Are you sure you want to reload the original mask?\n\n"
            "⚠️ WARNING: This will discard ALL current changes!\n"
            "• All manual edits will be lost\n"
            "• Smart interpolation results will be lost\n"
            "• Cleanup operations will be lost\n\n"
            "Consider saving your current work first if needed.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Reload original mask
            self.mask_layer.data = self.original_mask.copy()
            print("Original mask reloaded")
            QMessageBox.information(
                self.viewer.window._qt_window,
                "Original Mask Reloaded",
                "The original mask has been restored.\nAll modifications have been discarded."
            )
    
    def _clear_segmentation(self):
        """Clear all segmentation with enhanced warning"""
        if not hasattr(self, 'mask_layer'):
            return
        
        # Enhanced warning dialog with more details
        reply = QMessageBox.question(
            self.viewer.window._qt_window,
            "⚠️ CLEAR ALL SEGMENTATION",
            "🚨 <b>DANGER: This will permanently delete ALL segmentation!</b>\n\n"
            "<b>What will be lost:</b>\n"
            "• All manual annotations (GTVp and GTVn)\n"
            "• Smart interpolation results\n"
            "• Cleanup operations\n"
            "• All drawing/editing work\n\n"
            "<b>This action cannot be undone!</b>\n\n"
            "💡 <b>Alternative:</b> Use 'Reload Original' instead if you want to\n"
            "    go back to the original segmentation without losing everything.\n\n"
            "Are you absolutely sure you want to clear ALL segmentation?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Additional confirmation for destructive action
            final_reply = QMessageBox.warning(
                self.viewer.window._qt_window,
                "Final Confirmation",
                "⚠️ LAST CHANCE TO CANCEL ⚠️\n\n"
                "You are about to delete ALL segmentation work.\n"
                "This cannot be undone!\n\n"
                "Proceed with clearing everything?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel
            )
            
            if final_reply == QMessageBox.Yes:
                empty_mask = np.zeros_like(self.mask_layer.data)
                self.mask_layer.data = empty_mask
                print("All segmentation cleared")
                QMessageBox.information(
                    self.viewer.window._qt_window,
                    "Segmentation Cleared",
                    "All segmentation has been cleared.\nYou can start fresh or reload the original mask."
                )
    
    def load_patient(self, patient_id):
        """Load data for a specific patient"""
        self.current_patient_id = patient_id
        
        # Find CT and PT files with the correct naming pattern
        ct_file = os.path.join(self.data_folder, f"{patient_id}__CT.nii.gz")
        pt_file = os.path.join(self.data_folder, f"{patient_id}__PT.nii.gz")
        
        # Prioritize mask from finals folder with annotator ID, then fall back to original labels
        annotator_mask_file = os.path.join(self.finals_folder, f"{patient_id}_{self.annotator_id}.nii.gz")
        original_mask_file = os.path.join(self.labels_folder, f"{patient_id}.nii.gz")
        
        if os.path.exists(annotator_mask_file):
            mask_file = annotator_mask_file
            print(f"Loading previous annotation by {self.annotator_id} for patient {patient_id}")
        elif os.path.exists(original_mask_file):
            mask_file = original_mask_file
            print(f"Loading original segmentation for patient {patient_id}")
        else:
            mask_file = None
            print(f"Warning: No mask found for patient {patient_id}")
        
        # Check if files exist
        if not os.path.exists(ct_file) or not os.path.exists(pt_file):
            QMessageBox.warning(self.viewer.window._qt_window, 
                                "File Not Found", 
                                f"CT or PT scan for patient {patient_id} not found.")
            return
        
        print(f"Loading patient {patient_id}...")
        
        # Load CT image
        ct_sitk = sitk.ReadImage(ct_file)
        ct_array = sitk.GetArrayFromImage(ct_sitk)
        self.ct_spacing = ct_sitk.GetSpacing()
        self.ct_origin = ct_sitk.GetOrigin()
        self.ct_direction = ct_sitk.GetDirection()
        
        # Load PT image
        pt_sitk = sitk.ReadImage(pt_file)
        
        # Resample PT to CT space (same physical coordinates)
        print(f"Resampling PT to CT space for patient {patient_id}...")
        resampled_pt_sitk = self._register_pt_to_ct(pt_sitk, ct_sitk)
        pt_array = sitk.GetArrayFromImage(resampled_pt_sitk)
        
        # Clear viewer
        self.viewer.layers.clear()
        
        # Add CT as base layer
        self.ct_layer = self.viewer.add_image(
            ct_array, 
            name="CT",
            scale=(self.ct_spacing[2], self.ct_spacing[1], self.ct_spacing[0]),
            colormap='gray',
            contrast_limits=self._auto_contrast(ct_array)
        )
        
        # Add registered PT as another layer
        self.pt_layer = self.viewer.add_image(
            pt_array,
            name="PT (Registered)",
            scale=(self.ct_spacing[2], self.ct_spacing[1], self.ct_spacing[0]),
            colormap='hot',
            blending="additive",
            opacity=0.7,
            contrast_limits=self._auto_contrast(pt_array)
        )
        
        # Load and add mask if available
        if mask_file and os.path.exists(mask_file):
            mask_sitk = sitk.ReadImage(mask_file)
            mask_array = sitk.GetArrayFromImage(mask_sitk)
            
            # Store original mask for comparison and reloading
            self.original_mask = mask_array.copy()
            
            # Create the mask layer
            self.mask_layer = self.viewer.add_labels(
                mask_array,
                name="Segmentation",
                scale=(self.ct_spacing[2], self.ct_spacing[1], self.ct_spacing[0]),
                opacity=0.5
            )
        else:
            # Create empty mask with same dimensions as CT
            mask_array = np.zeros_like(ct_array, dtype=np.uint8)
            self.original_mask = mask_array.copy()
            
            # Create the mask layer
            self.mask_layer = self.viewer.add_labels(
                mask_array,
                name="Segmentation",
                scale=(self.ct_spacing[2], self.ct_spacing[1], self.ct_spacing[0]),
                opacity=0.5
            )
        
        # Save file paths for later use
        self.ct_file = ct_file
        self.pt_file = pt_file
        self.mask_file = mask_file
        
        # Set mask as active layer for drawing
        self.viewer.layers.selection.active = self.mask_layer
        self._change_tool(self.tool_combo.currentText())
        
        # Update UI
        self._update_patient_info()
        
        print(f"Patient {patient_id} loaded successfully")
    
    def _register_pt_to_ct(self, pt_image, ct_image):
        """Resample PT image to CT space (same physical coordinate system)"""
        print("Resampling PT to CT space...")
        
        # Simple resampling to match CT space - no registration needed
        # This assumes PT and CT are already aligned in the same coordinate system
        resampled_pt = sitk.Resample(
            pt_image,           # Input image to resample
            ct_image,           # Reference image (defines output space)
            sitk.Transform(),   # Identity transform (no registration)
            sitk.sitkLinear,    # Linear interpolation
            0.0,                # Default pixel value for outside regions
            pt_image.GetPixelID()  # Preserve original pixel type
        )
        
        print("PT resampled to CT space")
        return resampled_pt
    
    def _auto_contrast(self, image, p_low=0.5, p_high=99.5):
        """Automatically determine contrast limits for better visualization"""
        # Exclude zeros (background) from percentile calculation
        non_zeros = image[image > 0]
        if len(non_zeros) > 0:
            low = np.percentile(non_zeros, p_low)
            high = np.percentile(non_zeros, p_high)
        else:
            low, high = np.min(image), np.max(image)
        
        # Ensure we have a valid range
        if low >= high:
            high = low + 1
            
        return [low, high]
    
    def _change_tool(self, tool_name):
        """Change the active segmentation tool"""
        if not hasattr(self, 'mask_layer'):
            return
            
        # Set the appropriate napari tool
        if tool_name == "Paint":
            self.viewer.layers.selection.active = self.mask_layer
            self.viewer.layers.selection.active.mode = 'paint'
        elif tool_name == "Erase":
            self.viewer.layers.selection.active = self.mask_layer
            self.viewer.layers.selection.active.mode = 'erase'
        elif tool_name == "Fill":
            self.viewer.layers.selection.active = self.mask_layer
            self.viewer.layers.selection.active.mode = 'fill'
    
    def _change_brush_size(self, size):
        """Change brush size for painting/erasing"""
        if hasattr(self, 'mask_layer'):
            self.mask_layer.brush_size = size
    
    def _save_mask(self):
        """Save the segmentation to the finals folder with annotator ID"""
        if not hasattr(self, 'mask_layer') or self.current_patient_id is None:
            return
            
        # Create finals folder if it doesn't exist
        if not os.path.exists(self.finals_folder):
            os.makedirs(self.finals_folder)
        
        # Get final mask data
        final_mask = self.mask_layer.data
        
        # Create SimpleITK image with correct metadata
        mask_sitk = sitk.GetImageFromArray(final_mask)
        mask_sitk.SetSpacing((self.ct_spacing[0], self.ct_spacing[1], self.ct_spacing[2]))
        mask_sitk.SetOrigin(self.ct_origin)
        mask_sitk.SetDirection(self.ct_direction)
        
        # Save to finals folder with annotator ID in filename
        output_file = os.path.join(self.finals_folder, f"{self.current_patient_id}_{self.annotator_id}.nii.gz")
        sitk.WriteImage(mask_sitk, output_file)
        
        # Update the original mask to reflect the saved state
        self.original_mask = final_mask.copy()
        
        # Update completed patients and progress bar
        self.completed_patients.add(self.current_patient_id)
        self._update_progress_bar()
        self._update_patient_info()  # Update to show completion checkmark
        
        print(f"Saved segmentation for patient {self.current_patient_id} by {self.annotator_id} to {output_file}")
        
        # Show confirmation message
        QMessageBox.information(self.viewer.window._qt_window, 
                              "Save Successful", 
                              f"Saved segmentation for patient {self.current_patient_id}\n"
                              f"File: {self.current_patient_id}_{self.annotator_id}.nii.gz\n"
                              f"Annotator: {self.annotator_id}")
    
    def _next_patient(self):
        """Load the next patient with unsaved changes check"""
        if not self.patients:
            return
        
        # Check for unsaved changes before moving to next patient
        choice = self._show_unsaved_changes_dialog("moving to the next patient")
        
        if choice == "continue":
            self.current_patient_idx = (self.current_patient_idx + 1) % len(self.patients)
            self.load_patient(self.patients[self.current_patient_idx])
        # If cancelled, stay with current patient
    
    def _prev_patient(self):
        """Load the previous patient with unsaved changes check"""
        if not self.patients:
            return
        
        # Check for unsaved changes before moving to previous patient
        choice = self._show_unsaved_changes_dialog("moving to the previous patient")
        
        if choice == "continue":
            self.current_patient_idx = (self.current_patient_idx - 1) % len(self.patients)
            self.load_patient(self.patients[self.current_patient_idx])
        # If cancelled, stay with current patient


def get_annotator_id():
    """Get annotator ID through login dialog"""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    login_dialog = AnnotatorLoginDialog()
    result = login_dialog.exec_()
    
    if result == QDialog.Accepted:
        return login_dialog.get_annotator_id()
    else:
        return None


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Use data folder from arguments
    data_folder = args.data
    if not os.path.exists(data_folder):
        print(f"Data folder '{data_folder}' does not exist. Please provide a valid path.")
        sys.exit(1)
    
    # Optional: Path to logo image
    logo_path = "./logo.png"  # Change this to your logo path or set to None
    
    # Get annotator ID (from web interface or user input)
    if args.annotator:
        # Called from web interface with annotator ID
        annotator_id = args.annotator
        print(f"Starting annotation session for: {annotator_id}")
    else:
        # Called directly - show login dialog
        print("Starting HECKTOR Annotation Tool...")
        annotator_id = get_annotator_id()
        
        if annotator_id is None:
            print("Annotation cancelled - no annotator ID provided.")
            sys.exit(0)
        
        print(f"Starting annotation session for: {annotator_id}")
    
    # Create the viewer with annotator ID
    viewer = HECKTORViewer(data_folder, annotator_id, logo_path=logo_path)
    
    # If specific patient requested (from web interface), load it
    if args.patient and args.patient in viewer.patients:
        patient_index = viewer.patients.index(args.patient)
        viewer.current_patient_idx = patient_index
        viewer.load_patient(args.patient)
        print(f"Loaded patient: {args.patient}")
    
    # Start the napari event loop
    napari.run()

# MAKE SURE this is at the very end of your file:
if __name__ == "__main__":
    main()