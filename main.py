import os
import glob
import re
import numpy as np
import napari
from napari.layers import Image, Labels
import SimpleITK as sitk
from PyQt5.QtWidgets import (QPushButton, QVBoxLayout, QWidget, QHBoxLayout, 
                            QComboBox, QLabel, QMessageBox, QSlider, QSpinBox,
                            QProgressBar)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from scipy import ndimage
from skimage import morphology, segmentation
from skimage.filters import gaussian

class HECKTORViewer:
    def __init__(self, data_folder, finals_folder=None, logo_path=None):
        """
        Initialize the HECKTOR dataset viewer
        
        Parameters:
        -----------
        data_folder : str
            Path to the folder containing all scans
        finals_folder : str, optional
            Path to save the edited masks
        logo_path : str, optional
            Path to logo image file
        """
        self.data_folder = data_folder
        self.labels_folder = os.path.join(data_folder, "labels")
        self.logo_path = logo_path
        
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
        
        # Track completed patients
        self.completed_patients = self._get_completed_patients()
        
        # Initialize napari viewer
        self.viewer = napari.Viewer(title="HECKTOR Segmentation Editor")
        
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
        print("HECKTOR SEGMENTATION WITH CLINICAL PROTOCOL")
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
        print("=" * 80)
    
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
        
        # Get completed patients
        completed_patients = self._get_completed_patients_static()
        
        # Separate completed and incomplete patients
        incomplete_patients = [pid for pid in patient_ids if pid not in completed_patients]
        complete_patients = [pid for pid in patient_ids if pid in completed_patients]
        
        # Sort each group separately, then combine (incomplete first)
        incomplete_patients.sort()
        complete_patients.sort()
        
        return incomplete_patients + complete_patients
    
    def _get_completed_patients_static(self):
        """Static method to get completed patients without relying on instance variables"""
        finals_folder = os.path.join(os.path.dirname(self.data_folder), "finals")
        if not os.path.exists(finals_folder):
            return set()
        
        completed_files = glob.glob(os.path.join(finals_folder, "*.nii.gz"))
        completed_ids = set()
        
        for file_path in completed_files:
            # Extract patient ID from filename (remove .nii.gz extension)
            patient_id = os.path.basename(file_path).replace('.nii.gz', '')
            completed_ids.add(patient_id)
        
        return completed_ids
    
    def _get_completed_patients(self):
        """Get list of completed patients from finals folder"""
        if not os.path.exists(self.finals_folder):
            return set()
        
        completed_files = glob.glob(os.path.join(self.finals_folder, "*.nii.gz"))
        completed_ids = set()
        
        for file_path in completed_files:
            # Extract patient ID from filename (remove .nii.gz extension)
            patient_id = os.path.basename(file_path).replace('.nii.gz', '')
            completed_ids.add(patient_id)
        
        return completed_ids
    
    def _update_progress_bar(self):
        """Update the progress bar based on completed patients"""
        if not self.patients:
            self.progress_bar.setValue(0)
            self.progress_label.setText("Progress: 0/0 (0%)")
            return
        
        completed_count = len(self.completed_patients)
        total_count = len(self.patients)
        percentage = int((completed_count / total_count) * 100) if total_count > 0 else 0
        
        self.progress_bar.setValue(percentage)
        self.progress_label.setText(f"Progress: {completed_count}/{total_count} ({percentage}%)")
    
    def _create_ui(self):
        """Create custom UI widgets"""
        # Create a container widget
        container = QWidget()
        layout = QVBoxLayout(container)
        
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
        self.progress_label = QLabel("Progress: 0/0 (0%)")
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
        self.smart_interpolate_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 8px; }")
        
        # Morphological cleanup
        self.cleanup_btn = QPushButton("Clean Up")
        self.cleanup_btn.clicked.connect(self._cleanup_segmentation)
        self.cleanup_btn.setEnabled(False)
        
        # Clear button
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self._clear_segmentation)
        self.clear_btn.setEnabled(False)
        
        smart_interp_layout.addWidget(self.smart_interpolate_btn)
        smart_interp_layout.addWidget(self.cleanup_btn)
        smart_interp_layout.addWidget(self.clear_btn)
        layout.addLayout(smart_interp_layout)

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
            "7. Save final segmentation"
        )
        
        instructions_label = QLabel(instructions_text)
        instructions_label.setWordWrap(True)
        instructions_label.setStyleSheet(
            "QLabel { "
            "background-color: #f0f8ff; "
            "color: #333333; "
            "padding: 8px; "
            "border: 1px solid #cccccc; "
            "border-radius: 5px; "
            "font-size: 12px; "
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
            if logo_pixmap.width() > 300:
                logo_pixmap = logo_pixmap.scaledToWidth(300, Qt.SmoothTransformation)
            logo_label.setPixmap(logo_pixmap)
            logo_label.setAlignment(Qt.AlignCenter)
            logo_label.setStyleSheet(
                "QLabel { "
                "margin-top: 10px; "
                "margin-bottom: 5px; "
                "border-top: 1px solid #cccccc; "
                "padding-top: 10px; "
                "}"
            )
            layout.addWidget(logo_label)
        
        # Add the container to the viewer
        self.viewer.window.add_dock_widget(container, name="HECKTOR Tools", area="right")
    
    def _on_patient_selected(self, index):
        """Handle patient selection from dropdown"""
        if index >= 0 and index < len(self.patients):
            self.current_patient_idx = index
            self.load_patient(self.patients[index])
    
    def _update_patient_info(self):
        """Update UI based on currently loaded patient"""
        if self.current_patient_idx >= 0 and self.current_patient_idx < len(self.patients):
            patient_id = self.patients[self.current_patient_idx]
            
            # Check if patient is completed
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
        else:
            self.patient_label.setText("No patient loaded")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(len(self.patients) > 0)
            self.save_btn.setEnabled(False)
            self.smart_interpolate_btn.setEnabled(False)
            self.cleanup_btn.setEnabled(False)
            self.clear_btn.setEnabled(False)
    
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
    
    def _clear_segmentation(self):
        """Clear all segmentation"""
        if not hasattr(self, 'mask_layer'):
            return
            
        reply = QMessageBox.question(
            self.viewer.window._qt_window,
            "Clear Segmentation",
            "Are you sure you want to clear all segmentation?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            empty_mask = np.zeros_like(self.mask_layer.data)
            self.mask_layer.data = empty_mask
            print("Segmentation cleared")
    
    def load_patient(self, patient_id):
        """Load data for a specific patient"""
        self.current_patient_id = patient_id
        
        # Find CT and PT files with the correct naming pattern
        ct_file = os.path.join(self.data_folder, f"{patient_id}__CT.nii.gz")
        pt_file = os.path.join(self.data_folder, f"{patient_id}__PT.nii.gz")
        
        # Prioritize mask from finals folder, then fall back to original labels
        finals_mask_file = os.path.join(self.finals_folder, f"{patient_id}.nii.gz")
        original_mask_file = os.path.join(self.labels_folder, f"{patient_id}.nii.gz")
        
        if os.path.exists(finals_mask_file):
            mask_file = finals_mask_file
            print(f"Loading edited segmentation from finals folder for patient {patient_id}")
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
            
            # Store original mask for comparison
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
        """Save the segmentation to the finals folder"""
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
        
        # Save to finals folder using the same naming pattern as the labels
        output_file = os.path.join(self.finals_folder, f"{self.current_patient_id}.nii.gz")
        sitk.WriteImage(mask_sitk, output_file)
        
        # Update completed patients and progress bar
        self.completed_patients.add(self.current_patient_id)
        self._update_progress_bar()
        self._update_patient_info()  # Update to show completion checkmark
        
        print(f"Saved segmentation for patient {self.current_patient_id} to {output_file}")
        
        # Show confirmation message
        QMessageBox.information(self.viewer.window._qt_window, 
                              "Save Successful", 
                              f"Saved segmentation for patient {self.current_patient_id}")
    
    def _next_patient(self):
        """Load the next patient"""
        if not self.patients:
            return
            
        self.current_patient_idx = (self.current_patient_idx + 1) % len(self.patients)
        self.load_patient(self.patients[self.current_patient_idx])
    
    def _prev_patient(self):
        """Load the previous patient"""
        if not self.patients:
            return
            
        self.current_patient_idx = (self.current_patient_idx - 1) % len(self.patients)
        self.load_patient(self.patients[self.current_patient_idx])


def main():

    #get data folder from command line argument --data
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Run HECKTOR Viewer")
    parser.add_argument('--data', type=str, default="./test/", help="Path to the folder containing patient data")
    args = parser.parse_args()
    
    # Folder with all patient data
    data_folder = args.data
    if not os.path.exists(data_folder):
        print(f"Data folder '{data_folder}' does not exist. Please provide a valid path.")
        sys.exit(1)
    
    # Optional: Path to logo image (e.g., "./logo.png")
    logo_path = "./logo.png"  # Change this to your logo path or set to None
    
    # Create the viewer
    viewer = HECKTORViewer(data_folder, logo_path=logo_path)
    
    # Start the napari event loop
    napari.run()


if __name__ == "__main__":
    main()