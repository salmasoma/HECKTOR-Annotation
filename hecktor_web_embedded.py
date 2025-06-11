# hecktor_web_embedded.py - Super simple version that definitely works

import os
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display, HTML
import napari

def start_hecktor_web_app(data_folder="./test/"):
    """Start the HECKTOR web application"""
    
    # Display header
    display(HTML(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
        <h2>🚀 HECKTOR Web Application</h2>
        <p>Web Interface for Medical Image Annotation</p>
        <p><strong>Data Folder:</strong> {data_folder}</p>
    </div>
    """))
    
    # Create the app
    app = SimpleHECKTORApp(data_folder)
    return app

class SimpleHECKTORApp:
    def __init__(self, data_folder):
        self.data_folder = Path(data_folder)
        self.labels_folder = self.data_folder / "labels"
        self.finals_folder = self.data_folder.parent / "finals"
        self.finals_folder.mkdir(exist_ok=True)
        
        self.annotator_id = None
        self.patients = self.get_patients()
        self.current_patient = None
        self.current_data = None
        
        self.create_interface()
    
    def get_patients(self):
        """Get patient list"""
        ct_files = list(self.data_folder.glob("*__CT.nii.gz"))
        patients = []
        
        for ct_file in ct_files:
            patient_id = ct_file.name.replace("__CT.nii.gz", "")
            pt_file = self.data_folder / f"{patient_id}__PT.nii.gz"
            if pt_file.exists():
                patients.append(patient_id)
        
        return sorted(patients)
    
    def create_interface(self):
        """Create the interface"""
        
        # Login widgets
        self.id_input = widgets.Text(
            placeholder='Enter annotator ID',
            description='Annotator ID:'
        )
        
        self.login_btn = widgets.Button(description='Login', button_style='primary')
        self.login_btn.on_click(self.handle_login)
        
        # Patient widgets
        self.patient_dropdown = widgets.Dropdown(
            options=[(f"Patient {p}", p) for p in self.patients],
            description='Patient:'
        )
        self.patient_dropdown.observe(self.on_patient_select, names='value')
        
        # Action buttons
        self.launch_btn = widgets.Button(description='🚀 Launch Napari', button_style='success')
        self.launch_btn.on_click(self.launch_napari)
        
        self.complete_btn = widgets.Button(description='✅ Mark Complete', button_style='info')
        self.complete_btn.on_click(self.mark_complete)
        
        # Status widgets
        self.status = widgets.HTML('')
        self.info = widgets.HTML('')
        
        # Layout
        login_box = widgets.VBox([
            widgets.HTML('<h2>👤 Login</h2>'),
            self.id_input,
            self.login_btn
        ])
        
        main_box = widgets.VBox([
            widgets.HTML('<h2>🏥 HECKTOR Annotation</h2>'),
            self.status,
            widgets.HTML('<h3>Select Patient</h3>'),
            self.patient_dropdown,
            self.info,
            widgets.HTML('<h3>Actions</h3>'),
            widgets.HBox([self.launch_btn, self.complete_btn]),
            widgets.HTML('''
            <div style="background: #f0f8ff; padding: 15px; border-radius: 5px; margin: 20px 0;">
                <h4>📖 Instructions</h4>
                <ol>
                    <li>Select a patient</li>
                    <li>Click "Launch Napari" to annotate</li>
                    <li>Save your work in napari</li>
                    <li>Mark complete when done</li>
                </ol>
            </div>
            ''')
        ])
        
        # Show login first
        self.current_view = login_box
        display(login_box)
        
        # Store main view for later
        self.main_view = main_box
    
    def handle_login(self, btn):
        """Handle login"""
        annotator_id = self.id_input.value.strip()
        
        if len(annotator_id) < 2:
            print("❌ Please enter a valid annotator ID")
            return
        
        self.annotator_id = annotator_id
        
        # Hide login, show main
        self.current_view.close()
        display(self.main_view)
        
        # Update status
        self.status.value = f'<div style="background: #e3f2fd; padding: 10px; border-radius: 5px;"><strong>Logged in as:</strong> {annotator_id}</div>'
        
        print(f"✅ Welcome {annotator_id}! Found {len(self.patients)} patients.")
    
    def on_patient_select(self, change):
        """Handle patient selection"""
        if change['new']:
            self.current_patient = change['new']
            self.load_patient_data(self.current_patient)
    
    def load_patient_data(self, patient_id):
        """Load patient data"""
        ct_file = self.data_folder / f"{patient_id}__CT.nii.gz"
        pt_file = self.data_folder / f"{patient_id}__PT.nii.gz"
        
        try:
            ct_sitk = sitk.ReadImage(str(ct_file))
            ct_array = sitk.GetArrayFromImage(ct_sitk)
            
            pt_sitk = sitk.ReadImage(str(pt_file))
            pt_resampled = sitk.Resample(pt_sitk, ct_sitk, sitk.Transform(), sitk.sitkLinear, 0.0, pt_sitk.GetPixelID())
            pt_array = sitk.GetArrayFromImage(pt_resampled)
            
            self.current_data = {
                'ct': ct_array,
                'pt': pt_array,
                'spacing': ct_sitk.GetSpacing(),
                'origin': ct_sitk.GetOrigin(),
                'direction': ct_sitk.GetDirection()
            }
            
            # Check completion status
            completion_file = self.finals_folder / f"{patient_id}_{self.annotator_id}.nii.gz"
            status = "✅ Completed" if completion_file.exists() else "⏳ Pending"
            
            self.info.value = f'''
            <div style="background: #f8f9fa; padding: 10px; border-radius: 5px;">
                <p><strong>Patient:</strong> {patient_id}</p>
                <p><strong>Status:</strong> {status}</p>
                <p><strong>CT Shape:</strong> {ct_array.shape}</p>
                <p><strong>PT Shape:</strong> {pt_array.shape}</p>
            </div>
            '''
            
            print(f"✅ Loaded {patient_id}")
            
        except Exception as e:
            print(f"❌ Error loading {patient_id}: {e}")
            self.current_data = None
    
    def launch_napari(self, btn):
        """Launch napari for annotation"""
        if not self.current_patient or not self.current_data:
            print("❌ Please select a patient first")
            return
        
        if not self.annotator_id:
            print("❌ Please login first")
            return
        
        try:
            print(f"🚀 Launching napari for {self.current_patient}...")
            
            # Create napari viewer
            viewer = napari.Viewer(title=f"HECKTOR - {self.current_patient} - {self.annotator_id}")
            
            # Get data
            ct_array = self.current_data['ct']
            pt_array = self.current_data['pt']
            spacing = self.current_data['spacing']
            scale = (spacing[2], spacing[1], spacing[0])
            
            # Add images
            viewer.add_image(ct_array, name="CT", scale=scale, colormap='gray')
            viewer.add_image(pt_array, name="PET", scale=scale, colormap='hot', blending='additive', opacity=0.7)
            
            # Load mask
            annotator_mask = self.finals_folder / f"{self.current_patient}_{self.annotator_id}.nii.gz"
            original_mask = self.labels_folder / f"{self.current_patient}.nii.gz"
            
            if annotator_mask.exists():
                mask_sitk = sitk.ReadImage(str(annotator_mask))
                mask_array = sitk.GetArrayFromImage(mask_sitk)
                print("📁 Loaded your existing annotation")
            elif original_mask.exists():
                mask_sitk = sitk.ReadImage(str(original_mask))
                mask_array = sitk.GetArrayFromImage(mask_sitk)
                print("📁 Loaded original annotation")
            else:
                mask_array = np.zeros_like(ct_array, dtype=np.uint8)
                print("📄 Created empty mask")
            
            # Add mask layer
            mask_layer = viewer.add_labels(mask_array, name="Segmentation", scale=scale, opacity=0.5)
            viewer.layers.selection.active = mask_layer
            mask_layer.mode = 'paint'
            mask_layer.brush_size = 10
            mask_layer.selected_label = 1
            
            print(f"✅ Napari launched!")
            print(f"💾 Save your work as: {self.current_patient}_{self.annotator_id}.nii.gz")
            
        except Exception as e:
            print(f"❌ Error launching napari: {e}")
    
    def mark_complete(self, btn):
        """Mark patient as complete"""
        if not self.current_patient or not self.current_data:
            print("❌ Please select a patient first")
            return
        
        if not self.annotator_id:
            print("❌ Please login first")
            return
        
        # Create completion marker file
        completion_file = self.finals_folder / f"{self.current_patient}_{self.annotator_id}.nii.gz"
        
        # Create empty mask as marker
        ct_array = self.current_data['ct']
        empty_mask = np.zeros_like(ct_array, dtype=np.uint8)
        
        mask_sitk = sitk.GetImageFromArray(empty_mask)
        mask_sitk.SetSpacing(self.current_data['spacing'])
        mask_sitk.SetOrigin(self.current_data['origin'])
        mask_sitk.SetDirection(self.current_data['direction'])
        
        sitk.WriteImage(mask_sitk, str(completion_file))
        
        print(f"✅ Marked {self.current_patient} as complete")
        
        # Refresh patient info
        self.load_patient_data(self.current_patient)

if __name__ == "__main__":
    app = start_hecktor_web_app()