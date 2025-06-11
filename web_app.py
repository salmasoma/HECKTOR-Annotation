# web_app.py - Easy web wrapper for your napari app
# Save this file next to your main.py

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import glob
import re
import numpy as np
import SimpleITK as sitk
import json
import sys

# Import your existing napari app backend logic
# We'll reuse the core functions from your main.py
sys.path.append('.')

app = Flask(__name__)
CORS(app)

# Simple HTML template (embedded in Python file for easiness)
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>HECKTOR Annotation Tool</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        .header { background: #2c3e50; color: white; padding: 15px; margin: -20px -20px 20px -20px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .btn { background: #3498db; color: white; border: none; padding: 10px 20px; margin: 5px; border-radius: 4px; cursor: pointer; }
        .btn:hover { background: #2980b9; }
        .btn-success { background: #27ae60; }
        .btn-danger { background: #e74c3c; }
        .btn-warning { background: #f39c12; }
        .patient-list { max-height: 300px; overflow-y: auto; border: 1px solid #ddd; }
        .patient-item { padding: 10px; border-bottom: 1px solid #eee; cursor: pointer; display: flex; justify-content: space-between; }
        .patient-item:hover { background: #f8f9fa; }
        .patient-item.active { background: #3498db; color: white; }
        .patient-item.completed { background: #d5f4e6; }
        .status { font-weight: bold; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .status.success { background: #d4edda; color: #155724; }
        .status.error { background: #f8d7da; color: #721c24; }
        .progress-bar { width: 100%; height: 20px; background: #eee; border-radius: 10px; overflow: hidden; }
        .progress-fill { height: 100%; background: #27ae60; transition: width 0.3s; }
        .form-group { margin: 15px 0; }
        .form-group label { display: block; margin-bottom: 5px; font-weight: bold; }
        .form-group input, .form-group select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 HECKTOR Annotation Tool - Web Interface</h1>
            <p>Simple web interface for your napari annotation tool</p>
        </div>

        <!-- Login Section -->
        <div id="loginSection" class="section">
            <h3>👤 Annotator Login</h3>
            <div class="form-group">
                <label>Enter your Annotator ID:</label>
                <input type="text" id="annotatorId" placeholder="e.g., radiologist_01, student_jane">
            </div>
            <button onclick="login()" class="btn">Start Annotation Session</button>
        </div>

        <!-- Main Interface (hidden initially) -->
        <div id="mainInterface" style="display: none;">
            <!-- Status -->
            <div id="statusDiv" class="status" style="display: none;"></div>
            
            <!-- Annotator Info -->
            <div class="section">
                <h3>Current Annotator: <span id="currentAnnotator">-</span></h3>
                <div class="progress-bar">
                    <div id="progressFill" class="progress-fill" style="width: 0%;"></div>
                </div>
                <p id="progressText">Progress: 0/0 (0%)</p>
            </div>

            <!-- Patient Selection -->
            <div class="section">
                <h3>📋 Patient Selection</h3>
                <p><strong>Current Patient:</strong> <span id="currentPatient">None selected</span></p>
                <div id="patientList" class="patient-list">
                    <!-- Patients loaded here -->
                </div>
                <div style="margin-top: 10px;">
                    <button onclick="previousPatient()" class="btn">⬅️ Previous</button>
                    <button onclick="nextPatient()" class="btn">➡️ Next</button>
                </div>
            </div>

            <!-- Quick Actions -->
            <div class="section">
                <h3>🔧 Quick Actions</h3>
                <button onclick="openNapari()" class="btn btn-success">🖥️ Open in Napari (Desktop)</button>
                <button onclick="saveWork()" class="btn btn-warning">💾 Save Current Work</button>
                <button onclick="refreshStatus()" class="btn">🔄 Refresh Status</button>
            </div>

            <!-- File Operations -->
            <div class="section">
                <h3>📁 File Operations</h3>
                <button onclick="listFiles()" class="btn">📋 List Saved Files</button>
                <button onclick="downloadResults()" class="btn">⬇️ Download Results</button>
                <div id="fileList" style="margin-top: 10px;"></div>
            </div>

            <!-- Instructions -->
            <div class="section">
                <h3>📖 Instructions</h3>
                <div style="background: #f0f8ff; padding: 15px; border-radius: 5px; font-size: 14px;">
                    <p><strong>How to use this web interface:</strong></p>
                    <ol>
                        <li>Enter your annotator ID and click "Start Annotation Session"</li>
                        <li>Select a patient from the list</li>
                        <li>Click "Open in Napari" to launch the desktop annotation tool</li>
                        <li>Perform your annotations in the desktop app</li>
                        <li>Files are automatically saved with your annotator ID</li>
                        <li>Use this interface to track progress and manage files</li>
                    </ol>
                    <p><strong>File naming:</strong> PatientID_YourID.nii.gz</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentAnnotator = null;
        let patients = [];
        let currentPatientIndex = -1;

        function showStatus(message, type = 'success') {
            const statusDiv = document.getElementById('statusDiv');
            statusDiv.textContent = message;
            statusDiv.className = `status ${type}`;
            statusDiv.style.display = 'block';
            setTimeout(() => statusDiv.style.display = 'none', 3000);
        }

        async function login() {
            const annotatorId = document.getElementById('annotatorId').value.trim();
            
            if (!annotatorId || annotatorId.length < 2) {
                showStatus('Please enter a valid annotator ID (at least 2 characters)', 'error');
                return;
            }

            try {
                const response = await fetch('/api/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ annotator_id: annotatorId })
                });

                const data = await response.json();
                
                if (data.success) {
                    currentAnnotator = annotatorId;
                    patients = data.patients;
                    
                    document.getElementById('loginSection').style.display = 'none';
                    document.getElementById('mainInterface').style.display = 'block';
                    document.getElementById('currentAnnotator').textContent = annotatorId;
                    
                    updatePatientList();
                    updateProgress();
                    showStatus(`Welcome, ${annotatorId}! Found ${patients.length} patients.`);
                } else {
                    showStatus('Login failed: ' + data.error, 'error');
                }
            } catch (error) {
                showStatus('Login failed: ' + error.message, 'error');
            }
        }

        function updatePatientList() {
            const patientList = document.getElementById('patientList');
            patientList.innerHTML = '';
            
            patients.forEach((patient, index) => {
                const div = document.createElement('div');
                div.className = `patient-item ${patient.completed ? 'completed' : ''}`;
                div.innerHTML = `
                    <span>${patient.id}</span>
                    <span>${patient.completed ? '✅ Completed' : '⏳ Pending'}</span>
                `;
                div.onclick = () => selectPatient(index);
                patientList.appendChild(div);
            });
        }

        function updateProgress() {
            const completed = patients.filter(p => p.completed).length;
            const total = patients.length;
            const percentage = total > 0 ? Math.round((completed / total) * 100) : 0;
            
            document.getElementById('progressFill').style.width = `${percentage}%`;
            document.getElementById('progressText').textContent = 
                `Progress: ${completed}/${total} (${percentage}%)`;
        }

        function selectPatient(index) {
            currentPatientIndex = index;
            const patient = patients[index];
            
            document.getElementById('currentPatient').textContent = patient.id;
            
            // Update visual selection
            document.querySelectorAll('.patient-item').forEach((item, i) => {
                item.classList.toggle('active', i === index);
            });
            
            showStatus(`Selected patient: ${patient.id}`);
        }

        async function openNapari() {
            if (currentPatientIndex === -1) {
                showStatus('Please select a patient first', 'error');
                return;
            }

            const patient = patients[currentPatientIndex];
            
            try {
                const response = await fetch('/api/open_napari', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        patient_id: patient.id, 
                        annotator_id: currentAnnotator 
                    })
                });

                const data = await response.json();
                
                if (data.success) {
                    showStatus(`Opening ${patient.id} in Napari desktop app...`);
                } else {
                    showStatus('Failed to open Napari: ' + data.error, 'error');
                }
            } catch (error) {
                showStatus('Failed to open Napari: ' + error.message, 'error');
            }
        }

        async function saveWork() {
            showStatus('Work is automatically saved in the desktop app!');
        }

        async function refreshStatus() {
            if (!currentAnnotator) return;
            
            try {
                const response = await fetch(`/api/status/${currentAnnotator}`);
                const data = await response.json();
                
                patients = data.patients;
                updatePatientList();
                updateProgress();
                showStatus('Status refreshed!');
            } catch (error) {
                showStatus('Failed to refresh: ' + error.message, 'error');
            }
        }

        async function listFiles() {
            if (!currentAnnotator) return;
            
            try {
                const response = await fetch(`/api/files/${currentAnnotator}`);
                const data = await response.json();
                
                const fileList = document.getElementById('fileList');
                if (data.files.length === 0) {
                    fileList.innerHTML = '<p>No saved files found.</p>';
                } else {
                    fileList.innerHTML = '<h4>Your saved files:</h4><ul>' + 
                        data.files.map(file => `<li>${file}</li>`).join('') + '</ul>';
                }
            } catch (error) {
                showStatus('Failed to list files: ' + error.message, 'error');
            }
        }

        function downloadResults() {
            if (!currentAnnotator) return;
            window.open(`/api/download/${currentAnnotator}`, '_blank');
        }

        function previousPatient() {
            if (currentPatientIndex > 0) {
                selectPatient(currentPatientIndex - 1);
            }
        }

        function nextPatient() {
            if (currentPatientIndex < patients.length - 1) {
                selectPatient(currentPatientIndex + 1);
            }
        }
    </script>
</body>
</html>
'''

class SimpleHECKTORBackend:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.labels_folder = os.path.join(data_folder, "labels")
        self.finals_folder = os.path.join(os.path.dirname(data_folder), "finals")
        
        # Create finals folder if it doesn't exist
        if not os.path.exists(self.finals_folder):
            os.makedirs(self.finals_folder)
        
        self.patients = self._get_patients()

    def _get_patients(self):
        """Get list of patient IDs"""
        ct_files = glob.glob(os.path.join(self.data_folder, "*__CT.nii.gz"))
        patient_ids = []
        pattern = r"(.+)__CT\.nii\.gz"
        
        for ct_file in ct_files:
            match = re.search(pattern, os.path.basename(ct_file))
            if match:
                patient_id = match.group(1)
                pt_file = os.path.join(self.data_folder, f"{patient_id}__PT.nii.gz")
                if os.path.exists(pt_file):
                    patient_ids.append(patient_id)
        
        return sorted(patient_ids)

    def get_patient_list(self, annotator_id):
        """Get patient list with completion status"""
        completed_patients = self._get_completed_patients(annotator_id)
        
        patient_list = []
        for patient_id in self.patients:
            patient_info = {
                'id': patient_id,
                'completed': patient_id in completed_patients
            }
            patient_list.append(patient_info)
        
        return patient_list

    def _get_completed_patients(self, annotator_id):
        """Get completed patients for specific annotator"""
        if not os.path.exists(self.finals_folder):
            return set()
        
        pattern = f"*_{annotator_id}.nii.gz"
        completed_files = glob.glob(os.path.join(self.finals_folder, pattern))
        completed_ids = set()
        
        for file_path in completed_files:
            filename = os.path.basename(file_path)
            patient_id = filename.replace(f'_{annotator_id}.nii.gz', '')
            completed_ids.add(patient_id)
        
        return completed_ids

    def get_saved_files(self, annotator_id):
        """Get list of saved files for annotator"""
        pattern = f"*_{annotator_id}.nii.gz"
        files = glob.glob(os.path.join(self.finals_folder, pattern))
        return [os.path.basename(f) for f in files]

    def launch_napari_for_patient(self, patient_id, annotator_id):
        """Launch napari for specific patient (simplified)"""
        try:
            # Import your main napari app
            # You'll need to modify your main.py to accept command line arguments
            import subprocess
            import sys
            
            # Launch your napari app with specific patient
            cmd = [sys.executable, "main.py", "--patient", patient_id, "--annotator", annotator_id]
            subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
            
            return True
        except Exception as e:
            print(f"Error launching napari: {e}")
            return False

# Initialize backend (CHANGE THIS PATH TO YOUR DATA FOLDER)
DATA_FOLDER = "./test/"  # ⚠️ CHANGE THIS TO YOUR DATA PATH
backend = SimpleHECKTORBackend(DATA_FOLDER)

# Routes
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    annotator_id = data.get('annotator_id')
    
    if not annotator_id or len(annotator_id) < 2:
        return jsonify({'error': 'Invalid annotator ID'}), 400
    
    patients = backend.get_patient_list(annotator_id)
    
    return jsonify({
        'success': True,
        'patients': patients,
        'total_patients': len(backend.patients)
    })

@app.route('/api/open_napari', methods=['POST'])
def open_napari():
    data = request.json
    patient_id = data.get('patient_id')
    annotator_id = data.get('annotator_id')
    
    success = backend.launch_napari_for_patient(patient_id, annotator_id)
    
    if success:
        return jsonify({'success': True})
    else:
        return jsonify({'error': 'Failed to launch napari'}), 500

@app.route('/api/status/<annotator_id>')
def get_status(annotator_id):
    patients = backend.get_patient_list(annotator_id)
    return jsonify({'patients': patients})

@app.route('/api/files/<annotator_id>')
def get_files(annotator_id):
    files = backend.get_saved_files(annotator_id)
    return jsonify({'files': files})

@app.route('/api/download/<annotator_id>')
def download_files(annotator_id):
    # Simple file download - you can enhance this
    files = backend.get_saved_files(annotator_id)
    return jsonify({'files': files, 'message': 'Check the finals folder on the server'})

if __name__ == '__main__':
    import socket
    
    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except:
            return "127.0.0.1"
    
    local_ip = get_local_ip()
    port = 5050
    
    print("=" * 60)
    print("🏥 HECKTOR Web Interface Starting...")
    print("=" * 60)
    print(f"📍 Local access:     http://localhost:{port}")
    print(f"🌐 Network access:   http://{local_ip}:{port}")
    print("=" * 60)
    print("🚀 Next steps:")
    print("1. Open the web interface in your browser")
    print("2. Use ngrok for internet access:")
    print("   → Download ngrok from https://ngrok.com")
    print("   → Run: ngrok http 5000")
    print("   → Share the https URL with your team")
    print("=" * 60)
    
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)