/**
 * Person Registration Modal
 * Handles person registration with camera preview
 */

class PersonModal {
    constructor() {
        this.modal = null;
        this.isOpen = false;
        this.cameraStream = null;
        this.videoElement = null;
        this.init();
    }
    
    init() {
        // Create modal HTML
        this.createModal();
        this.setupEventListeners();
    }
    
    createModal() {
        const modalHTML = `
            <div id="personModal" class="person-modal" style="display: none;">
                <div class="person-modal-overlay"></div>
                <div class="person-modal-content">
                    <div class="person-modal-header">
                        <h2>Register New Person</h2>
                        <button class="person-modal-close" id="personModalClose">&times;</button>
                    </div>
                    <div class="person-modal-body">
                        <div class="camera-preview-container">
                            <video id="personModalVideo" autoplay playsinline></video>
                            <div class="camera-preview-placeholder" id="personModalPlaceholder">
                                <p>Camera preview will appear here</p>
                            </div>
                        </div>
                        <div class="person-form">
                            <label for="personName">Name:</label>
                            <input type="text" id="personName" placeholder="Enter person's name" />
                            <div class="person-modal-actions">
                                <button id="personModalCapture" class="btn-capture">Capture & Register</button>
                                <button id="personModalCancel" class="btn-cancel">Cancel</button>
                            </div>
                            <div id="personModalStatus" class="person-modal-status"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        document.body.insertAdjacentHTML('beforeend', modalHTML);
        this.modal = document.getElementById('personModal');
        this.videoElement = document.getElementById('personModalVideo');
    }
    
    setupEventListeners() {
        // Close button
        document.getElementById('personModalClose').addEventListener('click', () => this.close());
        
        // Cancel button
        document.getElementById('personModalCancel').addEventListener('click', () => this.close());
        
        // Overlay click to close
        this.modal.querySelector('.person-modal-overlay').addEventListener('click', () => this.close());
        
        // Capture button
        document.getElementById('personModalCapture').addEventListener('click', () => this.captureAndRegister());
        
        // Enter key on name input
        document.getElementById('personName').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.captureAndRegister();
            }
        });
    }
    
    async open() {
        this.isOpen = true;
        this.modal.style.display = 'flex';
        
        // Start camera preview
        try {
            await this.startCamera();
        } catch (error) {
            console.error('[PersonModal] Camera error:', error);
            this.showStatus('Error accessing camera. Make sure it is not being used by another application.', 'error');
        }
    }
    
    close() {
        this.isOpen = false;
        this.modal.style.display = 'none';
        this.stopCamera();
        this.clearForm();
    }
    
    async startCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: 'user' }
            });
            
            this.cameraStream = stream;
            this.videoElement.srcObject = stream;
            this.videoElement.style.display = 'block';
            document.getElementById('personModalPlaceholder').style.display = 'none';
        } catch (error) {
            console.error('[PersonModal] Failed to start camera:', error);
            throw error;
        }
    }
    
    stopCamera() {
        if (this.cameraStream) {
            this.cameraStream.getTracks().forEach(track => track.stop());
            this.cameraStream = null;
        }
        if (this.videoElement) {
            this.videoElement.srcObject = null;
            this.videoElement.style.display = 'none';
            document.getElementById('personModalPlaceholder').style.display = 'block';
        }
    }
    
    clearForm() {
        document.getElementById('personName').value = '';
        this.showStatus('', '');
    }
    
    async captureAndRegister() {
        const nameInput = document.getElementById('personName');
        const name = nameInput.value.trim();
        
        if (!name) {
            this.showStatus('Please enter a name', 'error');
            return;
        }
        
        // Disable button during registration
        const captureBtn = document.getElementById('personModalCapture');
        captureBtn.disabled = true;
        captureBtn.textContent = 'Registering...';
        
        try {
            // Use the server's camera frame (not the modal preview)
            // The server will capture from its own camera
            const response = await fetch('/api/person/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ name })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showStatus(`Successfully registered ${name}!`, 'success');
                setTimeout(() => {
                    this.close();
                }, 1500);
            } else {
                this.showStatus(data.message || 'Registration failed', 'error');
            }
        } catch (error) {
            console.error('[PersonModal] Registration error:', error);
            this.showStatus('Error registering person. Please try again.', 'error');
        } finally {
            captureBtn.disabled = false;
            captureBtn.textContent = 'Capture & Register';
        }
    }
    
    showStatus(message, type) {
        const statusEl = document.getElementById('personModalStatus');
        statusEl.textContent = message;
        statusEl.className = `person-modal-status ${type}`;
    }
}

// Export for use
if (typeof window !== 'undefined') {
    window.PersonModal = PersonModal;
}

