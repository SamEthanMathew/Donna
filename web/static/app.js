/**
 * Donna Web UI - Frontend JavaScript
 * Handles WebSocket communication, emotion control, and UI interactions
 */

class DonnaUI {
    constructor() {
        this.ws = null;
        this.isConnected = false;
        this.currentEmotion = null;
        this.eyes = null;
        this.recognition = null;
        this.isRecording = false;
        this.faceTrackingInterval = null;
        this.webcamFeed = null;
        
        this.init();
    }
    
    init() {
        // Initialize Web-Eye-Animation
        this.initEyes();
        
        // Initialize Web Speech API
        this.initSpeechRecognition();
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Connect WebSocket
        this.connectWebSocket();
        
        // Start face tracking for eye movement (uses server-side camera, no UI display)
        this.startFaceTracking();
        
        // Initialize person modal
        this.initPersonModal();
    }
    
    initPersonModal() {
        if (typeof PersonModal !== 'undefined') {
            this.personModal = new PersonModal();
            
            // Add event listener for "Add Person" button
            const addPersonBtn = document.getElementById('addPersonButton');
            if (addPersonBtn) {
                addPersonBtn.addEventListener('click', () => {
                    this.personModal.open();
                });
            }
        }
    }
    
    initEyes() {
        // Wait for Web-Eye-Animation library to load
        // The library automatically initializes when loaded and creates window.eyes
        if (typeof eyes !== 'undefined') {
            this.eyes = eyes;
            console.log('[UI] Web-Eye-Animation initialized');
        } else {
            // Retry after a short delay
            setTimeout(() => {
                if (typeof eyes !== 'undefined') {
                    this.eyes = eyes;
                    console.log('[UI] Web-Eye-Animation initialized (delayed)');
                } else {
                    console.warn('[UI] Web-Eye-Animation library not loaded');
                }
            }, 500);
        }
    }
    
    // Webcam feed removed from UI - face tracking still works via server API
    
    startFaceTracking() {
        // Poll for face position and update eye gaze
        this.faceTrackingInterval = setInterval(async () => {
            if (!this.eyes) return;
            
            try {
                const response = await fetch('/api/webcam/face-position');
                const data = await response.json();
                
                if (data.detected) {
                    // Convert normalized coordinates (0-1) to screen pixel coordinates
                    const eyeContainer = document.getElementById('eyeContainer');
                    if (eyeContainer) {
                        const rect = eyeContainer.getBoundingClientRect();
                        const x = rect.left + data.x * rect.width;
                        const y = rect.top + data.y * rect.height;
                        
                        // Use eyes.move() to look at face position
                        try {
                            this.eyes.move(x, y);
                        } catch (error) {
                            // Fallback: use target() method if move() doesn't work
                            // Convert to 3D space coordinates
                            const z = 1000; // Distance
                            const screenX = (data.x - 0.5) * 1000;
                            const screenY = (data.y - 0.5) * 1000;
                            this.eyes.target(screenX, screenY, z);
                        }
                    }
                } else {
                    // No face detected - center eyes
                    try {
                        const eyeContainer = document.getElementById('eyeContainer');
                        if (eyeContainer) {
                            const rect = eyeContainer.getBoundingClientRect();
                            const centerX = rect.left + rect.width / 2;
                            const centerY = rect.top + rect.height / 2;
                            this.eyes.move(centerX, centerY);
                        }
                    } catch (error) {
                        // Ignore errors
                    }
                }
            } catch (error) {
                // Silently handle errors (camera might not be available)
            }
        }, 100); // Update every 100ms
    }
    
    stopFaceTracking() {
        if (this.faceTrackingInterval) {
            clearInterval(this.faceTrackingInterval);
            this.faceTrackingInterval = null;
        }
    }
    
    initSpeechRecognition() {
        // Check for Web Speech API support
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        
        if (SpeechRecognition) {
            this.recognition = new SpeechRecognition();
            this.recognition.continuous = false;
            this.recognition.interimResults = false;
            this.recognition.lang = 'en-US';
            
            this.recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                this.handleVoiceInput(transcript);
            };
            
            this.recognition.onerror = (event) => {
                console.error('[UI] Speech recognition error:', event.error);
                this.updateStatus('Speech recognition error: ' + event.error, 'error');
                this.isRecording = false;
                this.updateVoiceButton();
            };
            
            this.recognition.onend = () => {
                this.isRecording = false;
                this.updateVoiceButton();
            };
        } else {
            console.warn('[UI] Web Speech API not supported');
        }
    }
    
    setupEventListeners() {
        const sendButton = document.getElementById('sendButton');
        const textInput = document.getElementById('textInput');
        const voiceButton = document.getElementById('voiceButton');
        
        // Send button
        sendButton.addEventListener('click', () => {
            this.handleTextInput();
        });
        
        // Enter key to send (Shift+Enter for new line)
        textInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleTextInput();
            }
        });
        
        // Voice button
        voiceButton.addEventListener('click', () => {
            this.toggleVoiceRecording();
        });
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            this.isConnected = true;
            this.updateStatus('Connected', 'success');
            console.log('[UI] WebSocket connected');
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
        
        this.ws.onerror = (error) => {
            console.error('[UI] WebSocket error:', error);
            this.updateStatus('Connection error', 'error');
        };
        
        this.ws.onclose = () => {
            this.isConnected = false;
            this.updateStatus('Disconnected', 'error');
            console.log('[UI] WebSocket disconnected');
            
            // Reconnect after 3 seconds
            setTimeout(() => {
                this.connectWebSocket();
            }, 3000);
        };
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'token':
                this.appendToken(data.content);
                break;
                
            case 'emotion':
                this.updateEmotion(data.emotion);
                break;
                
            case 'complete':
                this.finishResponse(data.content, data.emotion);
                break;
                
            case 'person_detected':
                this.handlePersonDetected(data.person, data.greeting);
                break;
                
            case 'error':
                this.updateStatus('Error: ' + data.content, 'error');
                break;
        }
    }
    
    handlePersonDetected(person, greeting) {
        // Display greeting notification
        this.updateStatus(`Recognized: ${person}`, 'success');
        
        // Add greeting to conversation
        if (greeting) {
            this.addMessage(greeting, 'assistant');
        }
        
        // Update emotion to joy for greeting
        this.updateEmotion('joy');
    }
    
    handleTextInput() {
        const textInput = document.getElementById('textInput');
        const message = textInput.value.trim();
        
        if (!message) {
            return;
        }
        
        if (!this.isConnected) {
            this.updateStatus('Not connected. Please wait...', 'error');
            return;
        }
        
        // Add user message to conversation
        this.addMessage(message, 'user');
        
        // Clear input
        textInput.value = '';
        
        // Send via WebSocket
        this.ws.send(JSON.stringify({
            type: 'message',
            content: message
        }));
        
        // Start streaming response
        this.startResponse();
    }
    
    handleVoiceInput(transcript) {
        // Add user message
        this.addMessage(transcript, 'user');
        
        // Send via WebSocket
        if (this.isConnected) {
            this.ws.send(JSON.stringify({
                type: 'message',
                content: transcript
            }));
            this.startResponse();
        } else {
            // Fallback to REST API
            this.sendViaREST(transcript);
        }
    }
    
    async sendViaREST(message) {
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message })
            });
            
            const data = await response.json();
            this.finishResponse(data.response, data.emotion);
        } catch (error) {
            console.error('[UI] REST API error:', error);
            this.updateStatus('Error sending message', 'error');
        }
    }
    
    toggleVoiceRecording() {
        if (!this.recognition) {
            this.updateStatus('Voice recognition not supported in this browser', 'error');
            return;
        }
        
        if (this.isRecording) {
            this.recognition.stop();
            this.isRecording = false;
        } else {
            this.recognition.start();
            this.isRecording = true;
            this.updateStatus('Listening...', 'success');
        }
        
        this.updateVoiceButton();
    }
    
    updateVoiceButton() {
        const voiceButton = document.getElementById('voiceButton');
        if (this.isRecording) {
            voiceButton.classList.add('recording');
        } else {
            voiceButton.classList.remove('recording');
        }
    }
    
    startResponse() {
        // Create assistant message container
        const conversationArea = document.getElementById('conversationArea');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant-message';
        messageDiv.id = 'currentResponse';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = '';
        
        messageDiv.appendChild(contentDiv);
        conversationArea.appendChild(messageDiv);
        
        // Scroll to bottom
        conversationArea.scrollTop = conversationArea.scrollHeight;
    }
    
    appendToken(token) {
        const responseDiv = document.getElementById('currentResponse');
        if (responseDiv) {
            const contentDiv = responseDiv.querySelector('.message-content');
            contentDiv.textContent += token;
            
            // Scroll to bottom
            const conversationArea = document.getElementById('conversationArea');
            conversationArea.scrollTop = conversationArea.scrollHeight;
        }
    }
    
    finishResponse(fullResponse, emotion) {
        // Update final response
        const responseDiv = document.getElementById('currentResponse');
        if (responseDiv) {
            const contentDiv = responseDiv.querySelector('.message-content');
            contentDiv.textContent = fullResponse;
            responseDiv.removeAttribute('id');
        }
        
        // Update emotion
        if (emotion) {
            this.updateEmotion(emotion);
        }
        
        // Clear status
        this.updateStatus('Ready', 'success');
    }
    
    updateEmotion(emotion) {
        if (!this.eyes || !emotion || emotion === this.currentEmotion) {
            return;
        }
        
        this.currentEmotion = emotion;
        
        // Web-Eye-Animation supports these emotions:
        // joy, sadness, surprise, anger, fear, disgust, confusion, love, sleepy, excitement
        const validEmotions = [
            'joy', 'sadness', 'surprise', 'anger', 'fear',
            'disgust', 'confusion', 'love', 'sleepy', 'excitement'
        ];
        
        if (validEmotions.includes(emotion)) {
            try {
                this.eyes.emotion(emotion);
                console.log('[UI] Emotion updated:', emotion);
            } catch (error) {
                console.error('[UI] Error setting emotion:', error);
            }
        } else {
            // Map unknown emotions to closest match
            const emotionMap = {
                'happiness': 'joy',
                'neutral': 'joy'
            };
            const mappedEmotion = emotionMap[emotion] || 'joy';
            try {
                this.eyes.emotion(mappedEmotion);
                console.log('[UI] Emotion mapped:', emotion, '->', mappedEmotion);
            } catch (error) {
                console.error('[UI] Error setting emotion:', error);
            }
        }
    }
    
    addMessage(text, type) {
        const conversationArea = document.getElementById('conversationArea');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = text;
        
        messageDiv.appendChild(contentDiv);
        conversationArea.appendChild(messageDiv);
        
        // Scroll to bottom
        conversationArea.scrollTop = conversationArea.scrollHeight;
    }
    
    updateStatus(message, type = '') {
        const statusDiv = document.getElementById('status');
        statusDiv.textContent = message;
        statusDiv.className = 'status ' + type;
    }
}

// Initialize UI when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Wait for Web-Eye-Animation library to load
    if (typeof eyes !== 'undefined') {
        window.donnaUI = new DonnaUI();
    } else {
        // Retry after a short delay
        setTimeout(() => {
            if (typeof eyes !== 'undefined') {
                window.donnaUI = new DonnaUI();
            } else {
                console.error('[UI] Failed to load Web-Eye-Animation library');
                // Initialize anyway - eyes will fail gracefully
                window.donnaUI = new DonnaUI();
            }
        }, 500);
    }
});

