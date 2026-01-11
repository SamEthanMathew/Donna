/**
 * RoboEyes Web Implementation
 * Inspired by FluxGarage RoboEyes library
 * Smoothly animated robot eyes for web browsers using SVG
 */

class RoboEyes {
    constructor() {
        this.container = null;
        this.svg = null;
        this.leftEye = null;
        this.rightEye = null;
        
        // Eye dimensions
        this.leftWidth = 80;
        this.leftHeight = 80;
        this.rightWidth = 80;
        this.rightHeight = 80;
        this.leftBorderRadius = 40;
        this.rightBorderRadius = 40;
        this.spaceBetween = 20;
        this.cyclops = false;
        
        // Eye state
        this.leftOpen = true;
        this.rightOpen = true;
        this.mood = 'DEFAULT'; // TIRED, ANGRY, HAPPY, DEFAULT
        this.position = 'DEFAULT'; // N, NE, E, SE, S, SW, W, NW, DEFAULT
        this.curiosity = false;
        this.sweat = false;
        
        // Animation state
        this.hFlicker = { enabled: false, amplitude: 0 };
        this.vFlicker = { enabled: false, amplitude: 0 };
        this.autoBlinker = { enabled: true, interval: 3, variation: 2 };
        this.idleMode = { enabled: true, interval: 5, variation: 3 };
        
        // Animation timers
        this.animationFrame = null;
        this.lastBlink = Date.now();
        this.lastIdle = Date.now();
        this.currentAnimation = null;
        
        // Position offsets (for smooth transitions)
        this.leftX = 0;
        this.leftY = 0;
        this.rightX = 0;
        this.rightY = 0;
        
        // Target positions
        this.targetLeftX = 0;
        this.targetLeftY = 0;
        this.targetRightX = 0;
        this.targetRightY = 0;
    }
    
    /**
     * Initialize eyes in container
     */
    begin(containerId, screenWidth = 800, screenHeight = 400, maxFramerate = 60) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error('[RoboEyes] Container not found:', containerId);
            return false;
        }
        
        this.screenWidth = screenWidth;
        this.screenHeight = screenHeight;
        this.maxFramerate = maxFramerate;
        this.frameInterval = 1000 / maxFramerate;
        this.lastFrameTime = 0;
        
        // Calculate dimensions based on container
        this.calculateDimensions();
        
        // Create SVG
        this.createSVG();
        
        // Start animation loop
        this.startAnimationLoop();
        
        // Start auto-blinker if enabled
        if (this.autoBlinker.enabled) {
            this.scheduleNextBlink();
        }
        
        // Start idle mode if enabled
        if (this.idleMode.enabled) {
            this.scheduleNextIdle();
        }
        
        console.log('[RoboEyes] Initialized');
        return true;
    }
    
    /**
     * Calculate eye dimensions based on container
     */
    calculateDimensions() {
        const rect = this.container.getBoundingClientRect();
        const containerWidth = rect.width || this.container.offsetWidth || 800;
        const containerHeight = rect.height || this.container.offsetHeight || 400;
        
        // Scale eyes to better match screen size
        // Use a larger portion of the container
        const widthScale = containerWidth / 400; // Base width of 400
        const heightScale = containerHeight / 200; // Base height of 200
        const scale = Math.min(widthScale, heightScale) * 0.8; // Use 80% of available space
        
        // Make eyes larger to match screen better
        this.leftWidth = Math.max(60, Math.min(150, 100 * scale));
        this.leftHeight = Math.max(60, Math.min(150, 100 * scale));
        this.rightWidth = Math.max(60, Math.min(150, 100 * scale));
        this.rightHeight = Math.max(60, Math.min(150, 100 * scale));
        this.leftBorderRadius = this.leftWidth / 2;
        this.rightBorderRadius = this.rightWidth / 2;
        this.spaceBetween = Math.max(15, 30 * scale);
        
        // Update SVG viewBox to match container
        this.svgWidth = containerWidth;
        this.svgHeight = containerHeight;
    }
    
    /**
     * Create SVG structure
     */
    createSVG() {
        // Clear container
        this.container.innerHTML = '';
        
        // Create SVG
        this.svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        this.svg.setAttribute('width', '100%');
        this.svg.setAttribute('height', '100%');
        this.svg.setAttribute('viewBox', `0 0 ${this.svgWidth || 800} ${this.svgHeight || 400}`);
        this.svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
        this.svg.style.overflow = 'visible';
        
        // Create groups
        const centerX = (this.svgWidth || 800) / 2;
        const centerY = (this.svgHeight || 400) / 2;
        const eyesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        eyesGroup.setAttribute('class', 'robo-eyes-group');
        eyesGroup.setAttribute('transform', `translate(${centerX}, ${centerY})`); // Center
        
        // Create left eye
        this.leftEye = this.createEye('left');
        eyesGroup.appendChild(this.leftEye);
        
        // Create right eye (or skip if cyclops)
        if (!this.cyclops) {
            this.rightEye = this.createEye('right');
            eyesGroup.appendChild(this.rightEye);
        }
        
        // Create sweat drops group
        const sweatGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        sweatGroup.setAttribute('class', 'sweat-group');
        sweatGroup.setAttribute('id', 'sweatGroup');
        
        this.svg.appendChild(eyesGroup);
        this.svg.appendChild(sweatGroup);
        this.container.appendChild(this.svg);
        
        // Update initial state
        this.updateEyeShapes();
        this.updateMood();
        this.updatePosition();
    }
    
    /**
     * Create a single eye
     */
    createEye(side) {
        const eyeGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        eyeGroup.setAttribute('class', `eye-${side}`);
        eyeGroup.setAttribute('data-side', side);
        
        // Eye shape (ellipse for rounded rectangle effect)
        const eye = document.createElementNS('http://www.w3.org/2000/svg', 'ellipse');
        eye.setAttribute('class', 'eye-shape');
        eye.setAttribute('fill', '#ffffff');
        eye.setAttribute('stroke', '#d4af37');
        eye.setAttribute('stroke-width', '3');
        eyeGroup.appendChild(eye);
        
        // Pupil
        const pupil = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        pupil.setAttribute('class', 'pupil');
        pupil.setAttribute('fill', '#000000');
        pupil.setAttribute('r', '15');
        eyeGroup.appendChild(pupil);
        
        return eyeGroup;
    }
    
    /**
     * Update eye shapes based on current dimensions
     */
    updateEyeShapes() {
        const centerX = (this.svgWidth || 800) / 2;
        const centerY = (this.svgHeight || 400) / 2;
        
        // Left eye
        if (this.leftEye) {
            const eyeShape = this.leftEye.querySelector('.eye-shape');
            const pupil = this.leftEye.querySelector('.pupil');
            
            if (eyeShape) {
                eyeShape.setAttribute('cx', centerX - this.spaceBetween / 2 - this.leftWidth / 2 + this.leftX);
                eyeShape.setAttribute('cy', centerY + this.leftY);
                eyeShape.setAttribute('rx', this.leftWidth / 2);
                eyeShape.setAttribute('ry', this.leftHeight / 2);
            }
            
            if (pupil) {
                pupil.setAttribute('cx', centerX - this.spaceBetween / 2 - this.leftWidth / 2 + this.leftX);
                pupil.setAttribute('cy', centerY + this.leftY);
            }
        }
        
        // Right eye
        if (this.rightEye && !this.cyclops) {
            const eyeShape = this.rightEye.querySelector('.eye-shape');
            const pupil = this.rightEye.querySelector('.pupil');
            
            if (eyeShape) {
                eyeShape.setAttribute('cx', centerX + this.spaceBetween / 2 + this.rightWidth / 2 + this.rightX);
                eyeShape.setAttribute('cy', centerY + this.rightY);
                eyeShape.setAttribute('rx', this.rightWidth / 2);
                eyeShape.setAttribute('ry', this.rightHeight / 2);
            }
            
            if (pupil) {
                pupil.setAttribute('cx', centerX + this.spaceBetween / 2 + this.rightWidth / 2 + this.rightX);
                pupil.setAttribute('cy', centerY + this.rightY);
            }
        }
    }
    
    /**
     * Update mood expression
     */
    updateMood() {
        const moodEffects = {
            'HAPPY': { leftHeight: this.leftHeight * 0.7, rightHeight: this.rightHeight * 0.7, leftY: -10, rightY: -10 },
            'TIRED': { leftHeight: this.leftHeight * 0.5, rightHeight: this.rightHeight * 0.5, leftY: 5, rightY: 5 },
            'ANGRY': { leftHeight: this.leftHeight * 0.6, rightHeight: this.rightHeight * 0.6, leftY: 8, rightY: 8 },
            'DEFAULT': { leftHeight: this.leftHeight, rightHeight: this.rightHeight, leftY: 0, rightY: 0 }
        };
        
        const effect = moodEffects[this.mood] || moodEffects['DEFAULT'];
        
        // Apply smooth transition
        this.animateToTarget({
            leftHeight: effect.leftHeight,
            rightHeight: effect.rightHeight,
            leftY: effect.leftY,
            rightY: effect.rightY
        });
    }
    
    /**
     * Update position (gaze direction)
     */
    updatePosition() {
        const positionOffsets = {
            'N': { leftX: 0, leftY: -30, rightX: 0, rightY: -30 },
            'NE': { leftX: 20, leftY: -20, rightX: 20, rightY: -20 },
            'E': { leftX: 30, leftY: 0, rightX: 30, rightY: 0 },
            'SE': { leftX: 20, leftY: 20, rightX: 20, rightY: 20 },
            'S': { leftX: 0, leftY: 30, rightX: 0, rightY: 30 },
            'SW': { leftX: -20, leftY: 20, rightX: -20, rightY: 20 },
            'W': { leftX: -30, leftY: 0, rightX: -30, rightY: 0 },
            'NW': { leftX: -20, leftY: -20, rightX: -20, rightY: -20 },
            'DEFAULT': { leftX: 0, leftY: 0, rightX: 0, rightY: 0 }
        };
        
        const offset = positionOffsets[this.position] || positionOffsets['DEFAULT'];
        
        // Apply curiosity effect (eyes get taller when looking far left/right)
        let heightMultiplier = 1;
        if (this.curiosity) {
            const horizontalOffset = Math.abs(offset.leftX);
            if (horizontalOffset > 15) {
                heightMultiplier = 1 + (horizontalOffset - 15) / 50;
            }
        }
        
        this.targetLeftX = offset.leftX;
        this.targetLeftY = offset.leftY;
        this.targetRightX = offset.rightX;
        this.targetRightY = offset.rightY * heightMultiplier;
    }
    
    /**
     * Smooth animation to target values
     */
    animateToTarget(targets) {
        // Simple lerp for smooth transitions
        const lerp = (start, end, factor) => start + (end - start) * factor;
        const factor = 0.1; // Smoothing factor
        
        if (targets.leftHeight !== undefined) {
            this.leftHeight = lerp(this.leftHeight, targets.leftHeight, factor);
        }
        if (targets.rightHeight !== undefined) {
            this.rightHeight = lerp(this.rightHeight, targets.rightHeight, factor);
        }
        if (targets.leftY !== undefined) {
            this.leftY = lerp(this.leftY, targets.leftY, factor);
        }
        if (targets.rightY !== undefined) {
            this.rightY = lerp(this.rightY, targets.rightY, factor);
        }
    }
    
    /**
     * Animation loop
     */
    startAnimationLoop() {
        const animate = (currentTime) => {
            // Frame rate limiting
            if (currentTime - this.lastFrameTime >= this.frameInterval) {
                this.update();
                this.lastFrameTime = currentTime;
            }
            
            this.animationFrame = requestAnimationFrame(animate);
        };
        
        this.animationFrame = requestAnimationFrame(animate);
    }
    
    /**
     * Update eyes (called in animation loop)
     */
    update() {
        // Smooth position transitions
        const lerp = (start, end, factor) => start + (end - start) * factor;
        this.leftX = lerp(this.leftX, this.targetLeftX, 0.1);
        this.leftY = lerp(this.leftY, this.targetLeftY, 0.1);
        this.rightX = lerp(this.rightX, this.targetRightX, 0.1);
        this.rightY = lerp(this.rightY, this.targetRightY, 0.1);
        
        // Apply flicker
        if (this.hFlicker.enabled) {
            this.leftX += (Math.random() - 0.5) * this.hFlicker.amplitude;
            this.rightX += (Math.random() - 0.5) * this.hFlicker.amplitude;
        }
        if (this.vFlicker.enabled) {
            this.leftY += (Math.random() - 0.5) * this.vFlicker.amplitude;
            this.rightY += (Math.random() - 0.5) * this.vFlicker.amplitude;
        }
        
        // Update eye shapes
        this.updateEyeShapes();
        
        // Update sweat animation
        if (this.sweat) {
            this.updateSweat();
        }
        
        // Handle window resize
        this.handleResize();
    }
    
    /**
     * Handle window resize
     */
    handleResize() {
        if (!this.container) return;
        
        const rect = this.container.getBoundingClientRect();
        const newWidth = rect.width || this.container.offsetWidth;
        const newHeight = rect.height || this.container.offsetHeight;
        
        // Only recalculate if size changed significantly
        if (Math.abs(newWidth - (this.svgWidth || 800)) > 10 || 
            Math.abs(newHeight - (this.svgHeight || 400)) > 10) {
            this.calculateDimensions();
            if (this.svg) {
                this.svg.setAttribute('viewBox', `0 0 ${this.svgWidth} ${this.svgHeight}`);
            }
        }
    }
    
    /**
     * Draw eyes (same as update but without framerate limit)
     */
    drawEyes() {
        this.update();
    }
    
    /**
     * Set eye widths
     */
    setWidth(leftEye, rightEye) {
        this.leftWidth = leftEye;
        this.rightWidth = rightEye;
        this.updateEyeShapes();
    }
    
    /**
     * Set eye heights
     */
    setHeight(leftEye, rightEye) {
        this.leftHeight = leftEye;
        this.rightHeight = rightEye;
        this.updateEyeShapes();
    }
    
    /**
     * Set border radius
     */
    setBorderradius(leftEye, rightEye) {
        this.leftBorderRadius = leftEye;
        this.rightBorderRadius = rightEye;
        // Note: SVG ellipses don't have border-radius, but we can adjust rx/ry ratio
    }
    
    /**
     * Set space between eyes
     */
    setSpacebetween(space) {
        this.spaceBetween = space;
        this.updateEyeShapes();
    }
    
    /**
     * Set cyclops mode
     */
    setCyclops(on) {
        this.cyclops = on;
        this.createSVG();
    }
    
    /**
     * Set mood
     */
    setMood(mood) {
        this.mood = mood;
        this.updateMood();
    }
    
    /**
     * Set position (gaze direction)
     */
    setPosition(position) {
        this.position = position;
        this.updatePosition();
    }
    
    /**
     * Set curiosity mode
     */
    setCuriosity(on) {
        this.curiosity = on;
        this.updatePosition();
    }
    
    /**
     * Set sweat animation
     */
    setSweat(on) {
        this.sweat = on;
        if (!on) {
            const sweatGroup = document.getElementById('sweatGroup');
            if (sweatGroup) {
                sweatGroup.innerHTML = '';
            }
        }
    }
    
    /**
     * Update sweat drops animation
     */
    updateSweat() {
        const sweatGroup = document.getElementById('sweatGroup');
        if (!sweatGroup) return;
        
        // Create sweat drops occasionally
        if (Math.random() < 0.02) {
            const centerX = (this.svgWidth || 800) / 2;
            const drop = document.createElementNS('http://www.w3.org/2000/svg', 'ellipse');
            drop.setAttribute('cx', centerX + (Math.random() - 0.5) * (this.svgWidth || 800) * 0.3);
            drop.setAttribute('cy', 50);
            drop.setAttribute('rx', '3');
            drop.setAttribute('ry', '5');
            drop.setAttribute('fill', '#d4af37');
            drop.setAttribute('opacity', '0.7');
            drop.setAttribute('data-speed', (0.5 + Math.random() * 0.5).toString());
            sweatGroup.appendChild(drop);
        }
        
        // Animate existing drops
        const drops = sweatGroup.querySelectorAll('ellipse');
        drops.forEach(drop => {
            const currentY = parseFloat(drop.getAttribute('cy'));
            const speed = parseFloat(drop.getAttribute('data-speed'));
            const newY = currentY + speed;
            const maxY = this.svgHeight || 400;
            
            if (newY > maxY) {
                drop.remove();
            } else {
                drop.setAttribute('cy', newY);
                drop.setAttribute('opacity', Math.max(0, 0.7 - (newY - 50) / 350));
            }
        });
    }
    
    /**
     * Open eyes
     */
    open(left = 1, right = 1) {
        if (left) this.leftOpen = true;
        if (right) this.rightOpen = true;
        this.updateEyeOpenness();
    }
    
    /**
     * Close eyes
     */
    close(left = 1, right = 1) {
        if (left) this.leftOpen = false;
        if (right) this.rightOpen = false;
        this.updateEyeOpenness();
    }
    
    /**
     * Update eye openness
     */
    updateEyeOpenness() {
        if (this.leftEye) {
            const eyeShape = this.leftEye.querySelector('.eye-shape');
            if (eyeShape) {
                const currentRy = parseFloat(eyeShape.getAttribute('ry'));
                eyeShape.setAttribute('ry', this.leftOpen ? this.leftHeight / 2 : 2);
            }
        }
        
        if (this.rightEye) {
            const eyeShape = this.rightEye.querySelector('.eye-shape');
            if (eyeShape) {
                const currentRy = parseFloat(eyeShape.getAttribute('ry'));
                eyeShape.setAttribute('ry', this.rightOpen ? this.rightHeight / 2 : 2);
            }
        }
    }
    
    /**
     * Set horizontal flicker
     */
    setHFlicker(on, amplitude) {
        this.hFlicker.enabled = on;
        this.hFlicker.amplitude = amplitude;
    }
    
    /**
     * Set vertical flicker
     */
    setVFlicker(on, amplitude) {
        this.vFlicker.enabled = on;
        this.vFlicker.amplitude = amplitude;
    }
    
    /**
     * Blink animation
     */
    blink(left = 1, right = 1) {
        if (left) this.close(1, 0);
        if (right) this.close(0, 1);
        
        setTimeout(() => {
            if (left) this.open(1, 0);
            if (right) this.open(0, 1);
        }, 150);
    }
    
    /**
     * Confused animation (shake left/right)
     */
    anim_confused() {
        this.currentAnimation = 'confused';
        let shakeCount = 0;
        const maxShakes = 10;
        
        const shake = () => {
            if (shakeCount >= maxShakes) {
                this.currentAnimation = null;
                this.setPosition('DEFAULT');
                return;
            }
            
            const offset = (Math.random() - 0.5) * 20;
            this.targetLeftX = offset;
            this.targetRightX = offset;
            
            shakeCount++;
            setTimeout(shake, 100);
        };
        
        shake();
    }
    
    /**
     * Laugh animation (shake up/down)
     */
    anim_laugh() {
        this.currentAnimation = 'laugh';
        let shakeCount = 0;
        const maxShakes = 10;
        
        const shake = () => {
            if (shakeCount >= maxShakes) {
                this.currentAnimation = null;
                this.setPosition('DEFAULT');
                return;
            }
            
            const offset = (Math.random() - 0.5) * 20;
            this.targetLeftY = offset;
            this.targetRightY = offset;
            
            shakeCount++;
            setTimeout(shake, 100);
        };
        
        shake();
    }
    
    /**
     * Set auto-blinker
     */
    setAutoblinker(on, interval, variation) {
        this.autoBlinker.enabled = on;
        this.autoBlinker.interval = interval;
        this.autoBlinker.variation = variation;
        
        if (on) {
            this.scheduleNextBlink();
        }
    }
    
    /**
     * Schedule next blink
     */
    scheduleNextBlink() {
        if (!this.autoBlinker.enabled) return;
        
        const delay = (this.autoBlinker.interval + 
                      (Math.random() - 0.5) * this.autoBlinker.variation * 2) * 1000;
        
        setTimeout(() => {
            this.blink();
            this.scheduleNextBlink();
        }, delay);
    }
    
    /**
     * Set idle mode
     */
    setIdleMode(on, interval, variation) {
        this.idleMode.enabled = on;
        this.idleMode.interval = interval;
        this.idleMode.variation = variation;
        
        if (on) {
            this.scheduleNextIdle();
        }
    }
    
    /**
     * Schedule next idle movement
     */
    scheduleNextIdle() {
        if (!this.idleMode.enabled) return;
        
        const delay = (this.idleMode.interval + 
                      (Math.random() - 0.5) * this.idleMode.variation * 2) * 1000;
        
        setTimeout(() => {
            const positions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'DEFAULT'];
            const randomPos = positions[Math.floor(Math.random() * positions.length)];
            this.setPosition(randomPos);
            this.scheduleNextIdle();
        }, delay);
    }
    
    /**
     * Set display colors (for compatibility, not really used in web)
     */
    setDisplayColors(background, main) {
        // Web implementation uses CSS, but we can store these values
        this.bgColor = background;
        this.mainColor = main;
    }
    
    /**
     * Cleanup
     */
    destroy() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
        if (this.container) {
            this.container.innerHTML = '';
        }
    }
}

// Export for use
if (typeof window !== 'undefined') {
    window.RoboEyes = RoboEyes;
}
if (typeof module !== 'undefined' && module.exports) {
    module.exports = RoboEyes;
}
