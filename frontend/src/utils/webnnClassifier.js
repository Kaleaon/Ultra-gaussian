/**
 * WebNN Image Classifier for NPU/TPU Acceleration
 * 
 * Optimized for Google Pixel devices with Tensor G3/G4/G5 TPU
 * Uses ONNX Runtime Web with WebNN backend for hardware acceleration
 * 
 * On Pixel devices:
 * - TPU access via Android Neural Networks API → LiteRT → WebNN
 * - Chrome flag required: chrome://flags → "WebNN API" → Enabled
 * - Falls back to GPU → CPU if NPU unavailable
 */

import * as ort from 'onnxruntime-web';

// Architecture/Landscape detection classes (ImageNet subset)
const ARCHITECTURE_INDICES = new Set([
  497, // church
  498, // cinema  
  483, // castle
  536, // dock
  538, // dome
  627, // library
  663, // monastery
  668, // mosque
  698, // palace
  727, // pier
  762, // prison
  832, // stone_wall
  833, // stupa
  838, // suspension_bridge
  855, // thatch
  862, // tile_roof
  878, // triumphal_arch
  900, // water_tower
  909, // wishing_well
]);

const LANDSCAPE_INDICES = new Set([
  970, // alp
  971, // cliff
  972, // coral_reef
  973, // geyser
  974, // lakeside
  975, // promontory
  976, // sandbar
  977, // seashore
  978, // valley
  979, // volcano
  334, // dam
  417, // breakwater
]);

const PEOPLE_VEHICLE_INDICES = new Set([
  // Vehicles
  407, // ambulance
  436, // beach_wagon
  468, // cab
  511, // convertible
  555, // fire_engine
  569, // garbage_truck
  573, // go-kart
  609, // jeep
  627, // limousine
  654, // minibus
  656, // minivan
  665, // moped
  670, // mountain_bike
  675, // motor_scooter
  717, // pickup
  734, // police_van
  751, // racer
  779, // school_bus
  817, // sports_car
  820, // streetcar
  864, // tow_truck
  867, // trailer_truck
  874, // trolleybus
]);

class WebNNClassifier {
  constructor() {
    this.session = null;
    this.isInitialized = false;
    this.backendType = 'unknown';
    this.deviceInfo = null;
    this.initPromise = null;
    this.labels = [];
  }

  /**
   * Load ImageNet labels
   */
  async loadLabels() {
    try {
      const response = await fetch('/models/imagenet_labels.json');
      this.labels = await response.json();
      console.log(`Loaded ${this.labels.length} ImageNet labels`);
    } catch (error) {
      console.warn('Could not load ImageNet labels:', error);
      this.labels = [];
    }
  }

  /**
   * Detect device capabilities and WebNN support
   */
  async detectCapabilities() {
    const capabilities = {
      webnn: false,
      webgpu: false,
      isPixel: false,
      tensorVersion: null,
      npuAvailable: false,
      gpuAvailable: false,
    };

    // Detect if running on Google Pixel
    const ua = navigator.userAgent;
    if (ua.includes('Pixel')) {
      capabilities.isPixel = true;
      // Extract Pixel model for TPU version detection
      const pixelMatch = ua.match(/Pixel\s*(\d+)/i);
      if (pixelMatch) {
        const pixelVersion = parseInt(pixelMatch[1]);
        if (pixelVersion >= 8) capabilities.tensorVersion = 'G3+';
        else if (pixelVersion >= 6) capabilities.tensorVersion = 'G1+';
      }
    }

    // Check WebNN availability
    if ('ml' in navigator) {
      capabilities.webnn = true;
      
      // Test NPU availability (Pixel TPU accessed as NPU via Android NNAPI)
      try {
        const npuContext = await navigator.ml.createContext({ deviceType: 'npu' });
        if (npuContext) {
          capabilities.npuAvailable = true;
        }
      } catch (e) {
        console.log('NPU not available:', e.message);
      }

      // Test GPU availability
      try {
        const gpuContext = await navigator.ml.createContext({ deviceType: 'gpu' });
        if (gpuContext) {
          capabilities.gpuAvailable = true;
        }
      } catch (e) {
        console.log('WebNN GPU not available:', e.message);
      }
    }

    // Check WebGPU
    if (navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          capabilities.webgpu = true;
          const info = await adapter.requestAdapterInfo?.();
          capabilities.gpuInfo = info;
        }
      } catch (e) {
        console.log('WebGPU not available');
      }
    }

    this.deviceInfo = capabilities;
    return capabilities;
  }

  /**
   * Initialize ONNX Runtime session with optimal backend for device
   * Priority: NPU (Pixel TPU) → GPU → WebGPU → WASM (CPU)
   */
  async initialize() {
    if (this.initPromise) return this.initPromise;
    if (this.isInitialized) return;

    this.initPromise = this._doInitialize();
    return this.initPromise;
  }

  async _doInitialize() {
    await this.detectCapabilities();
    await this.loadLabels();
    
    const modelUrl = '/models/mobilenetv2-12.onnx';
    
    // Try backends in order of preference for Pixel devices
    const backends = [];
    
    if (this.deviceInfo.npuAvailable) {
      // NPU - best for Pixel TPU
      backends.push({
        name: 'webnn-npu',
        options: {
          executionProviders: [{
            name: 'webnn',
            deviceType: 'npu',
            powerPreference: 'high-performance',
            numThreads: 4,
          }],
        }
      });
    }
    
    if (this.deviceInfo.gpuAvailable || this.deviceInfo.webnn) {
      // WebNN GPU
      backends.push({
        name: 'webnn-gpu',
        options: {
          executionProviders: [{
            name: 'webnn',
            deviceType: 'gpu',
            powerPreference: 'high-performance',
          }],
        }
      });
    }
    
    if (this.deviceInfo.webgpu) {
      // WebGPU
      backends.push({
        name: 'webgpu',
        options: {
          executionProviders: ['webgpu'],
        }
      });
    }
    
    // WASM CPU fallback (always available)
    backends.push({
      name: 'wasm-cpu',
      options: {
        executionProviders: ['wasm'],
      }
    });

    // Try each backend
    for (const backend of backends) {
      try {
        console.log(`Trying ${backend.name} backend...`);
        this.session = await ort.InferenceSession.create(modelUrl, backend.options);
        this.backendType = backend.name;
        console.log(`✓ Initialized with ${backend.name} backend`);
        break;
      } catch (error) {
        console.warn(`${backend.name} failed:`, error.message);
      }
    }

    if (!this.session) {
      console.error('All backends failed, classifier unavailable');
      this.backendType = 'none';
    }

    this.isInitialized = true;
  }

  /**
   * Preprocess image for MobileNetV2 (224x224, ImageNet normalization)
   */
  preprocessImage(imageElement) {
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d');
    
    // Draw and resize
    ctx.drawImage(imageElement, 0, 0, 224, 224);
    const imageData = ctx.getImageData(0, 0, 224, 224);
    const data = imageData.data;
    
    // ImageNet normalization
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    
    // NCHW format for MobileNetV2
    const float32Data = new Float32Array(3 * 224 * 224);
    
    for (let i = 0; i < 224 * 224; i++) {
      const r = data[i * 4] / 255.0;
      const g = data[i * 4 + 1] / 255.0;
      const b = data[i * 4 + 2] / 255.0;
      
      float32Data[i] = (r - mean[0]) / std[0];
      float32Data[224 * 224 + i] = (g - mean[1]) / std[1];
      float32Data[2 * 224 * 224 + i] = (b - mean[2]) / std[2];
    }
    
    return new ort.Tensor('float32', float32Data, [1, 3, 224, 224]);
  }

  /**
   * Classify an image for architecture/landscape content
   */
  async classify(image) {
    if (!this.isInitialized) {
      await this.initialize();
    }

    if (!this.session) {
      return {
        classification: 'scene',
        confidence: 0.5,
        topClasses: [],
        backend: 'none',
        error: 'Model not loaded - check /models/mobilenetv2-12.onnx exists'
      };
    }

    try {
      // Load image if needed
      let imgElement = image;
      if (typeof image === 'string') {
        imgElement = await this.loadImage(image);
      } else if (image instanceof Blob) {
        imgElement = await this.loadImageFromBlob(image);
      }

      const startTime = performance.now();
      
      // Preprocess and run inference
      const inputTensor = this.preprocessImage(imgElement);
      const feeds = {};
      feeds[this.session.inputNames[0]] = inputTensor;
      
      const results = await this.session.run(feeds);
      const output = results[this.session.outputNames[0]].data;
      
      const inferenceTime = performance.now() - startTime;
      
      // Get predictions
      const probabilities = this.softmax(Array.from(output));
      const topK = this.getTopK(probabilities, 10);
      
      // Classify based on detected content
      let classification = 'scene';
      let maxConfidence = 0;
      let reason = 'general scene';
      
      for (const { index, prob } of topK) {
        if (PEOPLE_VEHICLE_INDICES.has(index)) {
          classification = 'filtered';
          maxConfidence = prob;
          reason = 'contains people or vehicles';
          break;
        } else if (ARCHITECTURE_INDICES.has(index) && prob > maxConfidence) {
          classification = 'architecture';
          maxConfidence = prob;
          reason = 'architectural features detected';
        } else if (LANDSCAPE_INDICES.has(index) && prob > maxConfidence) {
          classification = 'landscape';
          maxConfidence = prob;
          reason = 'landscape features detected';
        }
      }
      
      if (maxConfidence === 0) {
        maxConfidence = topK[0]?.prob || 0.5;
      }

      return {
        classification,
        confidence: maxConfidence,
        reason,
        topClasses: topK.slice(0, 5).map(({ index, prob }) => ({
          classIndex: index,
          className: this.labels[index] || `class_${index}`,
          probability: prob
        })),
        backend: this.backendType,
        inferenceTimeMs: inferenceTime,
        deviceInfo: {
          isPixel: this.deviceInfo?.isPixel,
          tensorVersion: this.deviceInfo?.tensorVersion,
          npuAvailable: this.deviceInfo?.npuAvailable,
        }
      };
    } catch (error) {
      console.error('Classification error:', error);
      return {
        classification: 'scene',
        confidence: 0.5,
        backend: this.backendType,
        error: error.message
      };
    }
  }

  loadImage(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = url;
    });
  }

  loadImageFromBlob(blob) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        URL.revokeObjectURL(img.src);
        resolve(img);
      };
      img.onerror = reject;
      img.src = URL.createObjectURL(blob);
    });
  }

  softmax(arr) {
    const max = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
  }

  getTopK(arr, k) {
    const indexed = arr.map((val, idx) => ({ index: idx, prob: val }));
    indexed.sort((a, b) => b.prob - a.prob);
    return indexed.slice(0, k);
  }

  getBackendInfo() {
    return {
      type: this.backendType,
      isInitialized: this.isInitialized,
      device: this.deviceInfo,
      hasNPU: this.backendType.includes('npu'),
      hasGPU: this.backendType.includes('gpu'),
      isPixelOptimized: this.deviceInfo?.isPixel && this.backendType.includes('webnn'),
    };
  }
}

// Singleton instance
let classifierInstance = null;

export async function getClassifier() {
  if (!classifierInstance) {
    classifierInstance = new WebNNClassifier();
    await classifierInstance.initialize();
  }
  return classifierInstance;
}

export async function classifyImage(image) {
  const classifier = await getClassifier();
  return classifier.classify(image);
}

export async function getBackendInfo() {
  const classifier = await getClassifier();
  return classifier.getBackendInfo();
}

export async function detectDeviceCapabilities() {
  const classifier = new WebNNClassifier();
  return classifier.detectCapabilities();
}

export default WebNNClassifier;
