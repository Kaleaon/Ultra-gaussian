/**
 * WebNN Image Classifier using ONNX Runtime Web
 * Uses MobileNetV2 for architecture/landscape classification
 * Runs on NPU when available, falls back to GPU/CPU
 */

import * as ort from 'onnxruntime-web';

// ImageNet class labels for architecture/landscape detection
const ARCHITECTURE_CLASSES = new Set([
  'church', 'castle', 'monastery', 'palace', 'bell_cote', 'dome', 
  'triumphal_arch', 'suspension_bridge', 'steel_arch_bridge', 'viaduct',
  'pier', 'beacon', 'lighthouse', 'mobile_home', 'barn', 'greenhouse',
  'cinema', 'home_theater', 'library', 'boathouse', 'confectionery',
  'bookshop', 'tobacco_shop', 'toyshop', 'shoe_shop', 'barbershop',
  'bakery', 'butcher_shop', 'grocery_store', 'prison', 'planetarium',
]);

const LANDSCAPE_CLASSES = new Set([
  'cliff', 'valley', 'volcano', 'lakeside', 'seashore', 'sandbar',
  'promontory', 'alp', 'geyser', 'coral_reef', 'dam', 'breakwater',
]);

const PEOPLE_VEHICLE_CLASSES = new Set([
  'person', 'face', 'head', 'convertible', 'sports_car', 'racer',
  'cab', 'jeep', 'limousine', 'minivan', 'ambulance', 'fire_engine',
  'garbage_truck', 'pickup', 'trailer_truck', 'moving_van', 'police_van',
  'recreational_vehicle', 'streetcar', 'trolleybus', 'minibus', 'school_bus',
  'bicycle', 'motorcycle', 'moped', 'motor_scooter', 'go-kart',
]);

class WebNNClassifier {
  constructor() {
    this.session = null;
    this.isInitialized = false;
    this.backendType = 'cpu';
    this.labels = [];
  }

  /**
   * Initialize ONNX Runtime session with WebNN or fallback
   */
  async initialize() {
    if (this.isInitialized) return;

    try {
      // Try WebNN with NPU first
      const webnnOptions = {
        executionProviders: [{
          name: 'webnn',
          deviceType: 'npu',
          powerPreference: 'default'
        }],
      };

      // Check if WebNN is supported
      if ('ml' in navigator) {
        try {
          this.session = await ort.InferenceSession.create(
            '/models/mobilenetv2.onnx',
            webnnOptions
          );
          this.backendType = 'npu';
          console.log('WebNN NPU backend initialized');
        } catch (npuError) {
          console.log('NPU not available, trying GPU...');
          // Try GPU
          webnnOptions.executionProviders[0].deviceType = 'gpu';
          try {
            this.session = await ort.InferenceSession.create(
              '/models/mobilenetv2.onnx',
              webnnOptions
            );
            this.backendType = 'gpu';
            console.log('WebNN GPU backend initialized');
          } catch (gpuError) {
            throw new Error('WebNN not available');
          }
        }
      } else {
        throw new Error('WebNN not supported');
      }
    } catch (webnnError) {
      console.log('WebNN not available, falling back to WebAssembly');
      // Fallback to WebAssembly (CPU)
      try {
        this.session = await ort.InferenceSession.create(
          '/models/mobilenetv2.onnx',
          { executionProviders: ['wasm'] }
        );
        this.backendType = 'cpu';
        console.log('WASM CPU backend initialized');
      } catch (wasmError) {
        console.error('Failed to initialize any backend:', wasmError);
        this.session = null;
      }
    }

    this.isInitialized = true;
  }

  /**
   * Preprocess image for MobileNetV2 (224x224, normalized)
   */
  preprocessImage(imageElement) {
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d');
    
    // Draw and resize image
    ctx.drawImage(imageElement, 0, 0, 224, 224);
    const imageData = ctx.getImageData(0, 0, 224, 224);
    const data = imageData.data;
    
    // Convert to float32 and normalize (ImageNet mean/std)
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    
    const float32Data = new Float32Array(3 * 224 * 224);
    
    for (let i = 0; i < 224 * 224; i++) {
      const r = data[i * 4] / 255.0;
      const g = data[i * 4 + 1] / 255.0;
      const b = data[i * 4 + 2] / 255.0;
      
      // NCHW format
      float32Data[i] = (r - mean[0]) / std[0];           // R channel
      float32Data[224 * 224 + i] = (g - mean[1]) / std[1]; // G channel
      float32Data[2 * 224 * 224 + i] = (b - mean[2]) / std[2]; // B channel
    }
    
    return new ort.Tensor('float32', float32Data, [1, 3, 224, 224]);
  }

  /**
   * Classify an image
   * @param {HTMLImageElement|Blob|string} image - Image to classify
   * @returns {Promise<{classification: string, confidence: number, topClasses: Array}>}
   */
  async classify(image) {
    if (!this.session) {
      // Return a default classification if model not loaded
      return {
        classification: 'scene',
        confidence: 0.5,
        topClasses: [],
        backend: 'none',
        error: 'Model not initialized'
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

      // Preprocess
      const inputTensor = this.preprocessImage(imgElement);
      
      // Run inference
      const feeds = { input: inputTensor };
      const results = await this.session.run(feeds);
      
      // Get output (assuming output name is 'output')
      const outputName = this.session.outputNames[0];
      const output = results[outputName].data;
      
      // Apply softmax and get top-5
      const probabilities = this.softmax(Array.from(output));
      const topIndices = this.getTopK(probabilities, 5);
      
      // Classify based on detected classes
      let classification = 'scene';
      let maxConfidence = 0;
      
      for (const { index, prob } of topIndices) {
        const className = this.getClassName(index);
        
        if (ARCHITECTURE_CLASSES.has(className)) {
          if (prob > maxConfidence) {
            classification = 'architecture';
            maxConfidence = prob;
          }
        } else if (LANDSCAPE_CLASSES.has(className)) {
          if (prob > maxConfidence) {
            classification = 'landscape';
            maxConfidence = prob;
          }
        } else if (PEOPLE_VEHICLE_CLASSES.has(className)) {
          classification = 'filtered';
          maxConfidence = prob;
          break; // Immediately filter if people/vehicles detected
        }
      }
      
      return {
        classification,
        confidence: maxConfidence || topIndices[0]?.prob || 0.5,
        topClasses: topIndices.map(({ index, prob }) => ({
          class: this.getClassName(index),
          probability: prob
        })),
        backend: this.backendType
      };
    } catch (error) {
      console.error('Classification error:', error);
      return {
        classification: 'scene',
        confidence: 0.5,
        topClasses: [],
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

  getClassName(index) {
    // Simplified - return index as string if labels not loaded
    // In production, load ImageNet labels
    return `class_${index}`;
  }

  getBackendInfo() {
    return {
      type: this.backendType,
      isInitialized: this.isInitialized,
      hasNPU: this.backendType === 'npu',
      hasGPU: this.backendType === 'gpu' || this.backendType === 'npu',
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

export default WebNNClassifier;
