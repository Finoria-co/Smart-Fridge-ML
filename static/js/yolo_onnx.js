/**
 * In-Browser YOLO11s ONNX Inference Engine for Smart Fridge ML
 * Uses ONNX Runtime Web (WebAssembly/WebGL)
 * Runs 100% Client-Side on GitHub Pages without any Backend/Server!
 */

const YOLO11_CLASSES = [
  'cheese butter', 'milk', 'egg', 'apple', 'avocado', 'banana', 'strawberry', 'mango', 
  'lemon', 'grape', 'orange', 'steak', 'pork', 'chicken duck', 'sausage', 'fish', 
  'shellfish', 'shrimp', 'bread', 'corn', 'tofu', 'eggplant', 'potato', 'garlic', 
  'cauliflower', 'tomato', 'spring onion', 'rape', 'ginger', 'okra', 'lettuce', 
  'pumpkin', 'cucumber', 'white radish', 'carrot', 'asparagus', 'broccoli', 
  'celery stick', 'cilantro mint', 'snow peas', 'cabbage', 'bean sprouts', 'onion', 
  'pepper', 'green beans', 'french beans', 'mushroom', 'beans', 'blueberries', 
  'brussel sprouts', 'melon', 'meat', 'peach', 'plum', 'raspberries', 'salad', 
  'sauce', 'spinach', 'squash', 'watermelon', 'yogurt', 'zucchini', 'bacon', 
  'cream', 'cream cheese', 'flour', 'ham', 'lime', 'olive', 'sugar', 
  'sweet potato', 'sweet basil', 'kaffir lime', 'holy basil', 'wild betel'
];

const ONNX_EN_TO_TH = {
  'pork': 'หมูสับ',
  'chicken duck': 'อกไก่',
  'chicken': 'อกไก่',
  'shrimp': 'กุ้งสด',
  'bacon': 'เบคอน',
  'ham': 'เบคอน',
  'egg': 'ไข่ไก่',
  'rice': 'ข้าวสวย',
  'pasta': 'เส้นพาสต้า',
  'bread': 'ขนมปัง',
  'holy basil': 'ใบกะเพรา',
  'tomato': 'มะเขือเทศ',
  'cheese butter': 'ชีส',
  'cream cheese': 'ชีส',
  'cabbage': 'ผักกาดขาว',
  'wild betel': 'ใบชะพลู',
  'sweet basil': 'ใบโหระพา',
  'kaffir lime': 'ใบมะกรูด',
  'garlic': 'กระเทียม',
  'onion': 'หอมใหญ่',
  'carrot': 'แครอท',
  'potato': 'มันฝรั่ง',
  'sweet potato': 'มันหวาน',
  'mushroom': 'เห็ด',
  'tofu': 'เต้าหู้',
  'fish': 'ปลา',
  'shellfish': 'หอย',
  'sausage': 'ไส้กรอก',
  'steak': 'เนื้อวัว',
  'meat': 'เนื้อวัว',
  'milk': 'นมสด',
  'cream': 'ครีมสด',
  'pepper': 'พริก',
  'spring onion': 'ต้นหอม',
  'lemon': 'มะนาว',
  'lime': 'มะนาว',
  'corn': 'ข้าวโพด',
  'cucumber': 'แตงกวา',
  'pumpkin': 'ฟักทอง',
  'broccoli': 'บรอกโคลี',
  'spinach': 'ผักโขม',
  'lettuce': 'ผักสลัด',
  'ginger': 'ขิง',
  'bean sprouts': 'ถั่วงอก',
  'eggplant': 'มะเขือยาว',
  'apple': 'แอปเปิ้ล',
  'banana': 'กล้วย',
  'mango': 'มะม่วง',
  'avocado': 'อะโวคาโด',
  'strawberry': 'สตรอว์เบอร์รี'
};

let yoloOnnxSession = null;
let isYoloLoading = false;

async function loadYoloOnnxSession(onProgress) {
  if (yoloOnnxSession) return yoloOnnxSession;
  if (typeof ort === 'undefined') {
    throw new Error('ONNX Runtime Web library (ort) not loaded');
  }

  isYoloLoading = true;
  if (onProgress) onProgress('กำลังโหลดโมเดล YOLO11s ONNX เข้าหน่วยความจำเบราว์เซอร์...');

  ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
  ort.env.wasm.simd = true;

  try {
    // โหลดโมเดล static/models/best.onnx (36MB)
    yoloOnnxSession = await ort.InferenceSession.create('static/models/best.onnx', {
      executionProviders: ['wasm']
    });
    console.log('[YOLO-ONNX] โหลดโมเดล YOLO11s ONNX สำเร็จ พร้อมรันบนเบราว์เซอร์!');
    return yoloOnnxSession;
  } finally {
    isYoloLoading = false;
  }
}

/**
 * คำนวณ Intersection over Union (IoU) สำหรับ NMS
 */
function calculateIoU(boxA, boxB) {
  const xA = Math.max(boxA[0], boxB[0]);
  const yA = Math.max(boxA[1], boxB[1]);
  const xB = Math.min(boxA[2], boxB[2]);
  const yB = Math.min(boxA[3], boxB[3]);

  const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
  const boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
  const boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);

  return interArea / (boxAArea + boxBArea - interArea + 1e-6);
}

/**
 * รันการตรวจจับภาพด้วยโมเดล ONNX บนเบราว์เซอร์โดยตรง
 */
async function detectWithOnnxWeb(imageElement, confThreshold = 0.25, iouThreshold = 0.45, onStatus) {
  const session = await loadYoloOnnxSession(onStatus);
  if (onStatus) onStatus('กำลังประมวลผลรูปภาพด้วย YOLO11s ONNX (WebAssembly)...');

  const origW = imageElement.naturalWidth || imageElement.width || 640;
  const origH = imageElement.naturalHeight || imageElement.height || 640;

  // 1. วาดลง Canvas ขนาด 640x640 เพื่อเตรียม Tensor
  const canvas = document.createElement('canvas');
  canvas.width = 640;
  canvas.height = 640;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(imageElement, 0, 0, 640, 640);
  const imgData = ctx.getImageData(0, 0, 640, 640);
  const pixels = imgData.data;

  // 2. แปลงเป็น Float32Array [1, 3, 640, 640] normalized 0.0 - 1.0
  const floatData = new Float32Array(3 * 640 * 640);
  const totalPixels = 640 * 640;
  for (let i = 0; i < totalPixels; i++) {
    floatData[i] = pixels[i * 4] / 255.0;                      // R
    floatData[totalPixels + i] = pixels[i * 4 + 1] / 255.0;    // G
    floatData[totalPixels * 2 + i] = pixels[i * 4 + 2] / 255.0;// B
  }

  const tensor = new ort.Tensor('float32', floatData, [1, 3, 640, 640]);

  // 3. รัน Inference ด้วย WebAssembly
  const feeds = {};
  feeds[session.inputNames[0]] = tensor;
  const results = await session.run(feeds);
  const outputTensor = results[session.outputNames[0]]; // [1, 79, 8400]
  const data = outputTensor.data;
  const numClasses = 75;
  const numAnchors = 8400;

  // 4. สกัด Bounding Boxes และคลาสที่มีความมั่นใจสูงสุด
  const candidates = [];
  for (let col = 0; col < numAnchors; col++) {
    let maxScore = 0;
    let maxClass = -1;

    for (let c = 0; c < numClasses; c++) {
      const score = data[(4 + c) * numAnchors + col];
      if (score > maxScore) {
        maxScore = score;
        maxClass = c;
      }
    }

    if (maxScore >= confThreshold) {
      const cx = data[0 * numAnchors + col];
      const cy = data[1 * numAnchors + col];
      const w = data[2 * numAnchors + col];
      const h = data[3 * numAnchors + col];

      const x1 = Math.max(0, (cx - w / 2) / 640);
      const y1 = Math.max(0, (cy - h / 2) / 640);
      const x2 = Math.min(1, (cx + w / 2) / 640);
      const y2 = Math.min(1, (cy + h / 2) / 640);

      candidates.push({
        classId: maxClass,
        classNameEn: YOLO11_CLASSES[maxClass] || `class_${maxClass}`,
        score: maxScore,
        box: [x1, y1, x2, y2]
      });
    }
  }

  // 5. เรียงลำดับคะแนนจากมากไปน้อย
  candidates.sort((a, b) => b.score - a.score);

  // 6. Non-Maximum Suppression (NMS)
  const selected = [];
  for (const cand of candidates) {
    let keep = true;
    for (const prev of selected) {
      if (cand.classId === prev.classId || calculateIoU(cand.box, prev.box) > iouThreshold) {
        keep = false;
        break;
      }
    }
    if (keep) {
      selected.push(cand);
      if (selected.length >= 8) break; // จำกัดไม่เกิน 8 รายการหลัก
    }
  }

  // 7. จัดรูปแบบผลลัพธ์พร้อมวาดรูป Annotated
  const detections = [];
  const detectedIngredientsSet = new Set();

  selected.forEach(s => {
    const rawEn = s.classNameEn.toLowerCase();
    const thName = ONNX_EN_TO_TH[rawEn] || rawEn;
    detectedIngredientsSet.add(thName);

    const b = s.box;
    detections.push({
      name: thName,
      label: thName,
      en: rawEn,
      confidence: Math.round(s.score * 100),
      box: b,
      x: Math.round(((b[0] + b[2]) / 2) * 100),
      y: Math.round(((b[1] + b[3]) / 2) * 100)
    });
  });

  // 8. วาดรูปพร้อมกรอบ Bounding Box สวยงาม
  const annotCanvas = document.createElement('canvas');
  annotCanvas.width = origW;
  annotCanvas.height = origH;
  const aCtx = annotCanvas.getContext('2d');
  aCtx.drawImage(imageElement, 0, 0, origW, origH);

  const colors = ['#0d9488', '#2563eb', '#d97706', '#db2777', '#7c3aed'];

  detections.forEach((d, idx) => {
    const color = colors[idx % colors.length];
    const bx1 = d.box[0] * origW;
    const by1 = d.box[1] * origH;
    const bx2 = d.box[2] * origW;
    const by2 = d.box[3] * origH;
    const bw = bx2 - bx1;
    const bh = by2 - by1;

    // กรอบสี่เหลี่ยม
    aCtx.strokeStyle = color;
    aCtx.lineWidth = Math.max(3, Math.round(origW * 0.005));
    aCtx.strokeRect(bx1, by1, bw, bh);

    // ป้ายข้อความ
    const text = ` ${d.name} ${d.confidence}% `;
    const fontSize = Math.max(14, Math.round(origW * 0.035));
    aCtx.font = `600 ${fontSize}px Prompt, sans-serif`;
    const textWidth = aCtx.measureText(text).width;
    const tagH = fontSize + 8;
    const tagY = Math.max(0, by1 - tagH);

    aCtx.fillStyle = color;
    aCtx.fillRect(bx1, tagY, textWidth + 8, tagH);

    aCtx.fillStyle = '#ffffff';
    aCtx.fillText(text, bx1 + 2, tagY + fontSize);
  });

  const annotatedUrl = annotCanvas.toDataURL('image/jpeg', 0.9);

  return {
    success: true,
    is_real_model: true,
    is_onnx_web: true,
    message: detections.length > 0
      ? `YOLO11s ONNX ตรวจพบวัตถุดิบ: ${Array.from(detectedIngredientsSet).join(', ')}`
      : 'YOLO11s ONNX ไม่พบวัตถุดิบในภาพ หรือค่าความมั่นใจต่ำกว่า 25%',
    detected_ingredients: Array.from(detectedIngredientsSet),
    detections: detections,
    annotated_image_url: annotatedUrl
  };
}
