const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  HeadingLevel, AlignmentType, WidthType, ShadingType, BorderStyle, LevelFormat,
  Footer, PageNumber, TableOfContents, ExternalHyperlink, PageBreak,
} = require("docx");

const REPO = "C:/Users/patim/Project/Finoria/smart-fridge-cv";
const OUT = process.argv[2];
const FONT = "Tahoma";
const MONO = "Consolas";
const W = 9026; // A4 width minus 1" margins (DXA)
const ACCENT = "1F4E79";
const LIGHT = "DCE6F1";
const WARN = "FFF4CE";

// ---------- helpers ----------
const t = (text, o = {}) => new TextRun({ text, font: FONT, size: 21, sizeComplexScript: o.size || 21, ...o });
const P = (children, o = {}) => new Paragraph({
  children: typeof children === "string" ? [t(children)] : children,
  spacing: { after: 100, line: 300 }, ...o,
});
const H1 = (s) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(s)] });
const H2 = (s) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(s)] });
const H3 = (s) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(s)] });
const B = (children, level = 0) => new Paragraph({
  numbering: { reference: "bullets", level },
  children: typeof children === "string" ? [t(children)] : children,
  spacing: { after: 60, line: 290 },
});
const N = (children, ref = "steps") => new Paragraph({
  numbering: { reference: ref, level: 0 },
  children: typeof children === "string" ? [t(children)] : children,
  spacing: { after: 60, line: 290 },
});
const MONOF = { ascii: MONO, hAnsi: MONO, cs: FONT, eastAsia: FONT };
// แยกช่วงภาษาไทยออกมาใช้ฟอนต์ไทย (ฟอนต์ mono ไม่มีอักษรไทย Word จะ fallback ตัวเล็กมาก)
const splitThai = (s) => s.split(/([\u0E00-\u0E7F]+(?:[ ~.\-]*[\u0E00-\u0E7F0-9]+)*)/).filter((x) => x !== "");
const isThai = (x) => /[\u0E00-\u0E7F]/.test(x);
const monoRuns = (s, size, color) => splitThai(s).map((x) => new TextRun(isThai(x)
  ? { text: x, font: FONT, size: size + 1, color }
  : { text: x, font: MONO, size, color }));
const code = (s) => monoRuns(s, 19, "7A2E0E");
const bold = (s) => t(s, { bold: true });
const link = (text, url) => new ExternalHyperlink({ link: url, children: [t(text, { color: "0563C1", underline: {} })] });

function codeBlock(lines) {
  const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
  return new Table({
    width: { size: W, type: WidthType.DXA },
    columnWidths: [W],
    rows: [new TableRow({ cantSplit: true, children: [new TableCell({
      width: { size: W, type: WidthType.DXA },
      shading: { fill: "F3F3F3", type: ShadingType.CLEAR, color: "auto" },
      borders: { top: border, bottom: border, left: border, right: border },
      margins: { top: 100, bottom: 100, left: 160, right: 160 },
      children: lines.map((l) => new Paragraph({
        spacing: { after: 0, line: 260 },
        children: l === "" ? [new TextRun({ text: " ", font: MONO, size: 18 })] : monoRuns(l, 18, "000000"),
      })),
    })] })],
  });
}

function callout(title, lines, fill = WARN) {
  const border = { style: BorderStyle.SINGLE, size: 6, color: "C9A227" };
  return new Table({
    width: { size: W, type: WidthType.DXA },
    columnWidths: [W],
    rows: [new TableRow({ children: [new TableCell({
      width: { size: W, type: WidthType.DXA },
      shading: { fill, type: ShadingType.CLEAR, color: "auto" },
      borders: { top: border, bottom: border, left: border, right: border },
      margins: { top: 100, bottom: 100, left: 180, right: 180 },
      children: [P([bold(title)], { spacing: { after: 60 } }),
        ...lines.map((l) => P(typeof l === "string" ? [t(l)] : l, { spacing: { after: 40, line: 280 } }))],
    })] })],
  });
}

function table(headers, rows, widths) {
  const sum = widths.reduce((a, b) => a + b, 0);
  if (sum !== W) throw new Error(`widths sum ${sum} != ${W}`);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cell = (content, i, head) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    borders,
    shading: head ? { fill: ACCENT, type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: (Array.isArray(content) && content.length && content[0] instanceof Paragraph ? content :
      [new Paragraph({
        spacing: { after: 0, line: 260 },
        children: (Array.isArray(content) ? content : [String(content)]).map((c) =>
          typeof c === "string" ? t(c, head ? { bold: true, color: "FFFFFF", size: 19 } : { size: 19 }) : c),
      })]),
  });
  return new Table({
    width: { size: W, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, i, false)) })),
    ],
  });
}
const c = (s) => code(s); // code-styled cell
const gap = () => new Paragraph({ spacing: { after: 80 }, children: [] });

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), data: b };
}
function image(rel, widthPx, caption) {
  const { w, h, data } = pngSize(path.join(REPO, rel));
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 40 },
      children: [new ImageRun({ type: "png", data, transformation: { width: widthPx, height: Math.round(widthPx * h / w) },
        altText: { title: caption, description: caption, name: path.basename(rel) } })] }),
    P([t(caption, { italics: true, size: 18, color: "595959" }), t("  (" + rel + ")", { size: 17, color: "7F7F7F", font: MONO })],
      { alignment: AlignmentType.CENTER, spacing: { after: 160 } }),
  ];
}

// ---------- content ----------
const children = [];

// Cover
children.push(
  new Paragraph({ spacing: { before: 1800, after: 120 }, children: [t("Smart Fridge Recipe Recommender", { size: 28, color: "595959" })] }),
  new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun("คู่มือส่งต่องาน Computer Vision")] }),
  new Paragraph({ spacing: { after: 360 }, children: [t("คนที่ 2: Computer Vision ML → ส่งต่อให้คนที่ 3, 4, 5, 6", { size: 26, color: ACCENT })] }),
  table(["รายการ", "ค่า"], [
    ["Repo", [link("github.com/Finoria-co/smart-fridge-cv", "https://github.com/Finoria-co/smart-fridge-cv")]],
    ["Branch / commit ล่าสุดตอนเขียนเอกสาร", c("main @ 8414f20")],
    ["โมเดล (best.pt)", [link("Release v1.0-yolo11s", "https://github.com/Finoria-co/smart-fridge-cv/releases/tag/v1.0-yolo11s")]],
    ["ฟังก์ชันที่ส่งมอบ", c("predict.py → predict_image()")],
    ["ผลบน test (วัดครั้งเดียว)", "mAP50 0.643 · mAP50-95 0.525 (1,427 รูป)"],
    ["วันที่", "18 พฤศจิกายน 2026"],
  ], [3200, 5826]),
  gap(),
  P([t("ทุกตัวเลขในเอกสารนี้มาจากการรันจริง บันทึกไว้ใน ", { size: 19, color: "595959" }), ...code("experiments.md"),
    t(" และไฟล์ใน ", { size: 19, color: "595959" }), ...code("results/")]),
  new Paragraph({ children: [new PageBreak()] }),
);

// TOC
children.push(
  new Paragraph({ spacing: { after: 120 }, children: [t("สารบัญ", { bold: true, size: 30, color: ACCENT })] }),
  new TableOfContents("สารบัญ", { hyperlink: true, headingStyleRange: "1-2" }),
  P([t("(ถ้าสารบัญยังว่าง: คลิกขวาที่สารบัญ → Update Field)", { size: 17, color: "7F7F7F", italics: true })]),
  new Paragraph({ children: [new PageBreak()] }),
);

// 1. Quick start
children.push(
  H1("1. อ่านส่วนนี้ก่อน: ใครต้องทำอะไรต่อ"),
  P("ส่วน Computer Vision ทำครบตามแผนกลุ่มแล้ว (เทรน · วัดผล · ฟังก์ชัน predict_image · ไฟล์ best.pt) งานต่อจากนี้อยู่ที่เพื่อนแต่ละคน ตารางนี้สรุปว่าใครต้องทำอะไรและเปิดบทไหน"),
  table(["ใคร", "ต้องทำอะไรต่อ", "ดูบท"], [
    ["คนที่ 6 Web & Integration", "ดาวน์โหลด best.pt → เรียก predict_image() ใน app.py → ส่งรายชื่อวัตถุดิบต่อให้โมเดลของคนที่ 5 → แสดงกรอบบนหน้าเว็บ (ไม่บังคับ)", "3, 4, 5"],
    ["คนที่ 5 Core Recipe ML", "รับ list ชื่อวัตถุดิบมาตรฐานจาก predict_image() (รวมกับของคนที่ 4) แปลงเป็นเวกเตอร์ 0/1 ตาม 12 ชนิด", "4, 5, 6"],
    ["คนที่ 4 Text / NLP", "rice และ pasta ตรวจจากรูปไม่ได้ ต้องได้จากข้อความพิมพ์ · ใช้ชื่อมาตรฐานชุดเดียวกัน", "5, 6"],
    ["คนที่ 3 Recipe Data", "เช็คว่า cabbage ในสูตรหมายถึงผักกาดขาวหรือกะหล่ำปลี (โมเดลรู้จักแค่กะหล่ำปลี) แล้วแจ้งกลับ", "6"],
    ["คนรวมสไลด์ / คนที่ 2", "สไลด์หน้า 5 (การเทรนโมเดล + กราฟ) ใช้ข้อมูลจากบท 7 และ 8", "7, 8"],
    ["ถ้ามีคนต้องเทรน CV ต่อ", "อ่านกติกาและขั้นตอนก่อน (ห้ามใช้ test ซ้ำ)", "9, 10"],
  ], [2300, 5426, 1300]),
  gap(),
  callout("⚠️ 3 เรื่องที่ทั้งทีมต้องรู้", [
    [bold("1. rice และ pasta ตรวจจากรูปไม่ได้ "), t("เพราะไม่มีในรูปที่ใช้เทรน ต้องรับจากข้อความพิมพ์เท่านั้น")],
    [bold("2. cabbage ที่โมเดลรู้จักคือกะหล่ำปลีหัวกลม "), t("ไม่ใช่ผักกาดขาว ถ้าผู้ใช้ถ่ายผักกาดขาวอาจไม่เจอ")],
    [bold("3. pork ในรูปอาหารสุกบนจานยังแม่นน้อย "), t("(mAP50 0.28) แต่หมูดิบดีกว่ามาก (0.56) ระบบนี้เน้นวัตถุดิบดิบในตู้เย็นจึงพอใช้ได้")],
  ]),
);

// 2. What was done
children.push(
  H1("2. สรุปสิ่งที่ทำไปแล้ว"),
  P("งานตามแผนกลุ่มของคนที่ 2: เทรนโมเดลตรวจจับรูป วัดผล (mAP, Confusion Matrix, Loss Curve) และเขียนฟังก์ชันรับรูปแล้วคืนชื่อวัตถุดิบ"),
  table(["ขั้น", "ทำอะไร", "ผล", "commit"], [
    ["เตรียมข้อมูล", "แบ่ง test จาก val แบบ stratified (seed 42)", "train 10,055 / val 1,959 / test 1,317 · 75 class", c("c3df66b")],
    ["เตรียมเครื่อง", "เทรนบน Google Colab (L4 GPU) + WSL2 ในเครื่องสำหรับวัดผล", "notebook + train.py", c("33ac167")],
    ["Baseline", "YOLO11n 50 epoch", "val mAP50 0.510 / mAP50-95 0.409", c("daa7cee")],
    ["วิเคราะห์ข้อผิดพลาด", "confusion matrix + ดูรูปจริง", "pork เป็นอาหารสุก สับสนกับ steak 32%", c("7db5d18")],
    ["Dataset v2", "เพิ่มรูปวัตถุดิบดิบ 709 รูปจาก Roboflow (หมู กุ้ง แฮม เบคอน ฯลฯ)", "train 10,545 / val 2,068 / test 1,427", c("7d2240f")],
    ["เทรนบน v2", "YOLO11n บน v2", "val วัตถุดิบดิบ mAP50-95 0.392 → 0.674", c("7c12244")],
    ["โมเดลใหญ่ขึ้น", "YOLO11s บน v2 → เลือกเป็นโมเดลสุดท้าย", "val mAP50-95 0.436 → 0.486 ความเร็วเท่าเดิม", c("4de6a5e")],
    ["ฟังก์ชันส่งมอบ", "predict_image() + ตาราง 12 ชนิด + unit test", "รับ path/bytes/PIL/numpy", c("7eeb6b2")],
    ["ผลสุดท้าย", "เลือก conf จาก val แล้ววัด test ครั้งเดียว", "test mAP50 0.643 / mAP50-95 0.525", c("4c9b14f")],
    ["ส่งมอบ", "best.pt ขึ้น GitHub Release v1.0-yolo11s", "19.2 MB", c("8414f20")],
  ], [1500, 3326, 3000, 1200]),
);

// 3. Deliverables
children.push(
  H1("3. ไฟล์ที่ส่งมอบ: ใช้ไฟล์ไหนทำอะไร"),
  H2("3.1 ไฟล์ที่คนอื่นต้องใช้"),
  table(["ไฟล์ (ใน repo)", "ใช้ทำอะไร", "ใครใช้"], [
    [c("predict.py"), "ฟังก์ชัน predict_image() รับรูป คืนรายการวัตถุดิบ + CLI สำหรับลองเร็วๆ", "คนที่ 6, 5"],
    [c("ingredient_map.json"), "ตารางแปลง class ของโมเดล (75) → ชื่อมาตรฐาน 12 ชนิด + ชื่อไทย", "คนที่ 6, 5, 4"],
    [c("weights/best.pt"), "ไฟล์โมเดล ไม่อยู่ใน git ต้องดาวน์โหลดจาก Release v1.0-yolo11s", "คนที่ 6"],
    [c("README.md"), "วิธีติดตั้งและเรียกใช้แบบย่อ", "ทุกคน"],
    [c("tests/test_predict.py"), "ทดสอบตารางแปลงชื่อและการรวมผล (ไม่ต้องมี GPU)", "คนที่ 6 (ถ้าแก้ไฟล์)"],
    [c("requirements.txt"), "package ที่ต้องติดตั้ง (ใช้แค่ predict: pip install ultralytics พอ)", "คนที่ 6"],
  ], [2700, 4826, 1500]),
  H2("3.2 ไฟล์ผลการทดลอง (ใช้ทำสไลด์)"),
  table(["ไฟล์ / โฟลเดอร์", "เนื้อหา"], [
    [c("experiments.md"), "บันทึกการทดลองทุกครั้ง: config ตัวเลข การวิเคราะห์ (แหล่งตัวเลขหลัก)"],
    [c("results/final/summary_test.csv"), "ผลบน test ครั้งเดียว (ทั้งชุด / ส่วนเดิม / ส่วนวัตถุดิบดิบ)"],
    [c("results/final/per_class_test.csv"), "mAP บน test ราย class"],
    [c("results/final/test_confusion_matrix_normalized.png"), "confusion matrix บน test"],
    [c("results/final/test_BoxPR_curve.png"), "PR curve บน test"],
    [c("results/final/val_F1_curve.png, conf_from_val.csv"), "F1-confidence curve บน val ใช้เลือก conf 0.28"],
    [c("results/yolo11s_v2/loss_train_vs_val.png"), "loss curve train vs val ของโมเดลสุดท้าย"],
    [c("results/baseline_n*/ , results/yolo11s_v2*/"), "ผลของแต่ละการทดลอง (ใช้ทำตารางเทียบ)"],
    [c("results/inspect/"), "รูปตัวอย่างที่ใช้วิเคราะห์ (หมูถูกทายเป็นสเต็ก, ชะพลู, cabbage)"],
    [c("data/sources.md"), "ที่มาและ license ของ dataset (ต้องใส่เครดิตในสไลด์)"],
  ], [4300, 4726]),
  H2("3.3 ไฟล์สำหรับคนที่จะเทรนต่อ"),
  table(["ไฟล์", "ใช้ทำอะไร"], [
    [c("notebooks/colab_train.ipynb"), "เทรนบน Colab (เซลล์ 1–5 เทรน · Resume · 6 ดูผล · 7 ตัด runtime)"],
    [c("scripts/train.py"), "สคริปต์เทรน (seed 0, deterministic) ใช้ทั้ง Colab และเครื่อง"],
    [c("scripts/build_dataset_v2.py"), "สร้าง dataset v2 จาก Roboflow (ต้องมี API key ของตัวเอง)"],
    [c("scripts/analyze_run.py"), "สรุป run: copy กราฟ + กราฟ loss train vs val"],
    [c("scripts/eval_per_class.py"), "mAP ราย class บน val (หรือ yaml อื่นด้วย --data)"],
    [c("scripts/top_confusions.py"), "class ไหนถูกทายเป็นอะไรบ่อยที่สุด"],
    [c("scripts/final_test.py"), "เลือก conf จาก val และวัด test ครั้งเดียว (ใช้ไปแล้ว)"],
    [c("PLAN.md, CLAUDE.md"), "แผนงานและข้อมูลเครื่อง/Drive/กติกา"],
  ], [3400, 5626]),
);

// 4. Integration guide
children.push(
  H1("4. คู่มือสำหรับคนที่ 6: เชื่อม predict_image() เข้าเว็บ"),
  H2("4.1 ติดตั้ง"),
  N([t("Clone repo (หรือ copy แค่ "), ...code("predict.py"), t(" กับ "), ...code("ingredient_map.json"), t(" ไปไว้โฟลเดอร์เดียวกับ "), ...code("app.py"), t(")")]),
  N([t("ดาวน์โหลด "), ...code("best.pt"), t(" จาก "), link("Release v1.0-yolo11s", "https://github.com/Finoria-co/smart-fridge-cv/releases/tag/v1.0-yolo11s"),
    t(" แล้ววางที่ "), ...code("weights/best.pt"), t(" (ข้าง predict.py) หรือตั้ง env "), ...code("SMART_FRIDGE_WEIGHTS"), t(" ชี้ไปที่ไฟล์")]),
  N([t("ติดตั้ง "), ...code("pip install ultralytics"), t(" (มี torch, opencv, pillow มาด้วย · เครื่องไม่มี GPU ก็ใช้ได้)")]),
  N([t("ลองจาก command line: "), ...code("python predict.py รูปตู้เย็น.jpg"), t(" ต้องพิมพ์ JSON ออกมา")]),
  codeBlock([
    "git clone https://github.com/Finoria-co/smart-fridge-cv.git",
    "cd smart-fridge-cv",
    "pip install ultralytics",
    "mkdir weights",
    "gh release download v1.0-yolo11s -p best.pt -D weights   # หรือกดดาวน์โหลดจากหน้าเว็บ",
    "python predict.py fridge.jpg",
  ]),
  H2("4.2 เรียกใช้ในโค้ด"),
  codeBlock([
    "from predict import predict_image",
    "",
    "items = predict_image(\"fridge.jpg\")              # path",
    "items = predict_image(file_bytes)                 # bytes ของไฟล์ที่อัปโหลด",
    "items = predict_image(pil_image)                  # PIL.Image",
    "items = predict_image(cv2.imread(\"fridge.jpg\"))   # numpy (BGR)",
    "",
    "ingredients = [it[\"ingredient\"] for it in items] # ['chicken', 'tomato', 'cheese']",
  ]),
  H2("4.3 ตัวอย่าง FastAPI (Flask ใช้หลักเดียวกัน)"),
  codeBlock([
    "from fastapi import FastAPI, UploadFile",
    "from predict import predict_image, load_model",
    "",
    "app = FastAPI()",
    "load_model()   # โหลดโมเดลครั้งเดียวตอนเปิด server (ครั้งแรกใช้ ~5-15 วินาที)",
    "",
    "@app.post(\"/detect\")",
    "async def detect(image: UploadFile):",
    "    items = predict_image(await image.read())",
    "    return {\"ingredients\": [it[\"ingredient\"] for it in items], \"items\": items}",
  ]),
  P([t("Flask: "), ...code("items = predict_image(request.files[\"image\"].read())")]),
  H2("4.4 พารามิเตอร์"),
  table(["พารามิเตอร์", "ค่าเริ่มต้น", "ความหมาย"], [
    [c("image"), "–", "path (str/Path), bytes, PIL.Image หรือ numpy BGR"],
    [c("conf"), c("0.28"), "ความมั่นใจขั้นต่ำ เลือกจาก F1 curve บน val · เพิ่ม (เช่น 0.4) = ทายน้อยลงแต่แม่นขึ้น · ลด = เจอมากขึ้นแต่ผิดมากขึ้น"],
    [c("only_contract"), c("True"), "True = คืนเฉพาะ 12 ชนิดของกลุ่ม · False = คืนวัตถุดิบอื่นที่โมเดลรู้จักด้วย (รวม 75 class)"],
    [c("weights"), c("None"), "path ของ best.pt ถ้าไม่ได้วางที่ weights/best.pt"],
    [c("imgsz"), c("640"), "ขนาดรูปที่ป้อนโมเดล (ตรงกับตอนเทรน ไม่ควรเปลี่ยน)"],
  ], [1900, 1500, 5626]),
  H2("4.5 รูปแบบผลลัพธ์"),
  P("คืน list ของ dict เรียงตาม confidence มากไปน้อย ถ้าไม่เจออะไรเลยคืน [] (ไม่ error)"),
  codeBlock([
    "[",
    "  {\"ingredient\": \"tomato\", \"name_th\": \"มะเขือเทศ\", \"in_contract\": true,",
    "   \"confidence\": 0.924, \"count\": 1, \"boxes\": [[319.3, 119.1, 458.6, 258.9]]},",
    "  {\"ingredient\": \"bread\",  \"name_th\": \"ขนมปัง\",    \"in_contract\": true,",
    "   \"confidence\": 0.328, \"count\": 1, \"boxes\": [[54.0, 129.3, 172.7, 270.6]]}",
    "]",
  ]),
  table(["key", "ชนิด", "ความหมาย"], [
    [c("ingredient"), "str", "ชื่อมาตรฐาน (12 ชนิด) เช่น pork, holy_basil · ถ้า only_contract=False อาจเป็นชื่อ class อื่น เช่น spring_onion"],
    [c("name_th"), "str / null", "ชื่อไทยตาม Data Contract · null ถ้าไม่ใช่ 12 ชนิด"],
    [c("in_contract"), "bool", "อยู่ใน 12 ชนิดของกลุ่มหรือไม่"],
    [c("confidence"), "float", "ความมั่นใจสูงสุดของวัตถุดิบนี้ในรูป (0–1)"],
    [c("count"), "int", "จำนวนชิ้นที่เจอ (นับจากกรอบ)"],
    [c("boxes"), "list", "กรอบ [x1, y1, x2, y2] หน่วย pixel ของรูปต้นฉบับ ใช้วาดบนหน้าเว็บได้"],
  ], [1900, 1300, 5826]),
  H2("4.6 ความเร็ว (วัดจริงในเครื่อง i5-14500HX + RTX 4050)"),
  table(["สถานการณ์", "เวลา"], [
    ["เรียกครั้งแรก (โหลดโมเดล + เตรียม GPU)", "5–15 วินาที → ให้เรียก load_model() ตอนเปิด server"],
    ["ครั้งต่อไป GPU (รวมอ่านไฟล์และรวมผล)", "ประมาณ 22 ms ต่อรูป"],
    ["ครั้งต่อไป CPU อย่างเดียว", "ประมาณ 70 ms ต่อรูป (20 threads) · เครื่องที่ช้ากว่าจะนานขึ้น แต่ยังพอสำหรับ demo"],
  ], [4300, 4726]),
  H2("4.7 ปัญหาที่อาจเจอ"),
  table(["อาการ", "สาเหตุ / วิธีแก้"], [
    [c("FileNotFoundError: ไม่พบ weights"), "ยังไม่ได้วาง best.pt ที่ weights/best.pt หรือ env SMART_FRIDGE_WEIGHTS ชี้ผิด"],
    [c("ModuleNotFoundError: ultralytics"), "ยังไม่ได้ pip install ultralytics ใน environment ที่รัน server"],
    ["คืน [] ทั้งที่มีของในรูป", "ของชิ้นนั้นอาจไม่ใช่ 12 ชนิด (ลอง only_contract=False) หรือความมั่นใจต่ำกว่า conf (ลอง conf=0.15)"],
    ["เจอ bacon ทั้งที่เป็นแฮม", "ถูกต้อง: Data Contract รวมเบคอนและแฮมเป็น bacon"],
    ["ไม่เจอข้าวหรือพาสต้า", "ปกติ: โมเดลตรวจสองอย่างนี้ไม่ได้ ใช้ข้อความพิมพ์แทน"],
  ], [3600, 5426]),
);

// 5. Contract
children.push(
  H1("5. Data Contract 12 ชนิด: โมเดลแปลงชื่อยังไง"),
  P([t("โมเดลรู้จัก 75 class จาก dataset แต่กลุ่มตกลงใช้ 12 ชนิด ไฟล์ "), ...code("ingredient_map.json"), t(" แปลงดังนี้ (ผลบน test ใช้บอกความแม่นของแต่ละชนิด)")]),
  table(["ชื่อมาตรฐาน", "ชื่อไทย", "class ในโมเดล", "test mAP50", "test mAP50-95", "mAP50 รูปดิบ"], [
    [c("pork"), "หมูสับ", "pork", "0.278", "0.195", "0.563"],
    [c("chicken"), "อกไก่", "chicken duck", "0.584", "0.504", "0.973"],
    [c("shrimp"), "กุ้งสด", "shrimp", "0.636", "0.442", "0.995"],
    [c("bacon"), "เบคอน/แฮม", "bacon + ham", "0.490 / 0.824", "0.401 / 0.469", "0.558 / 0.989"],
    [c("egg"), "ไข่ไก่", "egg", "0.855", "0.649", "0.995"],
    [c("rice"), "ข้าวสวย", "– (ไม่มี)", "ตรวจไม่ได้", "–", "–"],
    [c("pasta"), "เส้นพาสต้า", "– (ไม่มี)", "ตรวจไม่ได้", "–", "–"],
    [c("bread"), "ขนมปัง", "bread", "0.575", "0.455", "0.995"],
    [c("holy_basil"), "ใบกะเพรา", "holy basil", "0.620", "0.568", "–"],
    [c("tomato"), "มะเขือเทศ", "tomato", "0.791", "0.640", "0.995"],
    [c("cheese"), "ชีส", "cheese butter + cream cheese", "0.621 / 0.939", "0.478 / 0.698", "0.995 / –"],
    [c("cabbage"), "ผักกาดขาว", "cabbage (= กะหล่ำปลี)", "0.598", "0.594", "–"],
  ], [1450, 1350, 2226, 1400, 1400, 1200]),
  P([t("test mAP = test ทั้งชุด 1,427 รูป · รูปดิบ = ส่วน test ที่เป็นรูปวัตถุดิบดิบ 110 รูป · \"–\" = ไม่มีของชนิดนั้นในชุดนั้น · cheese butter ใน dataset รวมเนยด้วย", { size: 17, color: "595959" })]),
  H2("5.1 สำหรับคนที่ 5: แปลงเป็นเวกเตอร์ 0/1"),
  codeBlock([
    "CONTRACT = [\"pork\", \"chicken\", \"shrimp\", \"bacon\", \"egg\", \"rice\",",
    "            \"pasta\", \"bread\", \"holy_basil\", \"tomato\", \"cheese\", \"cabbage\"]",
    "",
    "from_image = {it[\"ingredient\"] for it in predict_image(img)}",
    "from_text  = set(text_extractor(user_text))   # ของคนที่ 4",
    "found = from_image | from_text",
    "x = [1 if name in found else 0 for name in CONTRACT]   # ใช้กับ recipe_model",
  ]),
  P("ชื่อมาตรฐานต้องสะกดตรงกันทั้ง 3 ฝั่ง (CV, NLP, recipes_dataset.csv) ใช้ตัวพิมพ์เล็กและ _ แบบ holy_basil"),
);

// 6. Limitations
children.push(
  H1("6. ข้อจำกัดของโมเดล (ต้องรู้และพูดในสไลด์)"),
  table(["ข้อจำกัด", "รายละเอียด", "ผลต่อระบบ"], [
    ["ไม่มีข้าวและพาสต้า", "dataset ที่ได้รับไม่มีรูปสองอย่างนี้", "ต้องรับจากข้อความพิมพ์"],
    ["pork ในอาหารสุก", "รูป pork ใน dataset เดิมเป็นอาหารสุกบนจาน แยกจาก steak ยาก (ทายเป็น steak ~28–32%)", "รูปหมูดิบในตู้เย็นใช้ได้ดีกว่ามาก"],
    ["cabbage = กะหล่ำปลี", "dataset ไม่มีผักกาดขาว · label บางรูปปนกะหล่ำดาว (5 จาก 12 รูปที่สุ่มดู)", "ผักกาดขาวจริงอาจไม่เจอ (คนที่ 3 ช่วยเช็คสูตร)"],
    ["บาง class รูปน้อย", "เช่น okra, mango, spring onion mAP ต่ำ (ไม่อยู่ใน 12 ชนิด)", "กระทบเฉพาะตอน only_contract=False"],
    ["ยังไม่ได้ทดสอบกับรูปตู้เย็นจริงของทีม", "ตัวเลขทั้งหมดมาจาก test ใน dataset", "แนะนำให้ลองรูปจริงก่อน Live Demo"],
  ], [2300, 4226, 2500]),
);

// 7. Results for slide
children.push(
  H1("7. ผลลัพธ์หลัก (สำหรับสไลด์หน้า 5)"),
  H2("7.1 ผลบน test (วัดครั้งเดียวหลังเลือกโมเดลจาก val)"),
  table(["test", "รูป", "P", "R", "mAP50", "mAP50-95"], [
    ["test ทั้งชุด", "1,427", "0.728", "0.601", "0.643", "0.525"],
    ["ส่วน test เดิม", "1,317", "0.711", "0.548", "0.603", "0.502"],
    ["ส่วน test วัตถุดิบดิบ", "110", "0.939", "0.916", "0.950", "0.747"],
  ], [2826, 1200, 1200, 1200, 1300, 1300]),
  H2("7.2 ตารางเทียบการทดลอง (บน val)"),
  table(["#", "การทดลอง", "เปลี่ยนอะไร", "val mAP50", "val mAP50-95", "เวลาเทรน"], [
    ["1", "baseline_n", "YOLO11n · dataset v1", "0.510", "0.409", "46.1 นาที"],
    ["2", "baseline_n_v2", "+ รูปวัตถุดิบดิบ 709 รูป (dataset v2)", "0.556", "0.436", "48.1 นาที"],
    ["3", "yolo11s_v2 ✔", "โมเดลใหญ่ขึ้น YOLO11s", "0.602", "0.489", "80.3 นาที"],
  ], [500, 1800, 3026, 1200, 1300, 1200]),
  P([t("val ของ #1 คือ v1 (1,959 รูป) ส่วน #2–#3 คือ v2 (2,068 รูป) · เทียบบน val ชุดเดียวกันได้ใน experiments.md #2 และ #3 เช่น val วัตถุดิบดิบ mAP50-95: #1 0.392 → #2 0.674 → #3 0.718", { size: 17, color: "595959" })]),
  H2("7.3 เรื่องที่เล่าในสไลด์ได้ (โยงบทเรียน)"),
  B([bold("บท 01 (T, P, E): "), t("ข้อมูลเดิม (E) เป็นอาหารสุก ไม่ตรงกับงานจริง (T = วัตถุดิบในตู้เย็น) พอเพิ่มรูปวัตถุดิบดิบ ผลบนรูปดิบ (P) ขึ้นจาก 0.392 เป็น 0.674")]),
  B([bold("บท 03 (overfitting): "), t("YOLO11s มี val loss ต่ำสุดที่ epoch 27 แล้วกลับขึ้น ขณะที่ train loss ยังลด = เริ่ม overfit · เลือก best.pt จาก val และใช้ test ครั้งเดียว")]),
  B([bold("บท 05 (ANN): "), t("YOLO เป็น neural network หลายชั้น เทรนด้วย gradient descent + backpropagation ดู loss curve")]),
  B([bold("บท 08: "), t("เลือก confidence threshold 0.28 จาก F1 curve บน val (เนื้อหาเสริม: precision/recall/F1)")]),
  ...image("results/yolo11s_v2/loss_train_vs_val.png", 560, "รูป 1: loss ของ train vs val (YOLO11s) · val loss ต่ำสุดที่ epoch 27"),
  ...image("results/final/test_BoxPR_curve.png", 470, "รูป 2: Precision-Recall curve บน test"),
);

// 8. slide checklist
children.push(
  H1("8. เช็คลิสต์ทำสไลด์หน้า 5"),
  B("ชื่อโมเดล: YOLO11s (Ultralytics) ตระกูลเดียวกับ YOLOv8 ที่เขียนในแผน (เป็นรุ่นใหม่กว่า)"),
  B("Dataset: foodseg_fridge_thai_yolo 75 class + รูปวัตถุดิบดิบจาก Roboflow 709 รูป · แบ่ง train/val/test"),
  B("กราฟ: loss curve (รูป 1) · PR curve (รูป 2) · confusion matrix (results/final/test_confusion_matrix_normalized.png)"),
  B("ตัวเลข: test mAP50 0.643 / mAP50-95 0.525 · รูปวัตถุดิบดิบ mAP50 0.950"),
  B("ตารางเทียบ 3 การทดลอง (บท 7.2) + เหตุผลที่เพิ่มข้อมูล (บท 01)"),
  B("ข้อจำกัด: ข้าว/พาสต้า, pork อาหารสุก, cabbage"),
  B([t("เครดิต dataset (CC BY 4.0 / MIT) จาก "), ...code("data/sources.md")]),
);

// 9. Retrain guide
children.push(
  H1("9. ถ้าต้องเทรนต่อ: กติกาและขั้นตอน"),
  callout("กติกาที่ต้องรักษา", [
    [bold("ห้ามใช้ test ซ้ำ: "), t("test ถูกใช้ไปแล้วครั้งเดียว (commit 4c9b14f) ถ้าเทรนโมเดลใหม่ ให้เลือกจาก val เท่านั้น และรายงานว่า test ชุดนี้ใช้ไปแล้ว")],
    [bold("เปลี่ยนทีละอย่าง: "), t("ให้เทียบกับ run เดิมได้ (ดู args.yaml ของแต่ละ run)")],
    [bold("ห้ามแต่งตัวเลข: "), t("ทุกการทดลองบันทึกลง experiments.md และ results/<ชื่อ>/")],
    [bold("ประหยัด CU: "), t("กดเซลล์ 5 ครั้งเดียว · จบแล้วรันเซลล์ 7 ตัด runtime ทันที")],
  ], LIGHT),
  gap(),
  H2("9.1 เทรนบน Colab"),
  N([t("Google Drive โฟลเดอร์ "), ...code("MyDrive/Mini_Project_Finoria_ML/"), t(" ต้องมี "), ...code("foodseg_yolo_v2.zip"), t(" และ "), ...code("train.py")], "steps2"),
  N([t("Colab → File → Open notebook → GitHub → "), ...code("Finoria-co/smart-fridge-cv"), t(" → "), ...code("notebooks/colab_train.ipynb"), t(" → Copy to Drive")], "steps2"),
  N("Runtime → Change runtime type → L4 GPU (~1.94 CU/ชม.) → Connect", "steps2"),
  N("รันเซลล์ 1 → 4 ทีละเซลล์ (เซลล์ 3 ต้องนับได้ train 10545 / val 2068 / test 1427) · ถ้ามีปุ่ม Restart session ไม่ต้องกด", "steps2"),
  N([t("แก้ "), ...code("NAME"), t(" / "), ...code("MODEL"), t(" ในเซลล์ 5 แล้วกดครั้งเดียว · ห้าม Run all")], "steps2"),
  N("เสร็จแล้วรันเซลล์ 6 (ดูผล) และ 7 (ตัด runtime) · ใช้เซลล์ Resume เฉพาะตอน session หลุดกลางคัน", "steps2"),
  H2("9.2 วิเคราะห์ผลในเครื่อง (WSL)"),
  codeBlock([
    "# copy run จาก Drive มาไว้ runs/detect/<name>/ แล้ว",
    "python scripts/analyze_run.py runs/detect/<name> <name>",
    "python scripts/eval_per_class.py runs/detect/<name>/weights/best.pt <name>_on_v1val --data /root/data/foodseg_yolo/data.yaml",
    "python scripts/eval_per_class.py runs/detect/<name>/weights/best.pt <name>_on_v2added --data /root/data/foodseg_yolo_v2/data_added_val.yaml",
    "python scripts/top_confusions.py runs/detect/<name>/weights/best.pt <name> --classes pork shrimp",
  ]),
  H2("9.3 Run ที่อยู่ใน Drive (runs/detect/)"),
  table(["โฟลเดอร์ใน Drive", "คืออะไร", "ในเครื่อง / results"], [
    [c("baseline_n"), "#1 YOLO11n dataset v1", c("results/baseline_n/")],
    [c("baseline_n_v2-2"), "#2 YOLO11n dataset v2", c("results/baseline_n_v2/")],
    [c("yolo11s_v2-2"), "#3 YOLO11s dataset v2 (โมเดลสุดท้าย = Release)", c("results/yolo11s_v2/")],
  ], [2600, 3626, 2800]),
);

// 10. Commit history
children.push(
  H1("10. ประวัติ commit (ย้อนกลับได้ทีละขั้น)"),
  P([t("ทุก commit อยู่บน branch "), ...code("main"), t(" · ดูไฟล์ ณ commit ใดๆ: "), ...code("git show <commit>:<ไฟล์>"), t(" · ย้อนทีละขั้น: "), ...code("git revert <commit>")]),
  table(["commit", "วันที่", "เนื้อหา"], [
    [c("c3df66b"), "17 พ.ย.", "แบ่ง test จาก val (stratified, seed 42)"],
    [c("f070485"), "17 พ.ย.", "แผนงาน CV (PLAN.md) + CLAUDE.md"],
    [c("75e3647"), "17 พ.ย.", "scripts/train.py + smoke test"],
    [c("33ac167"), "17 พ.ย.", "notebook Colab + zip dataset"],
    [c("2ab33f3"), "17 พ.ย.", "README สำหรับทีม + โฟลเดอร์ results/"],
    [c("daa7cee"), "17 พ.ย.", "ผล baseline_n + analyze_run.py"],
    [c("7089428"), "17 พ.ย.", "mAP ราย class ของ baseline"],
    [c("7db5d18"), "17 พ.ย.", "วิเคราะห์ข้อผิดพลาด (confusion + รูปตัวอย่าง)"],
    [c("b69f6dd"), "17 พ.ย.", "เลือกแหล่ง Roboflow + ตารางแปลง class (data/rf_class_map.yaml)"],
    [c("7b06818"), "17 พ.ย.", "scripts/build_dataset_v2.py"],
    [c("7d2240f"), "17 พ.ย.", "สร้าง dataset v2 (+709 รูป)"],
    [c("7c12244"), "18 พ.ย.", "ผล baseline_n_v2 + เทียบ v1/v2"],
    [c("b3fe65c"), "18 พ.ย.", "ปรับขอบเขตงานให้ตรงหน้าที่คนที่ 2"],
    [c("7eeb6b2"), "18 พ.ย.", "predict.py + ingredient_map.json + tests"],
    [c("4de6a5e"), "18 พ.ย.", "ผล YOLO11s · เลือกเป็นโมเดลสุดท้าย"],
    [c("4c9b14f"), "18 พ.ย.", "ผล test ครั้งเดียว + conf 0.28"],
    [c("e0931ef"), "18 พ.ย.", "ตรวจ class cabbage"],
    [c("8414f20"), "18 พ.ย.", "ลิงก์ Release v1.0-yolo11s ใน README / PLAN"],
  ], [1400, 1300, 6326]),
  P([t("รายการเต็ม: ", { size: 18 }), ...code("git log --oneline")]),
);

// 11. Handoff checklist
children.push(
  new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun("11. เช็คลิสต์ส่งต่อ")] }),
  P([t("ติดปัญหาเรื่องโมเดลหรือ predict_image(): เปิด issue ใน repo "), link("Finoria-co/smart-fridge-cv", "https://github.com/Finoria-co/smart-fridge-cv/issues"), t(" หรือทักคนที่ 2")]),
  table(["งาน", "ใคร", "สถานะ"], [
    ["เทรน + วัดผล + กราฟ", "คนที่ 2", "✅ เสร็จ"],
    ["best.pt (Release v1.0-yolo11s)", "คนที่ 2", "✅ เสร็จ"],
    ["predict_image() + README", "คนที่ 2", "✅ เสร็จ"],
    ["เชื่อม predict_image() เข้า app.py และลองกับรูปจริง", "คนที่ 6", "⬜ รอทำ"],
    ["รวมผลรูป + ข้อความ เป็นเวกเตอร์ 12 ชนิด", "คนที่ 5", "⬜ รอทำ"],
    ["รับ rice / pasta จากข้อความ", "คนที่ 4", "⬜ รอทำ"],
    ["เช็ค cabbage ในสูตร (ผักกาดขาว vs กะหล่ำปลี)", "คนที่ 3", "⬜ รอทำ"],
    ["สไลด์หน้า 5", "คนที่ 2 (ทำพร้อมทีม)", "⬜ รอรวมสไลด์"],
    ["แจ้ง feedback / bug ของ predict_image()", "คนที่ 6 → คนที่ 2", "⬜ ระหว่างทาง"],
  ], [5026, 2200, 1800]),
);

// ---------- document ----------
const doc = new Document({
  creator: "Finoria · CV",
  title: "คู่มือส่งต่องาน Computer Vision",
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", next: "Normal",
        run: { font: FONT, size: 52, bold: true, color: ACCENT },
        paragraph: { spacing: { before: 0, after: 160 } } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 30, bold: true, color: ACCENT },
        paragraph: { spacing: { before: 360, after: 140 }, keepNext: true, outlineLevel: 0,
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT, space: 4 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, color: "2E75B6" },
        paragraph: { spacing: { before: 240, after: 100 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 22, bold: true },
        paragraph: { spacing: { before: 160, after: 80 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 1080, hanging: 270 } } } } ] },
      { reference: "steps", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
      { reference: "steps2", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [t("smart-fridge-cv · คู่มือส่งต่องาน CV · หน้า ", { size: 16, color: "7F7F7F" }),
        new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: "7F7F7F" })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT, buf.length); });
