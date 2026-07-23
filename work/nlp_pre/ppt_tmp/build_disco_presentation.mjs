import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT_DIR = "D:/Codex/rsfm-fairness-audit/outputs";
const QA_DIR = "D:/Codex/rsfm-fairness-audit/work/nlp_pre/ppt_tmp/qa_optimized";
const ASSET_DIR = "D:/Codex/rsfm-fairness-audit/work/nlp_pre/assets";
const FINAL_PPTX = process.env.FINAL_PPTX || path.join(OUT_DIR, "disco_nlp_shortcuts_biases_presentation_optimized.pptx");

const W = 1280;
const H = 720;
const FONT = "Helvetica Neue";
const INK = "#000000";
const MUTED = "#555555";
const PANEL = "#EDEDED";
const RULE = "#B8BCC4";
const ACCENT = "#FF6B35";
const PURPLE = "#6F42C1";
const BLUE = "#2563EB";

async function readImage(fileName) {
  const bytes = await fs.readFile(path.join(ASSET_DIR, fileName));
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

function addBox(slide, left, top, width, height, fill = PANEL, lineFill = "none") {
  return slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill: lineFill, width: lineFill === "none" ? 0 : 1 },
  });
}

function addText(slide, text, left, top, width, height, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left, top, width, height },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: FONT,
    fontSize: opts.fontSize ?? 24,
    bold: opts.bold ?? false,
    color: opts.color ?? INK,
    alignment: opts.alignment ?? "left",
    verticalAlignment: opts.verticalAlignment ?? "top",
    lineSpacing: opts.lineSpacing ?? 1.12,
    autoFit: opts.autoFit ?? "shrinkText",
    insets: { top: 0, right: 0, bottom: 0, left: 0 },
  };
  return shape;
}

function addTitle(slide, title, section = "") {
  if (section) addText(slide, section.toUpperCase(), 42, 36, 420, 22, { fontSize: 14, bold: true, color: MUTED });
  addText(slide, title, 42, 62, 980, 92, { fontSize: 43, bold: true, lineSpacing: 1.02 });
  slide.shapes.add({
    geometry: "rect",
    position: { left: 42, top: 155, width: 1196, height: 1 },
    fill: RULE,
    line: { style: "solid", fill: RULE, width: 0 },
  });
}

function addFooter(slide, n) {
  addText(slide, "Chen et al., 2023 | Natural Language Processing: Shortcuts and Biases in AI", 42, 672, 880, 22, {
    fontSize: 12,
    color: MUTED,
  });
  addText(slide, String(n).padStart(2, "0"), 1184, 662, 54, 25, { fontSize: 15, alignment: "right", color: MUTED });
}

function setNotes(slide, text) {
  slide.speakerNotes.textFrame.setText(text.trim());
  slide.speakerNotes.setVisible(true);
}

function addBullets(slide, items, left, top, width, height, fontSize = 24) {
  const text = items.map((item) => `- ${item}`).join("\n");
  return addText(slide, text, left, top, width, height, { fontSize, color: INK, lineSpacing: 1.18 });
}

function addMetric(slide, value, label, left, top, width, accent = INK) {
  addBox(slide, left, top, width, 138, "#F7F7F7", RULE);
  addText(slide, value, left + 22, top + 18, width - 44, 62, { fontSize: 48, bold: true, color: accent });
  addText(slide, label, left + 22, top + 88, width - 44, 38, { fontSize: 17, color: MUTED });
}

function addCompactMetric(slide, value, label, left, top, width, accent = INK) {
  addBox(slide, left, top, width, 118, "#F7F7F7", RULE);
  addText(slide, value, left + 18, top + 14, width - 36, 46, { fontSize: 39, bold: true, color: accent });
  addText(slide, label, left + 18, top + 68, width - 36, 38, { fontSize: 14, color: MUTED });
}

function addArrowMetric(slide, label, before, after, x, y, w, note = "") {
  addBox(slide, x, y, w, 116, "#F7F7F7", RULE);
  addText(slide, label, x + 18, y + 16, w - 36, 24, { fontSize: 17, bold: true, color: MUTED });
  addText(slide, before, x + 18, y + 50, 88, 38, { fontSize: 29, bold: true, color: "#777777" });
  addText(slide, "->", x + 108, y + 55, 42, 28, { fontSize: 24, bold: true, color: MUTED, alignment: "center" });
  addText(slide, after, x + 152, y + 50, 110, 38, { fontSize: 31, bold: true, color: BLUE });
  if (note) addText(slide, note, x + 18, y + 88, w - 36, 18, { fontSize: 12, color: MUTED });
}

function addMiniTable(slide, rows, left, top, width, rowHeight = 56) {
  rows.forEach((row, i) => {
    const y = top + i * rowHeight;
    addBox(slide, left, y, width, rowHeight - 4, i === 0 ? "#111111" : i % 2 ? "#F4F4F4" : "#FFFFFF", i === 0 ? "#111111" : RULE);
    const color = i === 0 ? "#FFFFFF" : INK;
    const bold = i === 0;
    const colW = width / row.length;
    row.forEach((cell, j) => {
      addText(slide, cell, left + j * colW + 12, y + 13, colW - 24, rowHeight - 20, {
        fontSize: i === 0 ? 15 : 18,
        bold,
        color,
      });
    });
  });
}

function addRouteSlide(presentation, n, title, section, body, notes) {
  const slide = presentation.slides.add();
  slide.background.fill = "#FFFFFF";
  addTitle(slide, title, section);
  body(slide);
  addFooter(slide, presentation.slides.items.length);
  setNotes(slide, notes);
  return slide;
}

async function build() {
  await fs.mkdir(OUT_DIR, { recursive: true });
  await fs.mkdir(QA_DIR, { recursive: true });

  const fig1 = await readImage("figure1_disco_pipeline_clean.png");
  const fig2 = await readImage("figure2_prompting_clean.png");
  const table1 = await readImage("table1_quality_crop.png");
  const table3 = await readImage("table3_results_crop.png");
  const table4 = await readImage("table4_gain_crop.png");
  const table5 = await readImage("table5_counterfactual_crop.png");

  const presentation = Presentation.create({ slideSize: { width: W, height: H } });

  {
    const slide = presentation.slides.add();
    slide.background.fill = "#FFFFFF";
    addText(slide, "DISCO", 42, 48, 240, 60, { fontSize: 32, bold: true, color: PURPLE });
    addText(slide, "Distilling Counterfactuals with Large Language Models", 42, 162, 800, 170, {
      fontSize: 56,
      bold: true,
      lineSpacing: 0.98,
    });
    addText(slide, "A paper presentation for Natural Language Processing: Shortcuts and Biases in Artificial Intelligence", 42, 356, 790, 70, {
      fontSize: 24,
      color: MUTED,
    });
    addBox(slide, 42, 506, 430, 78, "#F7F7F7", RULE);
    addText(slide, "Chen et al. (2023) | arXiv:2212.10534v3", 66, 532, 390, 32, { fontSize: 19, bold: true });
    addText(slide, "Your name", 42, 622, 500, 28, { fontSize: 18, color: MUTED });
    addBox(slide, 896, 0, 384, 720, PANEL);
    addText(slide, "Shortcut problem\nCounterfactual fix\nLLM distillation", 930, 214, 300, 210, {
      fontSize: 39,
      bold: true,
      lineSpacing: 1.18,
    });
    addFooter(slide, 1);
    setNotes(slide, `
Good morning everyone. Today I will present the paper DISCO: Distilling Counterfactuals with Large Language Models by Chen and colleagues. The three phrases on the right are my talk arc: first the shortcut problem, then the counterfactual fix, and finally LLM distillation. I chose this paper because it fits the course topic very directly. DISCO is about a concrete way to reduce shortcut learning in NLP: generate counterfactual training examples with a large language model, filter them with a task-specific teacher, and use them to train a smaller model that is less dependent on spurious patterns.
`);
  }

  addRouteSlide(
    presentation,
    2,
    "The paper asks how we can stop NLI models from taking shortcuts.",
    "Motivation",
    (slide) => {
      addText(slide, "Central question", 42, 194, 350, 32, { fontSize: 20, bold: true, color: MUTED });
      addText(slide, "Can generated counterfactual data make a model rely more on meaning and less on dataset artifacts?", 42, 232, 720, 120, {
        fontSize: 34,
        bold: true,
        lineSpacing: 1.08,
      });
      addMetric(slide, "+6.5", "robustness points over the SNLI-subset baseline", 818, 205, 180, ACCENT);
      addMetric(slide, "+9.2", "robustness points over the WANLI baseline", 1030, 205, 180, ACCENT);
      addBullets(
        slide,
        [
          "Shortcut: a pattern that correlates with the label but is not the true reason.",
          "Counterfactual: a minimally changed example whose label should change.",
          "DISCO: use GPT-3 to overgenerate edits, then filter them with an NLI teacher model.",
        ],
        42,
        420,
        1130,
        150,
        23,
      );
    },
    `
The main question is not just whether GPT-3 can generate more data. The deeper question is whether generated data can attack shortcuts. A shortcut is a pattern that works in the training set but fails under more careful evaluation. The authors focus on natural language inference, or NLI. They generate counterfactual examples: examples where the important part of the input changes, and the correct label should change too. Then they test whether models trained with those examples become more robust.
`,
  );

  addRouteSlide(
    presentation,
    3,
    "I will build the talk from the problem to the evidence.",
    "Roadmap",
    (slide) => {
      const steps = [
        ["1", "Introduction", "NLI, shortcuts, and the counterfactual intuition."],
        ["2", "Related work", "Why prior CAD and LLM data creation leave a gap."],
        ["3", "Methodology", "How DISCO generates, filters, and trains."],
        ["4", "Experiments", "How quality and robustness are measured."],
        ["5", "Results", "What improves, and what the tables show."],
        ["6", "Limits and impact", "Where the method helps, and where caution remains."],
      ];
      steps.forEach((s, i) => {
        const x = 42 + (i % 3) * 398;
        const y = 196 + Math.floor(i / 3) * 188;
        addText(slide, s[0], x, y + 10, 64, 58, { fontSize: 42, bold: true, color: INK });
        addBox(slide, x + 78, y, 300, 132, "#F7F7F7", RULE);
        addText(slide, s[1], x + 100, y + 22, 245, 30, { fontSize: 23, bold: true });
        addText(slide, s[2], x + 100, y + 66, 235, 44, { fontSize: 16, color: MUTED });
      });
      addText(slide, "The required seminar structure is the outer frame; shortcut learning is the story line connecting every part.", 42, 568, 1000, 38, {
        fontSize: 23,
        color: MUTED,
      });
    },
    `
The structure is designed for a 30-minute seminar presentation. I will first introduce the task and the shortcut problem, then explain the idea of counterfactual data augmentation, then go through the DISCO pipeline, and finally discuss experiments, results, limitations, and broader impact. This also directly matches the presentation requirements from the course slide.
`,
  );

  addRouteSlide(
    presentation,
    4,
    "NLI tests whether a hypothesis follows from a premise.",
    "Background",
    (slide) => {
      addBox(slide, 42, 202, 585, 238, "#F7F7F7", RULE);
      addText(slide, "Premise", 66, 228, 120, 24, { fontSize: 18, bold: true, color: MUTED });
      addText(slide, "A young girl looks up as she rides a merry-go-round.", 66, 264, 520, 54, { fontSize: 27, bold: true });
      addText(slide, "Hypothesis", 66, 346, 130, 24, { fontSize: 18, bold: true, color: MUTED });
      addText(slide, "The little girl cannot wait to ride the roller coaster.", 66, 378, 520, 42, { fontSize: 23 });
      addBox(slide, 690, 202, 235, 105, "#FFFFFF", RULE);
      addText(slide, "Entailment", 710, 228, 185, 28, { fontSize: 24, bold: true });
      addText(slide, "The hypothesis must be true.", 710, 262, 178, 34, { fontSize: 15, color: MUTED });
      addBox(slide, 690, 342, 235, 105, "#FFFFFF", RULE);
      addText(slide, "Neutral", 710, 368, 185, 28, { fontSize: 24, bold: true });
      addText(slide, "It might be true, but is not guaranteed.", 710, 402, 178, 34, { fontSize: 15, color: MUTED });
      addBox(slide, 962, 202, 235, 105, "#FFFFFF", RULE);
      addText(slide, "Contradiction", 982, 228, 185, 28, { fontSize: 24, bold: true });
      addText(slide, "The hypothesis must be false.", 982, 262, 178, 34, { fontSize: 15, color: MUTED });
      addText(slide, "NLI is useful because it looks like reasoning, but it is also vulnerable to shortcuts.", 42, 510, 900, 70, {
        fontSize: 30,
        bold: true,
      });
    },
    `
Before the method, we need the task. In natural language inference, the input is a premise and a hypothesis. The model predicts one of three labels: entailment, neutral, or contradiction. In the example, the premise says merry-go-round and the hypothesis says roller coaster. The girl might want to ride the roller coaster, but the premise does not prove it, so the label is neutral. NLI is a good testbed because it appears to require reasoning about meaning, but many datasets contain shallow clues.
`,
  );

  addRouteSlide(
    presentation,
    5,
    "Shortcut learning happens when a cue replaces the reasoning step.",
    "Shortcut Learning",
    (slide) => {
      const cols = [
        ["Lexical overlap", "If premise and hypothesis share many words, predict entailment."],
        ["Negation words", "If words like no, never, or not appear, predict contradiction."],
        ["Syntactic heuristics", "Assume simple word order patterns reveal the label."],
      ];
      cols.forEach((c, i) => {
        const x = 42 + i * 398;
        addBox(slide, x, 205, 340, 210, i === 0 ? "#F7F7F7" : "#FFFFFF", RULE);
        addText(slide, c[0], x + 24, 232, 292, 34, { fontSize: 27, bold: true });
        addText(slide, c[1], x + 24, 286, 285, 88, { fontSize: 21, color: MUTED });
      });
      addText(slide, "The model can look accurate while failing when the shortcut no longer works.", 42, 488, 760, 62, {
        fontSize: 30,
        bold: true,
        lineSpacing: 1.08,
      });
      addBox(slide, 856, 470, 360, 150, "#F7F7F7", RULE);
      addText(slide, "Course connection", 880, 492, 250, 24, { fontSize: 18, bold: true, color: MUTED });
      addText(
        slide,
        "DISCO belongs to data manipulation -> data augmentation. It mitigates shortcuts by breaking spurious feature-label correlations with counterfactual examples.",
        880,
        526,
        306,
        72,
        { fontSize: 15, color: INK, lineSpacing: 1.12 },
      );
    },
    `
A shortcut is not always a wrong feature. It is a feature that becomes dangerous when it replaces the real task. For example, high word overlap often correlates with entailment in NLI datasets, and negation often correlates with contradiction. In the seminar framework, DISCO belongs to data manipulation, specifically data augmentation. It tries to break spurious feature-label correlations by adding counterfactual examples.
`,
  );

  addRouteSlide(
    presentation,
    6,
    "Counterfactual data breaks the link between shortcut cues and labels.",
    "Key Idea",
    (slide) => {
      addBox(slide, 42, 205, 500, 190, "#F7F7F7", RULE);
      addText(slide, "Original", 66, 228, 120, 24, { fontSize: 18, bold: true, color: MUTED });
      addText(slide, "A girl rides a merry-go-round.", 66, 260, 420, 38, { fontSize: 26, bold: true });
      addText(slide, "Label: Neutral", 66, 328, 220, 32, { fontSize: 24, color: ACCENT, bold: true });
      addBox(slide, 690, 205, 500, 190, "#F7F7F7", RULE);
      addText(slide, "Counterfactual edit", 714, 228, 200, 24, { fontSize: 18, bold: true, color: MUTED });
      addText(slide, "A girl looks at a very tall roller coaster.", 714, 260, 420, 55, { fontSize: 26, bold: true });
      addText(slide, "Label: Entailment", 714, 328, 240, 32, { fontSize: 24, color: ACCENT, bold: true });
      addText(slide, "Same hypothesis, targeted context change", 382, 430, 520, 34, { fontSize: 26, bold: true, alignment: "center" });
      addText(slide, "If the model only memorizes superficial cues, it will struggle with the pair.", 270, 492, 760, 50, {
        fontSize: 28,
        color: MUTED,
        alignment: "center",
      });
    },
    `
Counterfactual data augmentation is powerful because it changes the part that should causally affect the label. Here the hypothesis stays about wanting to ride a roller coaster. In the original premise, the girl is on a merry-go-round, so the label is neutral. In the counterfactual version, the premise is changed so she looks at a tall roller coaster with excitement, so the label becomes entailment. A robust model should react to this meaningful change, not to accidental surface patterns.
`,
  );

  addRouteSlide(
    presentation,
    7,
    "DISCO contributes an automatic way to create counterfactual data at scale.",
    "Contribution",
    (slide) => {
      addText(slide, "The paper's claim", 42, 200, 300, 30, { fontSize: 20, bold: true, color: MUTED });
      addText(slide, "LLMs can generate many diverse counterfactual candidates, but a task-specific teacher is needed to distill the useful ones.", 42, 238, 850, 108, {
        fontSize: 35,
        bold: true,
        lineSpacing: 1.05,
      });
      addCompactMetric(slide, "75k", "DISCO examples used in experiments", 42, 426, 205, PURPLE);
      addCompactMetric(slide, "83%", "human-evaluated label-flip rate", 282, 426, 205, PURPLE);
      addCompactMetric(slide, "+6%", "average robustness improvement reported in the paper", 522, 426, 205, ACCENT);
      addCompactMetric(slide, "+2%", "OOD generalization improvement", 762, 426, 205, ACCENT);
      addCompactMetric(slide, "+10%", "counterfactual pair consistency improvement", 1002, 426, 205, ACCENT);
      addText(slide, "Detailed gains depend on the baseline: for example, robustness is +6.5 over SNLI-subset and +9.2 over WANLI.", 42, 574, 1050, 34, {
        fontSize: 19,
        color: MUTED,
      });
    },
    `
The contribution is not only generation. The authors argue that raw LLM generations are too noisy for training. DISCO therefore uses a distillation idea. GPT-3 overgenerates possible edits, and then a strong NLI teacher model filters for candidates that seem to change the label in the desired direction. The abstract reports about 6 percent robustness improvement, 2 percent OOD improvement, and 10 percent better counterfactual pair consistency. In the detailed tables, the exact gain depends on the baseline.
`,
  );

  addRouteSlide(
    presentation,
    8,
    "Related work shows why DISCO needs both generation and filtering.",
    "Related Work",
    (slide) => {
      addMiniTable(
        slide,
        [
          ["Approach", "Strength", "Main limitation"],
          ["Human CAD", "High control", "Costly and small"],
          ["Supervised generators", "Scalable", "Fixed perturbation types"],
          ["LLM data creation", "Diverse and flexible", "Needs quality control"],
          ["Debiasing methods", "Target shortcuts", "Often need known bias types"],
        ],
        42,
        190,
        850,
        66,
      );
      addText(slide, "DISCO's position", 958, 208, 220, 30, { fontSize: 21, bold: true, color: MUTED });
      addText(slide, "Use an LLM to avoid a fixed edit inventory, then use a teacher model to avoid unfiltered noisy data.", 958, 254, 240, 174, {
        fontSize: 25,
        bold: true,
        lineSpacing: 1.12,
      });
    },
    `
The paper is positioned between several research lines. Human counterfactual data is high quality but expensive and small. Supervised generators can scale, but they usually learn predefined perturbation types. Large language models are flexible, but their outputs are not automatically reliable. Debiasing methods means methods designed to reduce reliance on biased or spurious cues, but many of them require knowing the bias type in advance. DISCO combines these ideas: use the LLM for diverse candidate generation and use a teacher model for task-specific quality control.
`,
  );

  addRouteSlide(
    presentation,
    9,
    "The gap is not generation alone; it is scalable, targeted, and reliable generation.",
    "Related Work",
    (slide) => {
      addMiniTable(
        slide,
        [
          ["Method family", "What it gives", "Why DISCO still matters"],
          ["Human-CAD", "Meaningful edits with human judgment", "Hard to scale and can be repetitive"],
          ["Tailor / Polyjuice", "Automatic edits", "Often tied to fixed perturbation types"],
          ["WANLI", "GPT-3 data plus human labels", "Still depends on human annotation"],
          ["DISCO", "LLM diversity plus teacher filtering", "Targets counterfactual augmentation directly"],
        ],
        42,
        190,
        920,
        63,
      );
      addBox(slide, 1000, 225, 190, 260, "#F7F7F7", RULE);
      addText(slide, "Positioning", 1022, 256, 150, 28, { fontSize: 20, bold: true, color: MUTED });
      addText(slide, "DISCO treats the LLM as a data generator, not as the final reasoning model.", 1022, 304, 140, 110, {
        fontSize: 22,
        bold: true,
        lineSpacing: 1.12,
      });
    },
    `
This additional related-work slide makes the contrast more explicit. The closest alternatives each solve part of the problem. Human-CAD gives meaningful counterfactual edits, but it is expensive. Tailor and Polyjuice are automatic counterfactual editing systems: they can generate edits, but their edit operations are more fixed than open-ended LLM generation. WANLI uses GPT-3 but still relies on human annotation. DISCO's specific move is to use a general LLM for diversity and a task-specific teacher model for automatic filtering.
`,
  );

  addRouteSlide(
    presentation,
    10,
    "The DISCO pipeline turns one dataset into targeted counterfactual training data.",
    "Method Overview",
    (slide) => {
      slide.images.add({
        blob: fig1,
        contentType: "image/png",
        alt: "Figure 1 from Chen et al. showing the DISCO pipeline.",
        fit: "contain",
        position: { left: 70, top: 185, width: 470, height: 420 },
      });
      addBullets(
        slide,
        [
          "Select examples, especially ambiguous ones.",
          "Extract spans that could be edited.",
          "Ask GPT-3 to overgenerate label-flipping perturbations.",
          "Filter with heuristics and a DeBERTa-v2 NLI teacher.",
          "Use the distilled examples for student-model training.",
        ],
        612,
        203,
        548,
        262,
        25,
      );
      addText(slide, "Source: Figure 1 in Chen et al. (2023).", 72, 610, 420, 20, { fontSize: 12, color: MUTED });
    },
    `
This is the most important method slide. We can read the pipeline from top to bottom. Start with a seed dataset, choose the examples to edit, identify spans inside the premise, and then ask GPT-3 to produce many candidate edits. For example, an original premise about a girl riding a merry-go-round can be locally edited into a premise about the same girl looking at a roller coaster, so that the hypothesis about wanting to ride the roller coaster becomes entailed. Because generation is noisy, DISCO filters the candidates. The result is a counterfactual dataset, which is added to training data for a smaller student model. The key idea is overgeneration followed by conservative selection.
`,
  );

  addRouteSlide(
    presentation,
    11,
    "DISCO edits spans because shortcuts often hide in local phrases.",
    "Method Step 1",
    (slide) => {
      addBox(slide, 42, 205, 540, 250, "#F7F7F7", RULE);
      addText(slide, "Original premise", 72, 232, 180, 24, { fontSize: 18, bold: true, color: MUTED });
      addText(slide, "A young girl looks up as she rides a merry-go-round.", 72, 272, 465, 64, { fontSize: 30, bold: true });
      addText(slide, "Candidate span", 72, 365, 180, 24, { fontSize: 18, bold: true, color: MUTED });
      addText(slide, "as she rides a merry-go-round", 72, 398, 450, 36, { fontSize: 26, bold: true, color: PURPLE });
      addText(slide, "Data Cartography selects useful cases", 680, 214, 450, 32, { fontSize: 27, bold: true });
      addBullets(
        slide,
        [
          "DISCO uses Data Cartography to select ambiguous SNLI examples.",
          "Ambiguous examples are near decision boundaries.",
          "They are useful for robustness training because they expose hard cases.",
          "Span extraction then finds local places to intervene.",
        ],
        680,
        274,
        520,
        220,
        20,
      );
    },
    `
DISCO does not rewrite the whole input. It performs local edits. The authors first apply Data Cartography to select ambiguous examples from SNLI. These examples are useful because they are near decision boundaries and can expose shortcut behavior. Then a syntactic chunker identifies spans in the premise. These spans become possible intervention points for counterfactual edits.
`,
  );

  addRouteSlide(
    presentation,
    12,
    "GPT-3 is prompted to generate edits that flip the NLI label.",
    "Method Step 2",
    (slide) => {
      slide.images.add({
        blob: fig2,
        contentType: "image/png",
        alt: "Figure 2 from Chen et al. showing masked and insertion NLI prompts.",
        fit: "contain",
        position: { left: 64, top: 170, width: 1120, height: 360 },
      });
      addText(slide, "Two prompt styles", 80, 554, 220, 28, { fontSize: 21, bold: true, color: MUTED });
      addText(slide, "Masked prompting fills a blank; insertion mode lets the model see both prefix and suffix context.", 300, 554, 820, 52, {
        fontSize: 23,
        bold: true,
      });
      addText(slide, "Source: Figure 2 in Chen et al. (2023).", 80, 626, 420, 20, { fontSize: 12, color: MUTED });
    },
    `
This slide explains how generation works. The authors want GPT-3 to create a phrase that changes the label from the original label to a target label. They try masked prompting and insertion prompting. In insertion mode, GPT-3 can see both the prefix and the suffix around the edited span, which helps avoid a common completion problem: the model may continue the sentence without respecting the remaining context. The method overgenerates because many spans and many edits may not actually flip the label.
`,
  );

  addRouteSlide(
    presentation,
    13,
    "Two prompt modes serve the same goal: targeted local intervention.",
    "Method Step 2",
    (slide) => {
      addMiniTable(
        slide,
        [
          ["Prompting mode", "How it works", "Why it helps"],
          ["Masked prompting", "Replace a span with [blank] and ask GPT-3 to fill it", "Explicit target for a local edit"],
          ["Insertion mode", "Give the prefix and suffix around the missing span", "Uses surrounding context and lowers prompt cost"],
          ["Overgeneration", "Try multiple spans and target labels", "Creates diversity before filtering"],
        ],
        70,
        205,
        930,
        78,
      );
      addBox(slide, 1042, 232, 150, 230, "#F7F7F7", RULE);
      addText(slide, "Key point", 1065, 265, 110, 28, { fontSize: 20, bold: true, color: MUTED });
      addText(slide, "Fluent text is not enough. The edit must change the NLI relation.", 1065, 318, 100, 90, {
        fontSize: 21,
        bold: true,
        lineSpacing: 1.12,
      });
    },
    `
This slide turns Figure 2 into a comparison. Masked prompting and insertion prompting are two ways to ask GPT-3 for a local edit. The central idea is targeted intervention: the prompt includes the original label and the desired new label. The authors overgenerate because many outputs will be fluent English but not valid counterfactuals. This sets up the need for the teacher filter.
`,
  );

  addRouteSlide(
    presentation,
    14,
    "Teacher filtering decides which generated edits are reliable enough to train on.",
    "Method Step 3",
    (slide) => {
      addBox(slide, 42, 208, 520, 250, "#F7F7F7", RULE);
      addText(slide, "Heuristic filter removes obvious bad outputs", 72, 238, 450, 34, { fontSize: 27, bold: true });
      addBullets(
        slide,
        [
          "Prompt leakage or copied examples",
          "Repeated premise or hypothesis text",
          "Excessive overlap or negation shortcuts",
        ],
        72,
        305,
        430,
        110,
        22,
      );
      addBox(slide, 680, 208, 520, 250, "#F7F7F7", RULE);
      addText(slide, "NLI teacher checks prediction shift", 710, 238, 450, 34, { fontSize: 27, bold: true });
      addText(slide, "Delta l' = p(l' | P', H) - p(l' | P, H)", 710, 298, 430, 38, {
        fontSize: 25,
        bold: true,
        color: PURPLE,
      });
      addText(slide, "Keep candidates when the target-label probability increases strongly after the edit.", 710, 354, 420, 52, {
        fontSize: 20,
        color: MUTED,
      });
      addText(slide, "Teacher model: DeBERTa-v2 trained for NLI", 710, 425, 430, 28, { fontSize: 20, bold: true, color: PURPLE });
      addText(slide, "Filtering is the distillation step: the LLM proposes, the teacher selects.", 170, 520, 940, 44, {
        fontSize: 31,
        bold: true,
        alignment: "center",
      });
    },
    `
Teacher filtering is what turns generation into distillation. First, the system removes outputs that are obviously bad: copied prompt text, repeated input text, or artifacts like too much lexical overlap or negation shortcuts. Then a strong DeBERTa-v2 NLI model acts as a teacher. The formula shows the target-label probability shift: compare the target-label probability after the edit with the probability before the edit. If this delta is large, the candidate is treated as a useful counterfactual.
`,
  );

  addRouteSlide(
    presentation,
    15,
    "The experiments test quality, robustness, OOD generalization, and counterfactual consistency.",
    "Experiment Design",
    (slide) => {
      const blocks = [
        ["Data creation", "SNLI examples are selected and edited; DISCO creates about 75k distilled counterfactual examples."],
        ["Training", "RoBERTa-large student models are trained with baseline data, augmented data, or DISCO-only data."],
        ["Evaluation", "Stress tests, HANS, MNLI-hard, QNLI, Human-CAD, SNLI-hard counterfactual pairs, and WANLI counterfactual pairs."],
      ];
      blocks.forEach((b, i) => {
        const y = 195 + i * 128;
        addText(slide, `0${i + 1}`, 42, y + 8, 80, 54, { fontSize: 42, bold: true, color: i === 2 ? ACCENT : INK });
        addBox(slide, 140, y, 1010, 90, "#F7F7F7", RULE);
        addText(slide, b[0], 168, y + 18, 220, 28, { fontSize: 24, bold: true });
        addText(slide, b[1], 410, y + 17, 700, 48, { fontSize: 20, color: MUTED });
      });
    },
    `
The paper has two kinds of evaluation. First, it evaluates the generated counterfactual data itself, comparing DISCO with human-written CAD. Second, it trains student models and tests them on robustness and out-of-distribution benchmarks. The student model is RoBERTa-large. The evaluation sets include stress tests for known NLI shortcuts, HANS for syntactic heuristics, MNLI-hard and QNLI for distribution shift, and Human-CAD, SNLI-hard counterfactual pairs, and WANLI counterfactual pairs for consistency.
`,
  );

  addRouteSlide(
    presentation,
    16,
    "The evaluation is designed around the shortcut-learning story.",
    "Experiment Design",
    (slide) => {
      addMiniTable(
        slide,
        [
          ["Question", "Evidence used in the paper", "What it tells us"],
          ["Are the generated examples valid?", "Human labels, LFR, SLFR", "Whether labels actually flip"],
          ["Are they diverse?", "Self-BLEU and OTDD", "Whether data adds new information"],
          ["Does training improve robustness?", "Stress tests, HANS, MNLI-hard, QNLI", "Whether shortcuts are weakened"],
          ["Does the model respond to edits?", "Sensitivity and pair accuracy", "Whether predictions track causal changes"],
        ],
        55,
        190,
        1080,
        64,
      );
      addText(slide, "This is why the results section should not be read as one accuracy table. It is a chain of evidence.", 80, 555, 1000, 36, {
        fontSize: 24,
        bold: true,
      });
    },
    `
This slide makes the experiment design easier to understand. The paper does not simply report one benchmark accuracy. It builds a chain of evidence. First, are the generated counterfactuals valid and diverse? Second, do models trained with this data improve under robustness and out-of-distribution evaluation? Third, do they behave more consistently on original-counterfactual pairs? This connects the experiments back to shortcut learning.
`,
  );

  addRouteSlide(
    presentation,
    17,
    "Generated counterfactuals are at least as valid and more diverse than human CAD.",
    "Results: Data Quality",
    (slide) => {
      addMetric(slide, "83.14", "DISCO label-flip rate vs. 82.55 for Human-CAD", 42, 210, 250, ACCENT);
      addMetric(slide, "93.33", "DISCO soft label-flip rate vs. 88.24 for Human-CAD", 328, 210, 250, ACCENT);
      addMetric(slide, "0.23", "DISCO Self-BLEU vs. 0.79 for Human-CAD; lower means more diverse", 614, 210, 250, PURPLE);
      addMetric(slide, "240", "DISCO OTDD distance vs. 180 for Human-CAD", 900, 210, 250, PURPLE);
      addText(slide, "Interpretation", 42, 432, 180, 30, { fontSize: 21, bold: true, color: MUTED });
      addText(slide, "The generated examples flip labels at a similar or better rate, while covering a wider range of perturbations.", 42, 472, 980, 70, {
        fontSize: 32,
        bold: true,
      });
    },
    `
Table 1 compares DISCO with human-written counterfactual data. The label-flip rate is slightly higher for DISCO on average: 83.14 versus 82.55. The soft label-flip rate is also higher, meaning many edits at least change the label away from the original even if the exact target label is debated. Self-BLEU is much lower for DISCO, which means the examples are lexically more diverse. OTDD distance is higher, suggesting the new examples contain more different information from the original examples.
`,
  );

  addRouteSlide(
    presentation,
    18,
    "Table 1 is easier to read through three headline numbers.",
    "Results: Data Quality",
    (slide) => {
      slide.images.add({
        blob: table1,
        contentType: "image/png",
        alt: "Table 1 from Chen et al. comparing DISCO and Human-CAD data quality.",
        fit: "contain",
        position: { left: 70, top: 192, width: 330, height: 270 },
      });
      addText(slide, "Paper table as background evidence", 78, 486, 310, 22, { fontSize: 16, color: MUTED });
      addArrowMetric(slide, "Label flip rate", "82.55", "83.14", 470, 204, 300, "Human-CAD -> DISCO");
      addArrowMetric(slide, "Self-BLEU", "0.79", "0.23", 835, 204, 300, "lower means more diverse");
      addArrowMetric(slide, "OTDD distance", "180", "240", 655, 372, 300, "higher means more information-rich");
      addText(slide, "DISCO is not just cheaper than Human-CAD; it is comparably valid and more diverse.", 455, 542, 720, 46, {
        fontSize: 28,
        bold: true,
      });
      addText(slide, "Source: Table 1 in Chen et al. (2023).", 75, 620, 420, 20, { fontSize: 12, color: MUTED });
    },
    `
This slide keeps Table 1 only as background evidence and enlarges the numbers that matter for the audience. The average label-flip rate moves from 82.55 for Human-CAD to 83.14 for DISCO. Self-BLEU drops from 0.79 to 0.23, which means DISCO examples are more diverse. OTDD rises from 180 to 240, suggesting the generated examples add richer distributional information.
`,
  );

  addRouteSlide(
    presentation,
    19,
    "The robustness gain is clearest when Table 4 is simplified.",
    "Results: Robustness",
    (slide) => {
      addArrowMetric(slide, "SNLI-subset robustness", "71.0", "77.5", 86, 210, 330, "+6.5 points");
      addArrowMetric(slide, "WANLI robustness", "65.9", "75.1", 476, 210, 330, "+9.2 points");
      addBox(slide, 866, 210, 330, 116, "#F7F7F7", RULE);
      addText(slide, "OOD generalization", 884, 226, 250, 24, { fontSize: 17, bold: true, color: MUTED });
      addText(slide, "+2.7", 884, 260, 90, 38, { fontSize: 31, bold: true, color: BLUE });
      addText(slide, "SNLI-subset", 976, 267, 110, 22, { fontSize: 15, color: MUTED });
      addText(slide, "+2.1", 1084, 260, 90, 38, { fontSize: 31, bold: true, color: BLUE });
      addText(slide, "WANLI", 1172, 267, 80, 22, { fontSize: 15, color: MUTED });
      addBox(slide, 214, 405, 855, 112, "#FFFFFF", RULE);
      addText(slide, "Main reading", 242, 430, 160, 28, { fontSize: 20, bold: true, color: MUTED });
      addText(slide, "DISCO helps most where shortcut-sensitive robustness is tested; OOD gains are smaller but still positive.", 420, 424, 600, 50, {
        fontSize: 25,
        bold: true,
      });
      addText(slide, "Source: Table 4 in Chen et al. (2023).", 80, 620, 460, 20, { fontSize: 12, color: MUTED });
    },
    `
This slide simplifies Table 4 instead of asking the audience to read the dense original tables. The robustness average moves from 71.0 to 77.5 in the SNLI-subset setting, and from 65.9 to 75.1 in the WANLI setting. OOD generalization also improves, but by smaller margins: 2.7 and 2.1 points. This pattern makes sense because counterfactual augmentation directly targets shortcut behavior.
`,
  );

  addRouteSlide(
    presentation,
    20,
    "Counterfactual evaluation asks whether the model changes its mind for the right reason.",
    "Results: Consistency",
    (slide) => {
      addBox(slide, 42, 205, 500, 250, "#F7F7F7", RULE);
      addText(slide, "Counterfactual sensitivity", 72, 238, 420, 32, { fontSize: 29, bold: true });
      addText(slide, "How confidently does the model shift when the causal context changes?", 72, 292, 390, 72, { fontSize: 24, color: MUTED });
      addText(slide, "Higher is better.", 72, 390, 220, 28, { fontSize: 22, bold: true, color: ACCENT });
      addBox(slide, 680, 205, 500, 250, "#F7F7F7", RULE);
      addText(slide, "Pair accuracy", 710, 238, 420, 32, { fontSize: 29, bold: true });
      addText(slide, "The prediction counts as correct only if both the original and counterfactual example are correct.", 710, 292, 390, 82, {
        fontSize: 24,
        color: MUTED,
      });
      addText(slide, "Stricter than ordinary accuracy.", 710, 392, 320, 28, { fontSize: 22, bold: true, color: ACCENT });
      addText(slide, "DISCO improves both metrics across Human-CAD, SNLI-hard counterfactual pairs, and WANLI counterfactual pairs.", 135, 522, 1010, 44, {
        fontSize: 30,
        bold: true,
        alignment: "center",
      });
    },
    `
The counterfactual evaluation is conceptually important. Ordinary accuracy checks one example at a time. Counterfactual pair accuracy is stricter: the model must get the original example right and the edited example right. Sensitivity measures whether the model's confidence shifts when the causal context changes. If a model relies on a shortcut, it may ignore the edit and keep the same prediction. DISCO improves both metrics, meaning the model becomes more responsive to the relevant context change.
`,
  );

  addRouteSlide(
    presentation,
    21,
    "Two WANLI gains show the strongest link to shortcut mitigation.",
    "Results: Consistency",
    (slide) => {
      addArrowMetric(slide, "WANLI pair accuracy", "34.6", "52.7", 102, 225, 390, "+18.1 points");
      addArrowMetric(slide, "WANLI sensitivity", "44.9", "57.6", 102, 385, 390, "+12.7 points");
      addText(slide, "Why these metrics matter", 620, 220, 420, 32, { fontSize: 28, bold: true });
      addBullets(
        slide,
        [
          "Sensitivity asks whether predictions shift after a causal edit.",
          "Pair accuracy counts success only when both examples are correct.",
          "These gains mean the model is less likely to ignore counterfactual context changes.",
        ],
        620,
        280,
        500,
        150,
        22,
      );
      addText(slide, "Source: Table 5 in Chen et al. (2023), WANLI baseline vs. WANLI + DISCO average gains.", 102, 552, 720, 20, { fontSize: 12, color: MUTED });
      addText(slide, "This is the most direct evidence that the model is less stuck on superficial cues.", 210, 598, 850, 34, {
        fontSize: 25,
        bold: true,
        alignment: "center",
      });
    },
    `
Table 5 is the most relevant table for the course theme, but the full table is dense. This slide pulls out the WANLI baseline comparison from Table 4 and Table 5's counterfactual evaluation logic: pair accuracy improves from 34.6 to 52.7, and sensitivity improves from 44.9 to 57.6. If a model relies on shortcuts, it may ignore a counterfactual edit. These gains suggest the model is more responsive to the meaningful context change.
`,
  );

  addRouteSlide(
    presentation,
    22,
    "The limitations are mainly about scope, cost, and inherited bias.",
    "Limitations",
    (slide) => {
      addBullets(
        slide,
        [
          "The experiments are limited to English NLI.",
          "Only one LLM, GPT-3, is used for the generation pipeline.",
          "Large-scale prompt ablations are expensive, so the prompt design is not fully isolated.",
          "Human evaluation covers about 500 random instances.",
          "The teacher model can pass its own blind spots into the filtered dataset.",
        ],
        64,
        200,
        810,
        270,
        25,
      );
      addBox(slide, 940, 220, 238, 250, "#F7F7F7", RULE);
      addText(slide, "Critical point", 968, 254, 180, 28, { fontSize: 21, bold: true, color: MUTED });
      addText(slide, "DISCO reduces one kind of shortcut risk, but it does not prove that the model has human-like reasoning.", 968, 300, 176, 114, {
        fontSize: 24,
        bold: true,
      });
    },
    `
The authors are careful about limitations. The experiments are only in English and only on NLI. The generation pipeline uses one LLM, GPT-3, and prompt ablations at scale are expensive. Human evaluation is limited to around 500 examples. I would add another important limitation: the teacher model is not neutral. If the teacher has blind spots, filtering can preserve them. So DISCO is a promising mitigation method, but it does not prove that the student model truly reasons like a human.
`,
  );

  addRouteSlide(
    presentation,
    23,
    "DISCO supports responsible AI, but it is not a fairness guarantee.",
    "Impacts",
    (slide) => {
      addBox(slide, 58, 196, 520, 290, "#F7F7F7", RULE);
      addText(slide, "Positive impact", 88, 226, 320, 34, { fontSize: 29, bold: true, color: ACCENT });
      addBullets(
        slide,
        [
          "Scalable counterfactual data creation.",
          "Reduces reliance on superficial shortcuts.",
          "Helps robustness and OOD generalization.",
          "Uses LLMs as debiasing data generators, not only predictors.",
        ],
        88,
        286,
        440,
        154,
        20,
      );

      addBox(slide, 660, 196, 520, 290, "#F7F7F7", RULE);
      addText(slide, "Risks and responsible use", 690, 226, 390, 34, { fontSize: 29, bold: true, color: INK });
      addBullets(
        slide,
        [
          "Generated data can inherit biases from GPT-3.",
          "Teacher filtering can preserve the teacher model's own shortcuts.",
          "Robustness gains do not automatically imply fairness.",
          "Synthetic data should be audited, especially in high-stakes NLP.",
        ],
        690,
        286,
        432,
        168,
        20,
      );

      addBox(slide, 150, 535, 980, 72, "#FFFFFF", RULE);
      addText(
        slide,
        "Shortcut mitigation becomes more scalable, but fairness still needs auditing.",
        180,
        555,
        920,
        34,
        { fontSize: 25, bold: true, alignment: "center" },
      );
    },
    `
For impact, I would connect DISCO to responsible AI carefully. It is a step toward responsible AI because it makes shortcut mitigation more scalable and improves robustness against shortcut-sensitive evaluations. But it does not guarantee fairness. Generated data can inherit bias from GPT-3, and teacher filtering can preserve the teacher model's own shortcuts. So synthetic counterfactual data should still be audited, especially in high-stakes NLP.
`,
  );

  addRouteSlide(
    presentation,
    24,
    "The takeaway: generate broadly, filter conservatively, evaluate contrastively.",
    "Closing",
    (slide) => {
      addText(slide, "1", 80, 210, 60, 52, { fontSize: 48, bold: true, color: ACCENT });
      addText(slide, "Shortcut learning is a data problem, not only a model problem.", 160, 220, 880, 42, { fontSize: 31, bold: true });
      addText(slide, "2", 80, 330, 60, 52, { fontSize: 48, bold: true, color: ACCENT });
      addText(slide, "Counterfactual examples make the causal feature visible.", 160, 340, 880, 42, { fontSize: 31, bold: true });
      addText(slide, "3", 80, 450, 60, 52, { fontSize: 48, bold: true, color: ACCENT });
      addText(slide, "DISCO is effective because it pairs LLM diversity with task-specific filtering.", 160, 460, 920, 42, { fontSize: 31, bold: true });
      addText(slide, "Discussion question: when should we trust generated counterfactual data enough to train with it?", 160, 588, 850, 34, {
        fontSize: 22,
        color: MUTED,
      });
    },
    `
To close, I would leave the audience with three ideas. First, shortcut learning is strongly connected to the dataset, not just to the model architecture. Second, counterfactual examples are useful because they reveal which parts of the input should causally control the label. Third, DISCO works because it does not rely only on generation. It generates broadly, filters conservatively, and evaluates contrastively. A good discussion question is: when is generated counterfactual data trustworthy enough to become training data?
`,
  );

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    const png = await presentation.export({ slide, format: "png", scale: 1 });
    await writeBlob(path.join(QA_DIR, `${stem}.png`), png);
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(QA_DIR, `${stem}.layout.json`), await layout.text(), "utf8");
  }

  const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
  await writeBlob(path.join(QA_DIR, "montage.webp"), montage);
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(FINAL_PPTX);
  console.log(FINAL_PPTX);
}

build().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
