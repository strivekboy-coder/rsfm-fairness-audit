import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const pptxPath = process.env.PPTX_PATH || "D:/Codex/rsfm-fairness-audit/outputs/disco_nlp_shortcuts_biases_presentation.pptx";
const outDir = process.env.RENDER_DIR || "D:/Codex/rsfm-fairness-audit/work/nlp_pre/ppt_tmp/final_render";

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(outDir, { recursive: true });
  const presentation = await PresentationFile.importPptx(await FileBlob.load(pptxPath));
  const inspect = await presentation.inspect({ kind: "slide,notes,textbox,image,chart", maxChars: 500000 });
  await fs.writeFile(path.join(outDir, "inspect.ndjson"), inspect.ndjson, "utf8");
  for (const [index, slide] of presentation.slides.items.entries()) {
    const png = await presentation.export({ slide, format: "png", scale: 1 });
    await writeBlob(path.join(outDir, `slide-${String(index + 1).padStart(2, "0")}.png`), png);
  }
  console.log(`slides=${presentation.slides.items.length}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
