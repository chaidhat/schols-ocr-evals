import { writeFile, mkdir } from "fs/promises";
import { existsSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

const REPO_ID = "typhoon-ai/ThaiOCRBench";
const BASE_URL = `https://huggingface.co/datasets/${REPO_ID}/resolve/main`;
const DATA_DIR = join(__dirname, "data");

const FILES = [
  "data/test-00000-of-00004.parquet",
  "data/test-00001-of-00004.parquet",
  "data/test-00002-of-00004.parquet",
  "data/test-00003-of-00004.parquet",
];

async function downloadFile(remotePath: string, localPath: string) {
  if (existsSync(localPath)) {
    console.log(`  Skipping ${remotePath} (already exists)`);
    return;
  }

  const url = `${BASE_URL}/${remotePath}`;
  console.log(`  Downloading ${remotePath}...`);

  const response = await fetch(url, { redirect: "follow" });
  if (!response.ok) {
    throw new Error(`Failed to download ${url}: ${response.status} ${response.statusText}`);
  }

  const buffer = Buffer.from(await response.arrayBuffer());
  await writeFile(localPath, buffer);

  const sizeMB = (buffer.length / 1024 / 1024).toFixed(1);
  console.log(`  Saved ${remotePath} (${sizeMB} MB)`);
}

async function main() {
  console.log(`Downloading dataset: ${REPO_ID}`);
  console.log(`Destination: ${DATA_DIR}\n`);

  await mkdir(DATA_DIR, { recursive: true });

  for (const file of FILES) {
    const filename = file.split("/").pop()!;
    const localPath = join(DATA_DIR, filename);
    await downloadFile(file, localPath);
  }

  console.log("\nDone! All parquet files downloaded.");
}

main().catch((err) => {
  console.error("Setup failed:", err);
  process.exit(1);
});
