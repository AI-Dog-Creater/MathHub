import express from "express";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";
import { dirname } from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";
import multer from "multer";
import mammoth from "mammoth";
import sharp from "sharp";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

dotenv.config();

// Set up polyfills BEFORE pdfjs-dist is imported (must be at module scope, before any import uses it)
(global as any).DOMMatrix = class DOMMatrix {
  a = 1; b = 0; c = 0; d = 1; e = 0; f = 0;
  isIdentity = true;
  constructor(_?: string | number[]) {}
  multiply() { return this; }
  translate() { return this; }
  scale() { return this; }
  rotate() { return this; }
};
(global as any).HTMLCanvasElement = class {};
(global as any).Image = class {};

// pdfjs-dist is loaded dynamically inside startServer() so polyfills are ready first
let pdfjsLib: any = null;

async function startServer() {
  // Dynamically import pdfjs-dist here (after polyfills are set)
  const pdfjsModule = await import("pdfjs-dist/legacy/build/pdf.mjs");
  pdfjsLib = pdfjsModule;

  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // Setup Gemini
  const apiKey = process.env.GEMINI_API_KEY || "";
  const ai = new GoogleGenAI({
    apiKey: apiKey,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });

  // API Route for testing integration connection for Alibaba Cloud Bailian
  app.post("/api/test-bailian", async (req, res) => {
    try {
      const { apiKey, model } = req.body;
      if (!apiKey || apiKey.trim() === "") {
        return res.status(400).json({ error: "API Key 不能为空！" });
      }
      
      const testedModel = model || "qwen-plus";

      console.log(`[DashScope] Dispatching verification handshake to model ${testedModel}`);
      const requestBody = {
        model: testedModel,
        messages: [
          { role: "user", content: "请只返回纯文本: OK" }
        ],
        max_tokens: 15
      };

      const response = await fetch("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${apiKey.trim()}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify(requestBody)
      });

      const data: any = await response.json();
      if (!response.ok) {
        const errMsg = data.error?.message || JSON.stringify(data);
        return res.status(response.status).json({ error: `验证失败 - 阿里云百炼报错信息：${errMsg}` });
      }

      const reply = data.choices?.[0]?.message?.content || "";
      if (reply) {
        return res.json({ message: `连接测试成功！🎉\n已成功连通您的私有阿里云百炼算力接口。\n当前引擎：${testedModel}\n百炼验证即时回执："${reply.trim()}"` });
      } else {
        return res.status(500).json({ error: "连接校验失败，百炼接口未返回有效会话。请检查额度余额。" });
      }
    } catch (error: any) {
      console.error("Test connection exception:", error);
      res.status(500).json({ error: `网关层面连接测试异常：${error.message || error}` });
    }
  });

  // API Route for testing Vision model connection (qwen-vl / qwen3-vl series)
  app.post("/api/test-vision", async (req, res) => {
    try {
      const { apiKey, model } = req.body;
      if (!apiKey || apiKey.trim() === "") {
        return res.status(400).json({ error: "API Key 不能为空！" });
      }

      const testedModel = model || "qwen-vl-plus-latest";
      console.log(`[DashScope] Vision test for model ${testedModel}`);

      const requestBody = {
        model: testedModel,
        messages: [
          {
            role: "user",
            content: [
              { type: "image_url", image_url: { url: "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg" } },
              { type: "text", text: "请用一句话描述这张图片。" }
            ]
          }
        ],
        max_tokens: 50
      };

      const response = await fetch("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${apiKey.trim()}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify(requestBody)
      });

      let data: any = {};
      const text = await response.text();
      if (text) {
        try { data = JSON.parse(text); } catch {}
      }

      if (!response.ok) {
        const errMsg = data.error?.message || data.error?.code || JSON.stringify(data).slice(0, 200);
        return res.status(response.status).json({ error: `验证失败 - 阿里云百炼报错：${errMsg}` });
      }

      const reply = data.choices?.[0]?.message?.content || "";
      if (reply) {
        return res.json({ message: `视觉模型连通测试成功！\n已成功连通阿里云百炼视觉模型接口。\n当前模型：${testedModel}\n百炼回复："${reply.trim().slice(0, 100)}"` });
      } else {
        return res.status(500).json({ error: "视觉模型连通失败，百炼接口未返回有效内容。请检查额度余额。" });
      }
    } catch (error: any) {
      console.error("[/api/test-vision] exception:", error);
      res.status(500).json({ error: `网关层面连接测试异常：${error.message || error}` });
    }
  });

  app.post("/api/test-imagegen", async (req, res) => {
    try {
      const { apiKey, model } = req.body;
      if (!apiKey || apiKey.trim() === "") {
        return res.status(400).json({ error: "API Key 不能为空！" });
      }

      const testedModel = model || "wan2.7-image";

      console.log(`[DashScope] ImageGen test for model ${testedModel}`);

      const imageModels = ["wan2.7-image", "wan2.7-image-pro", "wordart-semantic", "wordart-texture"];
      const isImageModel = imageModels.includes(testedModel);

      const endpoint = isImageModel
        ? "https://dashscope.aliyuncs.com/api/v1/services/aigc/image-generation/generation"
        : "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation";

      const requestBody: any = {
        model: testedModel,
        input: {
          messages: [{ role: "user", content: [{ text: "a cute cat" }] }]
        },
        parameters: { n: 1 }
      };

      // Step 1: Submit async task (X-DashScope-Async header is REQUIRED)
      const submitRes = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${apiKey.trim()}`,
          "Content-Type": "application/json",
          "X-DashScope-Async": "enable"
        },
        body: JSON.stringify(requestBody)
      });

      const submitData: any = await submitRes.json();

      if (!submitRes.ok && !submitData.output?.task_id) {
        const errMsg = submitData.error?.message || submitData.message || JSON.stringify(submitData);
        return res.status(submitRes.status).json({ error: `验证失败 - 阿里云百炼报错信息：${errMsg}` });
      }

      const taskId = submitData.output?.task_id;
      if (!taskId) {
        return res.status(400).json({ error: "提交测试任务失败，未获得 task_id" });
      }

      console.log(`[DashScope] Test task submitted: ${taskId}, polling...`);

      // Step 2: Poll for completion (up to 3 minutes)
      const maxAttempts = 90;
      let finalStatus = "";
      let outputUrl = "";

      for (let attempt = 0; attempt < maxAttempts; attempt++) {
        await new Promise(r => setTimeout(r, 2000));

        const statusRes = await fetch(`https://dashscope.aliyuncs.com/api/v1/tasks/${taskId}`, {
          method: "GET",
          headers: {
            "Authorization": `Bearer ${apiKey.trim()}`,
            "Content-Type": "application/json"
          }
        });

        const statusData: any = await statusRes.json();
        const state = statusData.output?.task_status || statusData.output?.state || "";

        console.log(`[DashScope] Test poll ${attempt + 1}: state=${state}`);

        if (state === "SUCCEEDED" || state === "succeeded" || state === "COMPLETED") {
          // Official response shape: output.choices[0].message.content[0].image
          outputUrl = statusData.output?.choices?.[0]?.message?.content?.[0]?.image
            || statusData.output?.output?.url
            || statusData.output?.image
            || statusData.output?.url
            || "";
          finalStatus = "succeeded";
          break;
        }

        if (state === "FAILED" || state === "failed" || state === "ERROR" || state === "failed") {
          finalStatus = "failed";
          const errMsg = statusData.output?.error?.message || statusData.error?.message || JSON.stringify(statusData);
          return res.status(500).json({ error: `生成任务失败：${errMsg}` });
        }

        if (attempt === maxAttempts - 1) {
          return res.status(504).json({ error: `生成超时（超过 3 分钟），请稍后重试。任务ID：${taskId}` });
        }
      }

      if (finalStatus === "succeeded" && outputUrl) {
        console.log(`[DashScope] Test succeeded. URL: ${outputUrl.slice(0, 80)}...`);
        return res.json({
          message: `连通测试成功！\n已成功连通阿里云百炼生成模型接口。\n当前模型：${testedModel}\n测试图片已生成完成！`
        });
      }

      return res.status(500).json({ error: "生成完成但未返回媒体 URL，请检查模型配置。" });
    } catch (error: any) {
      console.error("[/api/test-imagegen] exception:", error);
      res.status(500).json({ error: `网关层面连接测试异常：${error.message || error}` });
    }
  });

  // Video generation connection test endpoint
  app.post("/api/test-videogen", async (req, res) => {
    try {
      const { apiKey, model } = req.body;
      if (!apiKey || apiKey.trim() === "") {
        return res.status(400).json({ error: "API Key 不能为空！" });
      }

      const testedModel = model || "wan2.7-t2v-2026-04-25";
      console.log(`[DashScope] VideoGen test for model ${testedModel}`);

      const requestBody = {
        model: testedModel,
        input: { prompt: "A cute cat running in a sunny park" },
        parameters: { duration: 5, resolution: "720P", ratio: "16:9", prompt_extend: true, watermark: false }
      };

      const submitRes = await fetch("https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${apiKey.trim()}`,
          "Content-Type": "application/json",
          "X-DashScope-Async": "enable"
        },
        body: JSON.stringify(requestBody)
      });

      let submitData: any = {};
      if (submitRes.ok) {
        const text = await submitRes.text();
        if (text) {
          try { submitData = JSON.parse(text); } catch {}
        }
      } else {
        const errText = await submitRes.text();
        console.error(`[DashScope] Video test submit failed (${submitRes.status}):`, errText.slice(0, 200));
        return res.status(502).json({ error: `阿里云百炼网关拒绝请求（HTTP ${submitRes.status}），请检查 API Key 是否有效。` });
      }

      const taskId = submitData.output?.task_id;

      if (!taskId) {
        const errMsg = submitData.error?.message || submitData.message || JSON.stringify(submitData);
        return res.status(400).json({ error: `验证失败 - 阿里云百炼报错信息：${errMsg}` });
      }

      console.log(`[DashScope] Video test task submitted: ${taskId}, polling...`);

      const maxAttempts = 90;
      let finalStatus = "";
      let outputUrl = "";

      for (let attempt = 0; attempt < maxAttempts; attempt++) {
        await new Promise(r => setTimeout(r, 2000));

        const statusRes = await fetch(`https://dashscope.aliyuncs.com/api/v1/tasks/${taskId}`, {
          method: "GET",
          headers: {
            "Authorization": `Bearer ${apiKey.trim()}`,
            "Content-Type": "application/json"
          }
        });

        let statusData: any = {};
        if (statusRes.ok) {
          const stext = await statusRes.text();
          if (stext) {
            try { statusData = JSON.parse(stext); } catch {}
          }
        } else {
          console.error(`[DashScope] Video test poll failed (${statusRes.status})`);
        }

        const state = statusData.output?.task_status || statusData.output?.state || "";

        console.log(`[DashScope] Video test poll ${attempt + 1}: state=${state}`);

        if (state === "SUCCEEDED" || state === "succeeded" || state === "COMPLETED") {
          outputUrl = statusData.output?.choices?.[0]?.message?.content?.[0]?.video?.video_url
            || statusData.output?.output?.url
            || statusData.output?.video_url
            || statusData.output?.url
            || "";
          finalStatus = "succeeded";
          break;
        }

        if (state === "FAILED" || state === "failed" || state === "ERROR") {
          finalStatus = "failed";
          const errMsg = statusData.output?.error?.message || statusData.error?.message || JSON.stringify(statusData);
          return res.status(500).json({ error: `视频生成任务失败：${errMsg}` });
        }

        if (attempt === maxAttempts - 1) {
          return res.status(504).json({ error: `视频生成超时（超过 3 分钟），请稍后重试。任务ID：${taskId}` });
        }
      }

      if (finalStatus === "succeeded" && outputUrl) {
        console.log(`[DashScope] Video test succeeded. URL: ${outputUrl.slice(0, 80)}...`);
        return res.json({
          message: `连通测试成功！\n已成功连通阿里云百炼视频生成模型接口。\n当前模型：${testedModel}\n测试视频已生成完成！`
        });
      }

      return res.status(500).json({ error: "视频生成完成但未返回媒体 URL，请检查模型配置。" });
    } catch (error: any) {
      console.error("[/api/test-videogen] exception:", error);
      res.status(500).json({ error: `网关层面连接测试异常：${error.message || error}` });
    }
  });

  const sendChunk = (res: express.Response, text: string) => {
    res.write(`data: ${JSON.stringify({ text })}\n\n`);
  };

  const sendErrorChunk = (res: express.Response, error: string) => {
    res.write(`data: ${JSON.stringify({ error })}\n\n`);
  };

  const CHUNK_INTERVAL = 5; // 每 20 个字符推送一次给前端

  const streamDashScopeChat = async (
    res: express.Response,
    apiKeyValue: string,
    selectedModel: string,
    formattedMessages: any[]
  ) => {
    const response = await fetch("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKeyValue.trim()}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: selectedModel,
        messages: formattedMessages,
        temperature: 0.7,
        stream: true
      })
    });

    if (!response.ok || !response.body) {
      const data: any = await response.json().catch(() => ({}));
      const errMsg = data.error?.message || JSON.stringify(data);
      console.error("[DashScope] API error:", errMsg);
      throw new Error(`阿里云百炼大模型层报错：${errMsg}`);
    }

    console.log("[DashScope] Stream started");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let fullText = "";
    let lastSendLen = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      while (buffer.includes("\n")) {
        const newlineIdx = buffer.indexOf("\n");
        const line = buffer.slice(0, newlineIdx);
        buffer = buffer.slice(newlineIdx + 1);

        const trimmed = line.trim();
        if (!trimmed) continue;
        if (trimmed === "data: [DONE]") continue;

        if (trimmed.startsWith("data: ")) {
          const dataStr = trimmed.slice(6);
          try {
            const parsed = JSON.parse(dataStr);
            const delta = parsed.choices?.[0]?.delta?.content || parsed.choices?.[0]?.message?.content || "";
            if (delta) {
              fullText += delta;
              if (fullText.length - lastSendLen >= CHUNK_INTERVAL) {
                sendChunk(res, fullText);
                lastSendLen = fullText.length;
              }
            }
          } catch {
            // ignore parse errors
          }
        }
      }
    }

    // Push any remaining text that hasn't been flushed yet
    if (fullText.length > lastSendLen) {
      sendChunk(res, fullText);
    }

    console.log(`[DashScope] Stream done. fullText length: ${fullText.length}`);
    return fullText;
  };

  const streamGeminiChat = async (res: express.Response, contents: any[], systemInstruction: string) => {
    console.log("[Gemini] Stream started");
    const stream = await ai.models.generateContentStream({
      model: "gemini-3.5-flash",
      contents,
      config: {
        systemInstruction,
        temperature: 0.7
      }
    });

    let fullText = "";
    let lastSendLen = 0;
    for await (const chunk of stream) {
      const delta = chunk.text || "";
      if (delta) {
        fullText += delta;
        if (fullText.length - lastSendLen >= CHUNK_INTERVAL) {
          sendChunk(res, fullText);
          lastSendLen = fullText.length;
        }
      }
    }

    if (fullText.length > lastSendLen) {
      sendChunk(res, fullText);
    }

    console.log(`[Gemini] Stream done. fullText length: ${fullText.length}`);
    return fullText;
  };

  // API Route for chat
  app.post("/api/chat", async (req, res) => {
    try {
      const { messages, customApiKey, customModel } = req.body;
      if (!Array.isArray(messages)) {
        return res.status(400).json({ error: "messages array is required" });
      }

      console.log("[API /api/chat] Request received. customApiKey:", customApiKey ? "provided" : "empty", "customModel:", customModel);

      res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
      res.setHeader("Cache-Control", "no-cache, no-transform");
      res.setHeader("Connection", "keep-alive");
      res.setHeader("X-Accel-Buffering", "no");
      res.flushHeaders();

      const clientSystemMessage = messages.find((m: any) => m.role === "system");
      const clientOtherMessages = messages.filter((m: any) => m.role !== "system");

      const defaultSystemPrompt = "You are a professional, helpful mathematical modeling tutor. You speak fluent Chinese (简体中文). Help modeling beginners choose suitable models, write Python/Matlab code, explain concepts with LaTeX or plain-text formulas, and structure competition papers. Focus on logical clarity, clear steps, and helpful educational advice.";
      const mathFormattingPrompt = `${defaultSystemPrompt}\n\nWhen you output any mathematical formula, you must use standard Markdown math syntax so the frontend can render it correctly. Use inline math with $...$ and display math with $$...$$ on its own lines. Do not use square brackets [ ... ] to wrap formulas. Do not use code fences for formulas unless you are giving code. Prefer display math for multi-line equations and keep surrounding explanation in normal Chinese paragraphs or bullet points. If there are multiple formulas, separate them clearly and keep notation consistent.`;

      if (customApiKey && customApiKey.trim() !== "") {
        const selectedModel = customModel || "qwen-plus";
        console.log(`[DashScope] Routing user prompt to Alibaba Cloud Bailian model: ${selectedModel}`);

        const formattedMessages = [
          {
            role: "system",
            content: clientSystemMessage?.content || defaultSystemPrompt
          },
          ...clientOtherMessages.map((m: any) => ({
            role: m.role,
            content: m.content
          }))
        ];

        const finalText = await streamDashScopeChat(res, customApiKey, selectedModel, formattedMessages);
        if (!finalText.trim()) {
          sendChunk(res, "抱歉，服务端未收到有效回复内容，请重试。");
        }
        res.write(`data: [DONE]\n\n`);
        res.end();
        return;
      }

      if (!apiKey) {
        throw new Error("API key is not configured. Please supply GEMINI_API_KEY in Settings.");
      }

      const contents = clientOtherMessages.map((msg: any) => ({
        role: msg.role === "assistant" ? "model" : "user",
        parts: [{ text: msg.content }]
      }));

      const finalText = await streamGeminiChat(res, contents, clientSystemMessage?.content || mathFormattingPrompt);
      if (!finalText.trim()) {
        sendChunk(res, "抱歉，服务端未收到有效回复内容，请重试。");
      }
      res.write(`data: [DONE]\n\n`);
      res.end();
    } catch (error: any) {
      console.error("Chat API Error:", error);
      if (!res.headersSent) {
        res.status(500).json({ error: error.message || "Internal Server Error" });
      } else {
        sendErrorChunk(res, error.message || "Internal Server Error");
        res.write(`data: [DONE]\n\n`);
        res.end();
      }
    }
  });

  // Multer config: store files in /uploads
  const uploadDir = path.join(process.cwd(), "uploads");
  if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, { recursive: true });
  const upload = multer({
    storage: multer.diskStorage({
      destination: uploadDir,
      filename: (_req, file, cb) => {
        cb(null, `${Date.now()}-${file.originalname}`);
      },
    }),
    limits: { fileSize: 20 * 1024 * 1024 }, // 20MB
  });

  // File upload + parse endpoint
  app.post("/api/upload", upload.single("file"), async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: "未检测到文件" });
      }

      const {
        visionApiKey: _unused1,
        visionModel: _unused2,
      } = req.body as { visionApiKey?: string; visionModel?: string };
      const filePath = req.file.path;
      const ext = path.extname(req.file.originalname).toLowerCase();
      let text = "";

      if ([".png", ".jpg", ".jpeg", ".gif", ".webp"].includes(ext)) {
        // Process image: resize + OCR via tesseract.js (local, no API needed)
        try {
          const buffer = await sharp(filePath)
            .resize({ width: 2000, height: 2000, fit: "inside" })
            .jpeg({ quality: 90 })
            .toBuffer();

          const Tesseract = (await import("tesseract.js")).default;
          const { data } = await Tesseract.recognize(buffer, "eng+chi_sim", {
            logger: (m: any) => {
              if (m.status === "recognizing text") {
                process.stdout.write(`\r[OCR] ${Math.round(m.progress * 100)}%`);
              }
            }
          });
          console.log(); // newline after progress

          const extracted = (data.text || "").trim();
          if (extracted) {
            text = `[图片文件: ${req.file.originalname}]\n[通过本地 OCR 识别]\n${extracted}`;
          } else {
            text = `[图片文件: ${req.file.originalname}]\n[OCR 未能识别出文字内容，图片可能是纯图形/截图或清晰度不足。]`;
          }
        } catch (err: any) {
          console.error("[/api/upload] Image OCR error:", err);
          text = `[图片文件: ${req.file.originalname}]\n[图片文字识别失败: ${err.message}]`;
        }
      } else if (ext === ".pdf") {
        // Process PDF: use pdfjs-dist text extraction (local, no API needed)
        try {
          const dataBuffer = fs.readFileSync(filePath);
          const loadingTask = pdfjsLib.getDocument({ data: new Uint8Array(dataBuffer) });
          const pdf = await loadingTask.promise;
          const pageCount = pdf.numPages;

          const pages: string[] = [];
          for (let i = 1; i <= pageCount; i++) {
            const page = await pdf.getPage(i);
            const content = await page.getTextContent();
            const pageText = content.items
              .map((item: any) => item.str || "")
              .join(" ")
              .replace(/\s+/g, " ")
              .trim();
            pages.push(pageText || `[第 ${i} 页未检测到可提取的文字内容]`);
          }

          const extracted = pages.join("\n\n").trim();
          if (extracted) {
            text = `[PDF文件: ${req.file.originalname}]（共 ${pageCount} 页）\n[通过本地 PDF 解析]\n${extracted}`;
          } else {
            text = `[PDF文件: ${req.file.originalname}]\n[PDF 内未检测到可提取的文字内容，该 PDF 可能为纯图片扫描件，建议另存为图片后使用图片 OCR 识别。]`;
          }
        } catch (err: any) {
          console.error("[/api/upload] PDF parse error:", err);
          text = `[PDF文件: ${req.file.originalname}]\n[PDF 解析失败: ${err.message}]`;
        }
      } else if (ext === ".docx" || ext === ".doc") {
        const result = await mammoth.extractRawText({ path: filePath });
        text = result.value.trim() || `[Word文件: ${req.file.originalname}]\n[文字提取结果为空]`;
      } else if (ext === ".csv" || ext === ".xlsx" || ext === ".xls") {
        const buffer = fs.readFileSync(filePath);
        text = buffer.toString("utf-8").slice(0, 10000);
      } else {
        text = fs.readFileSync(filePath, "utf-8").slice(0, 10000);
      }

      // Clean up uploaded file after parsing
      fs.unlinkSync(filePath);

      // Truncate very long content
      if (text.length > 15000) {
        text = text.slice(0, 15000) + "\n\n[文件内容过长，已截断...]";
      }

      res.json({
        filename: req.file.originalname,
        mimeType: req.file.mimetype,
        size: req.file.size,
        content: text.trim(),
      });
    } catch (err: any) {
      console.error("[/api/upload] Error:", err);
      res.status(500).json({ error: err.message || "文件解析失败" });
    }
  });

  // Image generation endpoint (wan2.7-image / wan2.7-video)
  app.post("/api/image-generation", async (req, res) => {
    try {
      const { apiKey, model, prompt } = req.body;
      if (!apiKey || apiKey.trim() === "") {
        return res.status(400).json({ error: "API Key 不能为空！" });
      }
      if (!prompt || !prompt.trim()) {
        return res.status(400).json({ error: "提示词不能为空！" });
      }

      const selectedModel = model || "wan2.7-image";
      console.log(`[DashScope] ImageGeneration for model ${selectedModel}: "${prompt.slice(0, 50)}..."`);

      const imageModels = ["wan2.7-image", "wan2.7-image-pro", "wordart-semantic", "wordart-texture"];
      const isImageModel = imageModels.includes(selectedModel);
      const endpoint = isImageModel
        ? "https://dashscope.aliyuncs.com/api/v1/services/aigc/image-generation/generation"
        : "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation";

      const requestBody: any = {
        model: selectedModel,
        input: {
          messages: [{ role: "user", content: [{ text: prompt.trim() }] }]
        },
        parameters: { n: 1 }
      };

      // Step 1: Submit async task (X-DashScope-Async header is REQUIRED)
      const submitRes = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${apiKey.trim()}`,
          "Content-Type": "application/json",
          "X-DashScope-Async": "enable"
        },
        body: JSON.stringify(requestBody)
      });

      let submitData: any = {};
      if (submitRes.ok) {
        const text = await submitRes.text();
        if (text) {
          try { submitData = JSON.parse(text); } catch {}
        }
      } else {
        const errText = await submitRes.text();
        console.error(`[DashScope] Image submit failed (${submitRes.status}):`, errText.slice(0, 200));
        return res.status(502).json({ error: `阿里云百炼网关拒绝请求（HTTP ${submitRes.status}），请检查 API Key 是否有效。` });
      }

      const taskId = submitData.output?.task_id;

      if (!taskId) {
        const errMsg = submitData.error?.message || submitData.message || JSON.stringify(submitData);
        return res.status(400).json({ error: `提交生成任务失败：${errMsg}` });
      }

      console.log(`[DashScope] Task submitted. task_id=${taskId}, polling...`);

      // Step 2: Poll for completion (up to 3 minutes)
      const maxAttempts = 90;
      let outputUrl = "";
      let finalStatus = "";

      for (let attempt = 0; attempt < maxAttempts; attempt++) {
        await new Promise(r => setTimeout(r, 2000));

        const statusRes = await fetch(`https://dashscope.aliyuncs.com/api/v1/tasks/${taskId}`, {
          method: "GET",
          headers: {
            "Authorization": `Bearer ${apiKey.trim()}`,
            "Content-Type": "application/json"
          }
        });

        let statusData: any = {};
        if (statusRes.ok) {
          const stext = await statusRes.text();
          if (stext) {
            try { statusData = JSON.parse(stext); } catch {}
          }
        } else {
          console.error(`[DashScope] Image poll failed (${statusRes.status})`);
        }

        const state = statusData.output?.task_status || statusData.output?.state || "";

        console.log(`[DashScope] Poll attempt ${attempt + 1}: state=${state}`);

        if (state === "SUCCEEDED" || state === "succeeded" || state === "COMPLETED") {
          outputUrl = statusData.output?.choices?.[0]?.message?.content?.[0]?.image
            || statusData.output?.output?.url
            || statusData.output?.image
            || statusData.output?.url
            || "";
          finalStatus = "succeeded";
          break;
        }

        if (state === "FAILED" || state === "failed" || state === "ERROR") {
          finalStatus = "failed";
          const errMsg = statusData.output?.error?.message || statusData.error?.message || JSON.stringify(statusData);
          return res.status(500).json({ error: `生成任务失败：${errMsg}` });
        }

        if (attempt === maxAttempts - 1) {
          return res.status(504).json({ error: `生成超时（超过 3 分钟），请稍后重试。任务ID：${taskId}` });
        }
      }

      if (!outputUrl) {
        return res.status(500).json({ error: `生成完成但未返回媒体 URL，请检查模型配置。任务ID：${taskId}` });
      }

      console.log(`[DashScope] Generation succeeded. URL: ${outputUrl.slice(0, 80)}...`);
      res.json({
        url: outputUrl,
        taskId,
        model: selectedModel,
        type: isImageModel ? "image" : "video"
      });
    } catch (error: any) {
      console.error("[/api/image-generation] exception:", error);
      res.status(500).json({ error: `生成异常：${error.message || error}` });
    }
  });

  // Video generation endpoint (wan2.7-t2v / wan2.7-i2v / happyhorse video models)
  app.post("/api/video-generation", async (req, res) => {
    try {
      const { apiKey, model, prompt, duration, resolution, ratio, watermark, seed } = req.body;
      if (!apiKey || apiKey.trim() === "") {
        return res.status(400).json({ error: "API Key 不能为空！" });
      }
      if (!prompt || !prompt.trim()) {
        return res.status(400).json({ error: "提示词不能为空！" });
      }

      const selectedModel = model || "wan2.7-t2v-2026-04-25";
      console.log(`[DashScope] VideoGeneration for model ${selectedModel}: "${prompt.slice(0, 50)}..."`);

      const requestBody: any = {
        model: selectedModel,
        input: {
          prompt: prompt.trim()
        },
        parameters: {
          duration: parseInt(duration) || 5,
          resolution: resolution || "720P",
          ratio: ratio || "16:9",
          prompt_extend: true,
          watermark: !!watermark,
          seed: seed ? parseInt(seed) : undefined
        }
      };

      // Step 1: Submit async task
      const submitRes = await fetch("https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${apiKey.trim()}`,
          "Content-Type": "application/json",
          "X-DashScope-Async": "enable"
        },
        body: JSON.stringify(requestBody)
      });

      let submitData: any = {};
      if (submitRes.ok) {
        const text = await submitRes.text();
        if (text) {
          try { submitData = JSON.parse(text); } catch {}
        }
      } else {
        const errText = await submitRes.text();
        console.error(`[DashScope] Video submit failed (${submitRes.status}):`, errText.slice(0, 200));
        return res.status(502).json({ error: `阿里云百炼网关拒绝请求（HTTP ${submitRes.status}），请检查 API Key 是否有效。` });
      }

      const taskId = submitData.output?.task_id;

      if (!taskId) {
        const errMsg = submitData.error?.message || submitData.message || JSON.stringify(submitData);
        return res.status(400).json({ error: `提交视频生成任务失败：${errMsg}` });
      }

      console.log(`[DashScope] Video task submitted. task_id=${taskId}, polling...`);

      // Step 2: Poll for completion (up to 5 minutes for video)
      const maxAttempts = 150;
      let outputUrl = "";
      let finalStatus = "";

      for (let attempt = 0; attempt < maxAttempts; attempt++) {
        await new Promise(r => setTimeout(r, 2000));

        const statusRes = await fetch(`https://dashscope.aliyuncs.com/api/v1/tasks/${taskId}`, {
          method: "GET",
          headers: {
            "Authorization": `Bearer ${apiKey.trim()}`,
            "Content-Type": "application/json"
          }
        });

        let statusData: any = {};
        if (statusRes.ok) {
          const stext = await statusRes.text();
          if (stext) {
            try { statusData = JSON.parse(stext); } catch {}
          }
        } else {
          console.error(`[DashScope] Video poll failed (${statusRes.status})`);
        }

        const state = statusData.output?.task_status || statusData.output?.state || "";

        console.log(`[DashScope] Video poll ${attempt + 1}: state=${state}`);

        if (state === "SUCCEEDED" || state === "succeeded" || state === "COMPLETED") {
          // Official response: output.choices[0].message.content[0].video.video_url
          outputUrl = statusData.output?.choices?.[0]?.message?.content?.[0]?.video?.video_url
            || statusData.output?.output?.url
            || statusData.output?.video_url
            || statusData.output?.url
            || "";
          finalStatus = "succeeded";
          break;
        }

        if (state === "FAILED" || state === "failed" || state === "ERROR") {
          finalStatus = "failed";
          const errMsg = statusData.output?.error?.message || statusData.error?.message || JSON.stringify(statusData);
          return res.status(500).json({ error: `视频生成任务失败：${errMsg}` });
        }

        if (attempt === maxAttempts - 1) {
          return res.status(504).json({ error: `视频生成超时（超过 5 分钟），请稍后重试。任务ID：${taskId}` });
        }
      }

      if (!outputUrl) {
        return res.status(500).json({ error: `视频生成完成但未返回 URL，请检查模型配置。任务ID：${taskId}` });
      }

      console.log(`[DashScope] Video generation succeeded. URL: ${outputUrl.slice(0, 80)}...`);
      res.json({
        url: outputUrl,
        taskId,
        model: selectedModel,
        type: "video"
      });
    } catch (error: any) {
      console.error("[/api/video-generation] exception:", error);
      res.status(500).json({ error: `视频生成异常：${error.message || error}` });
    }
  });

  // Serve static files in production / Vite dev server in dev
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on port ${PORT}`);
  });
}

startServer();
