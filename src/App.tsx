/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "motion/react";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import {
  BookOpen,
  Award,
  Compass,
  Sparkles,
  Search,
  Check,
  Copy,
  ChevronRight,
  ChevronLeft,
  TrendingUp,
  BrainCircuit,
  Terminal,
  Calendar,
  AlertTriangle,
  Lightbulb,
  CheckCircle2,
  Send,
  MessageSquare,
  X,
  FileCode2,
  Users,
  Star,
  ExternalLink,
  Lock,
  ArrowLeft,
  ArrowRight,
  Info,
  User,
  Key,
  Eye,
  EyeOff,
  UploadCloud,
  Sliders,
  Trash2,
  Settings,
  LogIn,
  LogOut,
  ClipboardList,
  BarChart3,
  Dna,
  Bot,
  Paperclip,
  FileUp,
  FileText,
  Image as ImageIcon,
  FileCode,
  FileSpreadsheet,
  Radio,
  Database,
  Video,
  Plus
} from "lucide-react";
import Markdown from "react-markdown";
import {
  MODELS_DATA,
  COMPETITIONS_DATA,
  PATHWAYS_DATA,
  ASSESSMENT_QUESTIONS
} from "./data";
import { ModelDoc, Competition, LearningPath, Message, AssessmentQuestion, AgentSession } from "./types";
import {
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from "recharts";
// @ts-ignore
import mainLogo from "./assets/images/logo.png";

function normalizeMathBlocks(content: string) {
  const codeBlocks: string[] = [];
  const protectedContent = content.replace(/```[\s\S]*?```/g, (match) => {
    const token = `__CODE_BLOCK_${codeBlocks.length}__`;
    codeBlocks.push(match);
    return token;
  });

  const blockPlaceholders: string[] = [];
  const protectedWithBlocks = protectedContent.replace(/\$\$([\s\S]*?)\$\$/g, (match) => {
    const token = `__MATH_BLOCK_${blockPlaceholders.length}__`;
    blockPlaceholders.push(match);
    return token;
  });

  const normalized = protectedWithBlocks
    .replace(/\\\[([\s\S]*?)\\\]/g, (_match, formula) => `$$\n${formula.trim()}\n$$`)
    .replace(/\\\(([^\n]*?)\\\)/g, (_match, formula) => `$${formula.trim()}$`)
    .replace(/(^|\n)\[\s*([\s\S]*?)\s*\](?=\n|$)/g, (_match, prefix, formula) => `${prefix}$$\n${formula.trim()}\n$$`)
    .replace(/(^|\n)→/g, "$1");

  const restoredBlocks = blockPlaceholders.reduce(
    (acc, block, index) => acc.replace(`__MATH_BLOCK_${index}__`, block),
    normalized
  );

  return codeBlocks.reduce((acc, block, index) => acc.replace(`__CODE_BLOCK_${index}__`, block), restoredBlocks);
}

function renderMarkdownWithMath(content: string) {
  const normalized = normalizeMathBlocks(content);

  return (
    <Markdown
      remarkPlugins={[remarkMath]}
      rehypePlugins={[rehypeKatex]}
      components={{
        p({ children }) {
          return <p className="mb-3 last:mb-0">{children}</p>;
        },
        pre({ children }) {
          return <pre className="overflow-x-auto rounded-xl bg-slate-950 px-4 py-3 text-sm text-slate-100">{children}</pre>;
        },
        code({ inline, children, className, ...props }: any) {
          if (inline) {
            return <code className="rounded bg-slate-100 px-1.5 py-0.5 text-[0.9em] font-medium text-slate-900">{children}</code>;
          }

          return <code className={className} {...props}>{children}</code>;
        }
      }}
    >
      {normalized}
    </Markdown>
  );
}


async function parseSseStream(
  res: Response,
  onText: (text: string) => void,
  onError: (message: string) => void
): Promise<string> {
  const reader = res.body?.getReader();
  if (!reader) throw new Error("响应流不可读");

  const decoder = new TextDecoder();
  let buffer = "";
  let fullText = "";

  try {
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
            if (parsed.error !== undefined) {
              onError(String(parsed.error));
            } else if (typeof parsed.text === "string") {
              fullText = parsed.text;
              onText(fullText);
            }
          } catch {
            console.warn("[SSE] JSON parse failed on:", dataStr.slice(0, 100));
          }
        }
      }
    }
  } catch (err) {
    console.error("[SSE] Stream read error:", err);
    throw err;
  }

  if (buffer.trim()) {
    const lines = buffer.split("\n");
    for (const rawLine of lines) {
      const trimmed = rawLine.trim();
      if (!trimmed || trimmed === "data: [DONE]") continue;
      if (trimmed.startsWith("data: ")) {
        const dataStr = trimmed.slice(6);
        try {
          const parsed = JSON.parse(dataStr);
          if (parsed.error !== undefined) {
            onError(String(parsed.error));
          } else if (typeof parsed.text === "string") {
            fullText = parsed.text;
            onText(fullText);
          }
        } catch {
          console.warn("[SSE] tail JSON parse failed on:", dataStr.slice(0, 100));
        }
      }
    }
  }

  return fullText;
}

export default function App() {
  // Current Navigation State
  const [activeTab, setActiveTab] = useState<"home" | "models" | "competitions" | "pathways" | "assessment" | "agent" | "profile" | "settings" | "login" | "register">("home");
  
  // Search state across sections
  const [globalSearch, setGlobalSearch] = useState("");

  const [saveStatus, setSaveStatus] = useState<"idle" | "success" | "error">("idle");
  const [testStatus, setTestStatus] = useState<"idle" | "testing" | "success" | "error">("idle");
  const [testFeedback, setTestFeedback] = useState("");
  const [visionTestStatus, setVisionTestStatus] = useState<"idle" | "testing" | "success" | "error">("idle");
  const [visionTestFeedback, setVisionTestFeedback] = useState("");
  const [showApiKey, setShowApiKey] = useState(false);
  const [showVisionApiKey, setShowVisionApiKey] = useState(false);

  // User Authenticated State
  const [isLoggedIn, setIsLoggedIn] = useState(() => {
    const saved = localStorage.getItem("is_logged_in");
    return saved === "true";
  });
  
  const [currentAccount, setCurrentAccount] = useState(() => {
    return localStorage.getItem("current_account") || "guest";
  });

  // Header Avatar Dropdown Hover State
  const [showDropdown, setShowDropdown] = useState(false);
  const dropdownTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Login Form States
  const [loginAccount, setLoginAccount] = useState("");
  const [loginPassword, setLoginPassword] = useState("");
  const [loginError, setLoginError] = useState("");

  // Register Form States
  const [regAccount, setRegAccount] = useState("");
  const [regPassword, setRegPassword] = useState("");
  const [regConfirmPassword, setRegConfirmPassword] = useState("");
  const [regError, setRegError] = useState("");

  // Hero Carousel State
  const [heroIndex, setHeroIndex] = useState(0);
  const heroContent = [
    {
      title: "开启你的数学建模智慧之旅",
      subtitle: "为初学者量身定做的算法模型库与备赛指导平台。我们将错综复杂的高数公理转化成优雅易跑的 Python / MATLAB 模型，并内置了先进建模 AI Copilot。",
      tag: "2026年数模黄金备赛期",
      icon: <BrainCircuit className="w-4 h-4" />,
      bg: "from-[#1F6FEB] to-[#124294]",
      accent: "f(x)"
    },
    {
      title: "100+ 核心算法模型一触即达",
      subtitle: "从数据预处理到深度神经网络，从层次分析法到遗传算法。每一行代码都经过严格测试，助你快速跑通模型，告别报错与调参焦虑。",
      tag: "全能算法库",
      icon: <FileCode2 className="w-4 h-4" />,
      bg: "from-[#0969DA] to-[#033D8B]",
      accent: "Σ"
    },
    {
      title: "探索 300+ 往届数模实战案例",
      subtitle: "深度拆解国赛、美赛历年各奖项论文。涵盖建模思维、可视化绘图技巧以及学术级 LaTeX 排版指南，助你冲击特等奖。",
      tag: "实战案例库",
      icon: <Award className="w-4 h-4" />,
      bg: "from-[#2da44e] to-[#1a7f37]",
      accent: "lim"
    }
  ];

  useEffect(() => {
    const timer = setInterval(() => {
      setHeroIndex((prev) => (prev + 1) % heroContent.length);
    }, 5000);
    return () => clearInterval(timer);
  }, [heroContent.length]);

  // Helper to get scoped value
  const getScopedKey = (key: string, account?: string) => `user_${account || currentAccount}_${key}`;

  // User Profile States
  const [profileName, setProfileName] = useState(() => localStorage.getItem(getScopedKey("profile_name", currentAccount)) || "数模先锋");
  const [profileGender, setProfileGender] = useState(() => localStorage.getItem(getScopedKey("profile_gender", currentAccount)) || "男");
  const [profileAge, setProfileAge] = useState(() => localStorage.getItem(getScopedKey("profile_age", currentAccount)) || "20");
  const [profileEmail, setProfileEmail] = useState(() => localStorage.getItem(getScopedKey("profile_email", currentAccount)) || "user@example.com");
  const [profileSchool, setProfileSchool] = useState(() => localStorage.getItem(getScopedKey("profile_school", currentAccount)) || "未设置院校");
  const [profileMajor, setProfileMajor] = useState(() => localStorage.getItem(getScopedKey("profile_major", currentAccount)) || "未设置专业");
  const [profileRole, setProfileRole] = useState(() => localStorage.getItem(getScopedKey("profile_role", currentAccount)) || "建模爱好者");
  const [profileAvatar, setProfileAvatar] = useState(() => localStorage.getItem(getScopedKey("profile_avatar", currentAccount)) || "");
  const [bailianApiKey, setBailianApiKey] = useState(() => localStorage.getItem(getScopedKey("bailian_api_key", currentAccount)) || "");
  const [bailianModel, setBailianModel] = useState(() => localStorage.getItem(getScopedKey("bailian_model", currentAccount)) || "qwen-plus");
  const [bailianVisionApiKey, setBailianVisionApiKey] = useState(() => localStorage.getItem(getScopedKey("bailian_vision_api_key", currentAccount)) || "");
  const [bailianVisionModel, setBailianVisionModel] = useState(() => localStorage.getItem(getScopedKey("bailian_vision_model", currentAccount)) || "qwen3-vl-plus");

  // Image Generation model state
  const [bailianImageGenApiKey, setBailianImageGenApiKey] = useState(() => localStorage.getItem(getScopedKey("bailian_imagegen_api_key", currentAccount)) || "");
  const [bailianImageGenModel, setBailianImageGenModel] = useState(() => localStorage.getItem(getScopedKey("bailian_imagegen_model", currentAccount)) || "wan2.7-image");
  const [imageGenTestStatus, setImageGenTestStatus] = useState<"idle" | "testing" | "success" | "error">("idle");
  const [imageGenTestFeedback, setImageGenTestFeedback] = useState("");
  const [showImageGenApiKey, setShowImageGenApiKey] = useState(false);

  // Video Generation model state
  const [bailianVideoGenApiKey, setBailianVideoGenApiKey] = useState(() => localStorage.getItem(getScopedKey("bailian_videogen_api_key", currentAccount)) || "");
  const [bailianVideoGenModel, setBailianVideoGenModel] = useState(() => localStorage.getItem(getScopedKey("bailian_videogen_model", currentAccount)) || "wan2.7-t2v-2026-04-25");
  const [videoGenTestStatus, setVideoGenTestStatus] = useState<"idle" | "testing" | "success" | "error">("idle");
  const [videoGenTestFeedback, setVideoGenTestFeedback] = useState("");
  const [showVideoGenApiKey, setShowVideoGenApiKey] = useState(false);

  // Local editing states (sync'd to activeTab === "profile" via useEffect)
  const [editName, setEditName] = useState(profileName);
  const [editGender, setEditGender] = useState(profileGender);
  const [editAge, setEditAge] = useState(profileAge);
  const [editEmail, setEditEmail] = useState(profileEmail);
  const [editSchool, setEditSchool] = useState(profileSchool);
  const [editMajor, setEditMajor] = useState(profileMajor);
  const [editRole, setEditRole] = useState(profileRole);
  const [editAvatar, setEditAvatar] = useState(profileAvatar);
  const [editApiKey, setEditApiKey] = useState(bailianApiKey);
  const [editModel, setEditModel] = useState(bailianModel);
  const [editVisionApiKey, setEditVisionApiKey] = useState(bailianVisionApiKey);
  const [editVisionModel, setEditVisionModel] = useState(bailianVisionModel);
  const [editImageGenApiKey, setEditImageGenApiKey] = useState(bailianImageGenApiKey);
  const [editImageGenModel, setEditImageGenModel] = useState(bailianImageGenModel);
  const [editVideoGenApiKey, setEditVideoGenApiKey] = useState(bailianVideoGenApiKey);
  const [editVideoGenModel, setEditVideoGenModel] = useState(bailianVideoGenModel);

  // Function to load user specific data upon login
  const loadUserData = (account: string) => {
    setCurrentAccount(account);
    setProfileName(localStorage.getItem(`user_${account}_profile_name`) || account);
    setProfileGender(localStorage.getItem(`user_${account}_profile_gender`) || "男");
    setProfileAge(localStorage.getItem(`user_${account}_profile_age`) || "20");
    setProfileEmail(localStorage.getItem(`user_${account}_profile_email`) || `${account}@example.com`);
    setProfileSchool(localStorage.getItem(`user_${account}_profile_school`) || "未设置院校");
    setProfileMajor(localStorage.getItem(`user_${account}_profile_major`) || "未设置专业");
    setProfileRole(localStorage.getItem(`user_${account}_profile_role`) || "建模爱好者");
    setProfileAvatar(localStorage.getItem(`user_${account}_profile_avatar`) || "");
    setBailianApiKey(localStorage.getItem(`user_${account}_bailian_api_key`) || "");
    setBailianModel(localStorage.getItem(`user_${account}_bailian_model`) || "qwen-plus");
    setBailianVisionApiKey(localStorage.getItem(`user_${account}_bailian_vision_api_key`) || "");
    setBailianVisionModel(localStorage.getItem(`user_${account}_bailian_vision_model`) || "qwen3-vl-plus");
    setBailianImageGenApiKey(localStorage.getItem(`user_${account}_bailian_imagegen_api_key`) || "");
    setBailianImageGenModel(localStorage.getItem(`user_${account}_bailian_imagegen_model`) || "wan2.7-image");
    setBailianVideoGenApiKey(localStorage.getItem(`user_${account}_bailian_videogen_api_key`) || "");
    setBailianVideoGenModel(localStorage.getItem(`user_${account}_bailian_videogen_model`) || "wan2.7-t2v-2026-04-25");
  };

  // Function to clear user data on logout
  const clearUserData = () => {
    setIsLoggedIn(false);
    setCurrentAccount("guest");
    localStorage.setItem("is_logged_in", "false");
    localStorage.removeItem("current_account");
    
    // Reset to defaults
    setProfileName("未分配");
    setProfileEmail("guest@example.com");
    setBailianApiKey("");
    setBailianModel("qwen-plus");
    setBailianVisionApiKey("");
    setBailianVisionModel("qwen3-vl-plus");
    setBailianImageGenApiKey("");
    setBailianImageGenModel("wan2.7-image");
    setBailianVideoGenApiKey("");
    setBailianVideoGenModel("wan2.7-t2v-2026-04-25");
  };

  // Sync edits when activeTab becomes profile or settings
  useEffect(() => {
    if (activeTab === "profile" || activeTab === "settings") {
      setEditName(profileName);
      setEditGender(profileGender);
      setEditAge(profileAge);
      setEditEmail(profileEmail);
      setEditSchool(profileSchool);
      setEditMajor(profileMajor);
      setEditRole(profileRole);
      setEditAvatar(profileAvatar);
      setEditApiKey(bailianApiKey);
      setEditModel(bailianModel);
      setEditVisionApiKey(bailianVisionApiKey);
      setEditVisionModel(bailianVisionModel);
      setEditImageGenApiKey(bailianImageGenApiKey);
      setEditImageGenModel(bailianImageGenModel);
      setEditVideoGenApiKey(bailianVideoGenApiKey);
      setEditVideoGenModel(bailianVideoGenModel);
      setSaveStatus("idle");
      setTestStatus("idle");
      setTestFeedback("");
      setVisionTestStatus("idle");
      setVisionTestFeedback("");
      setImageGenTestStatus("idle");
      setImageGenTestFeedback("");
      setVideoGenTestStatus("idle");
      setVideoGenTestFeedback("");
    }
  }, [activeTab, profileName, profileGender, profileAge, profileEmail, profileSchool, profileMajor, profileRole, profileAvatar, bailianApiKey, bailianModel, bailianVisionApiKey, bailianVisionModel, bailianImageGenApiKey, bailianImageGenModel, bailianVideoGenApiKey, bailianVideoGenModel]);

  // Models module states
  const [selectedCategory, setSelectedCategory] = useState<"all" | "preprocess" | "prediction" | "evaluation" | "difference-analysis" | "correlation-analysis" | "machine-learning" | "machine-learning-regression" | "stats-analysis" | "econometrics" | "programming-solvers">("all");
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [activeCodeTab, setActiveCodeTab] = useState<"python" | "matlab">("python");
  const [copiedCodeFlag, setCopiedCodeFlag] = useState(false);

  // Assessment state
  const [assessmentStep, setAssessmentStep] = useState<"start" | "quiz" | "result">("start");
  const [currentQuestionIdx, setCurrentQuestionIdx] = useState(0);
  const [userAnswers, setUserAnswers] = useState<Record<number, number>>({});
  const [assessmentResults, setAssessmentResults] = useState<any>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [aiAnalysisPrompt, setAiAnalysisPrompt] = useState("");

  // Modeling Agent States — multi-session management
  const [agentSessions, setAgentSessions] = useState<AgentSession[]>(() => {
    try {
      const saved = localStorage.getItem(`user_${currentAccount}_agent_sessions`);
      if (saved) {
        const parsed = JSON.parse(saved) as AgentSession[];
        if (parsed.length > 0) return parsed;
      }
    } catch {}
    const defaultSession: AgentSession = {
      id: `session_${Date.now()}`,
      title: "新会话",
      messages: [{
        role: "assistant",
        content: "你好！我是你的数模 Agent 首席教练。我已经准备好接手你的建模任务了。我们将严格遵循【规划-执行-观察】循环来推进项目。\n\n**请提供赛题附件或描述任务目标，我将首先启动分析工作。**",
        timestamp: new Date().toLocaleTimeString()
      }],
      createdAt: Date.now(),
      updatedAt: Date.now()
    };
    return [defaultSession];
  });
  const [activeSessionId, setActiveSessionId] = useState<string>(() => {
    const saved = localStorage.getItem(`user_${currentAccount}_agent_active_session`);
    if (saved) return saved;
    const sessions = (() => { try { return JSON.parse(localStorage.getItem(`user_${currentAccount}_agent_sessions`) || "[]") as AgentSession[]; } catch { return []; } })();
    return sessions[0]?.id || "";
  });

  const currentSession = agentSessions.find(s => s.id === activeSessionId) || agentSessions[0];
  const agentMessages = currentSession?.messages || [];

  const persistSessions = (sessions: AgentSession[]) => {
    localStorage.setItem(`user_${currentAccount}_agent_sessions`, JSON.stringify(sessions));
  };
  const persistActiveId = (id: string) => {
    localStorage.setItem(`user_${currentAccount}_agent_active_session`, id);
  };

  const createNewAgentSession = () => {
    const newSession: AgentSession = {
      id: `session_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
      title: "新会话",
      messages: [{
        role: "assistant",
        content: "你好！我是你的数模 Agent 首席教练。我已经准备好接手你的建模任务了。我们将严格遵循【规划-执行-观察】循环来推进项目。\n\n**请提供赛题附件或描述任务目标，我将首先启动分析工作。**",
        timestamp: new Date().toLocaleTimeString()
      }],
      createdAt: Date.now(),
      updatedAt: Date.now()
    };
    const updated = [newSession, ...agentSessions];
    setAgentSessions(updated);
    setActiveSessionId(newSession.id);
    persistSessions(updated);
    persistActiveId(newSession.id);
  };

  const switchAgentSession = (id: string) => {
    setActiveSessionId(id);
    persistActiveId(id);
  };

  const deleteAgentSession = (id: string) => {
    if (agentSessions.length === 1) return; // keep at least one session
    const updated = agentSessions.filter(s => s.id !== id);
    setAgentSessions(updated);
    persistSessions(updated);
    if (activeSessionId === id) {
      const newActive = updated[0]?.id || "";
      setActiveSessionId(newActive);
      persistActiveId(newActive);
    }
  };

  const truncateTitle = (text: string) => {
    const stripped = text.replace(/<[^>]+>/g, "").replace(/\n+/g, " ").trim();
    return stripped.length > 20 ? stripped.slice(0, 20) + "…" : stripped;
  };

  const updateCurrentSession = (updater: (messages: Message[]) => Message[], targetId?: string) => {
    setAgentSessions(prev => {
      const updated = prev.map(s => {
        if (s.id !== (targetId ?? activeSessionId)) return s;
        return { ...s, messages: updater(s.messages), updatedAt: Date.now() };
      });
      persistSessions(updated);
      return updated;
    });
  };
  const [agentInput, setAgentInput] = useState("");
  const [isAgentTyping, setIsAgentTyping] = useState(false);
  const [agentAttachedFile, setAgentAttachedFile] = useState<File | null>(null);
  const [agentMode, setAgentMode] = useState<"text" | "image" | "video">("text");
  const [agentFileContent, setAgentFileContent] = useState<string>("");
  const [agentFilePreview, setAgentFilePreview] = useState<string>("");
  const agentFileRef = useRef<HTMLInputElement>(null);

  const getFileIcon = (file: File) => {
    const ext = file.name.split(".").pop()?.toLowerCase();
    if (["pdf"].includes(ext || "")) return <FileCode2 className="w-4 h-4 text-red-500" />;
    if (["doc", "docx"].includes(ext || "")) return <FileText className="w-4 h-4 text-blue-500" />;
    if (["png", "jpg", "jpeg", "gif", "webp"].includes(ext || "")) return <ImageIcon className="w-4 h-4 text-green-500" />;
    if (["csv", "xlsx", "xls"].includes(ext || "")) return <FileSpreadsheet className="w-4 h-4 text-emerald-500" />;
    return <FileCode className="w-4 h-4 text-gray-500" />;
  };

  // Agent Component States (Workbench)
  const [agentPlanning, setAgentPlanning] = useState<string>("等待初始化计划...");
  const [agentMemory, setAgentMemory] = useState<string[]>(["尚未录入任何核心建模变量"]);
  const [agentModelingStage, setAgentModelingStage] = useState<"IDLE" | "UNDERSTANDING" | "PLANNING" | "EXECUTING" | "OBSERVING">("IDLE");
  const [agentActionLog, setAgentActionLog] = useState<{tool: string, status: string, result?: string}[]>([]);

  const handleAgentFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || !e.target.files[0]) return;
    const file = e.target.files[0];

    // Generate image preview if it's an image
    const preview = file.type.startsWith("image/") ? await createPreviewUrl(file) : "";

    const formData = new FormData();
    formData.append("file", file);
    formData.append("textApiKey", bailianApiKey);
    formData.append("textModel", bailianModel);
    formData.append("visionApiKey", bailianVisionApiKey);
    formData.append("visionModel", bailianVisionModel);

    try {
      const res = await fetch("/api/upload", { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) {
        alert(`文件解析失败: ${data.error}`);
        return;
      }
      setAgentAttachedFile(file);
      setAgentFileContent(data.content || "");
      setAgentFilePreview(preview);
    } catch {
      alert("文件上传失败，请重试");
    }
  };

  const createPreviewUrl = (file: File): Promise<string> => {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target?.result as string);
      reader.readAsDataURL(file);
    });
  };

  const removeAgentFile = () => {
    setAgentAttachedFile(null);
    setAgentFileContent("");
    setAgentFilePreview("");
    if (agentFileRef.current) agentFileRef.current.value = "";
  };

  const handleAgentSendMessage = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if ((!agentInput.trim() && !agentAttachedFile) || isAgentTyping) return;

    const sentSessionId = activeSessionId;

    const fileContext = agentAttachedFile && agentFileContent
      ? `\n\n【附件文件内容】\n文件名: ${agentAttachedFile.name}\n${agentFileContent}\n【附件结束】`
      : "";
    const userMsg: Message = {
      role: "user",
      content: (agentInput || "") + fileContext,
      timestamp: new Date().toLocaleTimeString()
    };

    const assistantPlaceholder: Message = {
      role: "assistant",
      content: "",
      timestamp: new Date().toLocaleTimeString()
    };

    // 自动命名：若标题仍为默认"新会话"，用第一条用户消息内容命名
    setAgentSessions(prev => {
      const title = truncateTitle(userMsg.content);
      const updated = prev.map(s =>
        s.id === sentSessionId
          ? { ...s, title: s.title === "新会话" ? title : s.title }
          : s
      );
      persistSessions(updated);
      return updated;
    });

    // 将用户消息和 AI placeholder 加入当前会话
    updateCurrentSession(prev => [...prev, userMsg, assistantPlaceholder], sentSessionId);

    setAgentInput("");
    removeAgentFile();
    setIsAgentTyping(true);

    // === 文生图 / 文生视频模式分支 ===
    if (agentMode === "image" || agentMode === "video") {
      setAgentModelingStage("IDLE");
      const isVideo = agentMode === "video";
      const apiEndpoint = isVideo ? "/api/video-generation" : "/api/image-generation";
      const apiKey = isVideo ? bailianVideoGenApiKey : bailianImageGenApiKey;
      const model = isVideo ? bailianVideoGenModel : bailianImageGenModel;
      const keyLabel = isVideo ? "文生视频" : "文生图";

      if (!apiKey) {
        updateCurrentSession(prev => prev.map((m, idx) => idx === prev.length - 1 ? {
          ...m,
          content: `❌ 请先在设置中配置 ${keyLabel} API Key！`
        } : m), sentSessionId);
        setIsAgentTyping(false);
        return;
      }

      try {
        const res = await fetch(apiEndpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ apiKey, model, prompt: agentInput.trim() })
        });

        const data = await res.json();

        if (!res.ok) {
          updateCurrentSession(prev => prev.map((m, idx) => idx === prev.length - 1 ? {
            ...m,
          content: `❌ 出错啦: ${data.error || "未知错误"}`
          } : m), sentSessionId);
        return;
        }

        const resultMsg: Message = {
          role: "assistant",
          content: "",
          timestamp: new Date().toLocaleTimeString(),
          imageUrl: data.url,
          mediaType: isVideo ? "video" : "image"
        };

        updateCurrentSession(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = resultMsg;
          return updated;
        }, sentSessionId);
      } catch (err: any) {
        updateCurrentSession(prev => prev.map((m, idx) => idx === prev.length - 1 ? {
          ...m,
          content: `❌ 网络异常: ${err.message || err}`
        } : m), sentSessionId);
      } finally {
        setIsAgentTyping(false);
      }
      return;
    }

    setAgentModelingStage("UNDERSTANDING");

    try {
      const systemPrompt = `你是一名顶级全能数学建模 Agent。
你拥有【大脑(LLM)】、【记忆(Memory)】、【工具(Tools)】、【规划(Planning)】、【执行(Action)】、【观察(Observation)】六个核心模块。

**你的工作流程：**
1. 接收输入后，首先更新 <Planning>，展示你的逻辑拆解。
2. 调用模拟工具 <Action> (如: Python计算, 统计分析, 文献检索)。
3. 提供 <Observation>，即你从工具结果中观察到的洞察。
4. 在 <Memory> 中记录当前建模的关键参数、约束条件、选定模型。
5. 最后输出正式的引导式回复。

**输出格式规范 (必须包含以下标签)：**
<Planning> 你的思考路径和接下来的步骤 </Planning>
<Action> 你针对该问题调用的工具（如：描述性统计分析、遗传算法模块等） </Action>
<Observation> 该行动得出的初步结论或数据规律 </Observation>
<MemoryUpdate> 需要持久化记忆的新增变量、约束、或模型选择 </MemoryUpdate>

**正式内容：** 你的回复内容。保持结构化、专业，严禁直接给最终答案，除非被明确要求。

**公式输出规则：**
- 所有数学公式必须使用标准 Markdown 数学语法。
- 行内公式请使用 $...$。
- 块级公式请使用 $$...$$，且必须单独占一行。
- 不要使用 [ ... ] 包裹公式。
- 不要使用代码块来展示公式。
- 多行公式请拆成多个 $$...$$ 块，或用 aligned 环境放在 $$...$$ 中。
- 公式周围的解释文字用普通中文段落，不要把说明文字放进公式环境里。`;

      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          messages: [
            { role: "system", content: systemPrompt },
            ...agentMessages.map(m => ({ role: m.role, content: m.content })),
            { role: "user", content: userMsg.content }
          ],
          customApiKey: bailianApiKey,
          customModel: bailianModel
        })
      });

      if (!res.ok || !res.body) {
        const data = await res.json().catch(() => ({}));
        const errorText = data.error || "未知服务器异常";
        updateCurrentSession(prev => prev.map((m, idx) => idx === prev.length - 1 ? {
          ...m,
          content: `❌ 出错啦: ${errorText}`
        } : m));
        return;
      }

      const finalContent = await parseSseStream(
        res,
        (text) => {
          updateCurrentSession(prev => prev.map((m, idx) => idx === prev.length - 1 ? { ...m, content: text } : m));
        },
        (message) => {
          updateCurrentSession(prev => prev.map((m, idx) => idx === prev.length - 1 ? { ...m, content: `❌ 出错啦: ${message}` } : m));
        }
      );

      const rawContent = finalContent || "";

      const planMatch = rawContent.match(/<Planning>([\s\S]*?)<\/Planning>/);
      const actionMatch = rawContent.match(/<Action>([\s\S]*?)<\/Action>/);
      const obsMatch = rawContent.match(/<Observation>([\s\S]*?)<\/Observation>/);
      const memMatch = rawContent.match(/<MemoryUpdate>([\s\S]*?)<\/MemoryUpdate>/);

      if (planMatch) setAgentPlanning(planMatch[1].trim());
      if (actionMatch) {
        setAgentActionLog(prev => [...prev, {
          tool: actionMatch[1].trim(),
          status: "SUCCESS",
          result: obsMatch ? obsMatch[1].trim() : "已完成"
        }]);
      }
      if (memMatch) {
        const newMem = memMatch[1].trim().split("\n").filter(Boolean);
        setAgentMemory(prev => [...prev, ...newMem]);
      }

      const cleanContent = rawContent
        .replace(/<Planning>[\s\S]*?<\/Planning>/g, "")
        .replace(/<Action>[\s\S]*?<\/Action>/g, "")
        .replace(/<Observation>[\s\S]*?<\/Observation>/g, "")
        .replace(/<MemoryUpdate>[\s\S]*?<\/MemoryUpdate>/g, "")
        .trim();

      updateCurrentSession(prev => prev.map((m, idx) => idx === prev.length - 1 ? {
        ...m,
        content: cleanContent || rawContent || "抱歉，本次没有生成有效回答，请重试。"
      } : m), sentSessionId);
    } catch (err) {
      console.error(err);
    } finally {
      setIsAgentTyping(false);
      setAgentModelingStage("IDLE");
    }
  };

  // Pathways states
  const [selectedPathId, setSelectedPathId] = useState<string>("zero-base");

  const handleStartAssessment = () => {
    setAssessmentStep("quiz");
    setCurrentQuestionIdx(0);
    setUserAnswers({});
    setAssessmentResults(null);
  };

  const handleAnswer = (questionId: number, optionIdx: number) => {
    setUserAnswers(prev => ({ ...prev, [questionId]: optionIdx }));
    if (currentQuestionIdx < ASSESSMENT_QUESTIONS.length - 1) {
      setCurrentQuestionIdx(prev => prev + 1);
    } else {
      calculateResults();
    }
  };

  const calculateResults = async () => {
    setAssessmentStep("result");
    setIsAnalyzing(true);

    // Heuristic scoring mapping
    const dimensions = {
      abstraction: { score: 0, total: 0 },
      selection: { score: 0, total: 0 },
      foundation: { score: 0, total: 0 },
      algorithm: { score: 0, total: 0 },
      programming: { score: 0, total: 0 },
      expression: { score: 0, total: 0 }
    };

    ASSESSMENT_QUESTIONS.forEach(q => {
      dimensions[q.dimension].total += 1;
      if (userAnswers[q.id] === q.correctOptionIndex) {
        dimensions[q.dimension].score += 1;
      }
    });

    const radarData = Object.entries(dimensions).map(([key, val]) => {
      const labels = {
        abstraction: "问题抽象",
        selection: "模型选择",
        foundation: "数学基础",
        algorithm: "算法求解",
        programming: "编程实现",
        expression: "论文表达"
      };
      return {
        subject: labels[key as keyof typeof labels],
        A: Math.round((val.score / val.total) * 100) || 10, // Avoid 0 for better visual
        fullMark: 100
      };
    });

    setAssessmentResults(radarData);

    // AI Analysis call
    try {
      const summary = radarData.map(d => `${d.subject}: ${d.A}`).join(", ");
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          messages: [
            { 
              role: "user", 
              content: `我完成了一次数学建模能力测评，我的各项得分为：${summary}。请作为资深数模竞赛教练，为我输出一份简短的能力分析报告和针对性的学习建议（300字以内）。重点放在我的弱项上。` 
            }
          ],
          customApiKey: bailianApiKey,
          customModel: bailianModel
        })
      });
      if (!res.ok || !res.body) {
        const data = await res.json().catch(() => ({}));
        setAiAnalysisPrompt(data.error || "AI 分析引擎正在维护中，请根据雷达图自行评估。");
        setIsAnalyzing(false);
        return;
      }

      const finalText = await parseSseStream(
        res,
        (text) => setAiAnalysisPrompt(text),
        (message) => setAiAnalysisPrompt(`❌ 出错啦: ${message}`)
      );

      if (!finalText) {
        setAiAnalysisPrompt("AI 分析引擎正在维护中，请根据雷达图自行评估。");
      }
    } catch (e) {
      setAiAnalysisPrompt("无法连接到 AI 诊断引擎，请检查网络或 API 配置。");
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Competitions filter state
  const [selectedDifficulty, setSelectedDifficulty] = useState<number | null>(null);
  const [selectedMonth, setSelectedMonth] = useState<number | null>(null);

  // Copilot (AI chat assistant) states
  const [chatOpen, setChatOpen] = useState(false);
  const [chatInput, setChatInput] = useState("");
  const [chatMessages, setChatMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "你好！感谢使用 MathModel Hub 智能学习助手。💻\n你可以向我提报任何数学建模的问题，比如：\n🔸 *哪个模型适合处理人口预测？*\n🔸 *帮我解释一下 AHP 层次分析法的具体步骤？*\n🔸 *提供一段多元线性回归的 Python 求解代码*",
      timestamp: new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })
    }
  ]);
  const [chatLoading, setChatLoading] = useState(false);
  const chatBottomRef = useRef<HTMLDivElement | null>(null);
  const agentBottomRef = useRef<HTMLDivElement | null>(null);

  // Auto-scroll chat to bottom
  useEffect(() => {
    if (chatBottomRef.current) {
      chatBottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [chatMessages, chatLoading, chatOpen]);

  // Auto-scroll agent chat to bottom
  useEffect(() => {
    if (agentBottomRef.current) {
      agentBottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [agentMessages, isAgentTyping]);

  // Code Copy utility
  const readStreamResponse = async (
    res: Response,
    onText: (text: string) => void,
    onError: (message: string) => void
  ): Promise<string> => {
    const reader = res.body?.getReader();
    if (!reader) throw new Error("响应流不可读");

    const decoder = new TextDecoder();
    let buffer = "";
    let fullText = "";

    try {
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
              if (parsed.error !== undefined) {
                onError(String(parsed.error));
              } else if (typeof parsed.text === "string") {
                fullText = parsed.text;
                onText(fullText);
              }
            } catch {
              console.warn("[SSE] JSON parse failed on:", dataStr.slice(0, 100));
            }
          }
        }
      }
    } catch (err) {
      console.error("[SSE] Stream read error:", err);
      throw err;
    }

    if (buffer.trim()) {
      const lines = buffer.split("\n");
      for (const rawLine of lines) {
        const trimmed = rawLine.trim();
        if (!trimmed || trimmed === "data: [DONE]") continue;
        if (trimmed.startsWith("data: ")) {
          const dataStr = trimmed.slice(6);
          try {
            const parsed = JSON.parse(dataStr);
            if (parsed.error !== undefined) {
              onError(String(parsed.error));
            } else if (typeof parsed.text === "string") {
              fullText = parsed.text;
              onText(fullText);
            }
          } catch {
            console.warn("[SSE] tail JSON parse failed on:", dataStr.slice(0, 100));
          }
        }
      }
    }

    return fullText;
  };

  // Code copy utility
  const copyToClipboard = (text: string) => {
    if (!text) return;
    navigator.clipboard.writeText(text);
    setCopiedCodeFlag(true);
    setTimeout(() => setCopiedCodeFlag(false), 2000);
  };

  // Chat message submit
  const handleSendMessage = async (textToSend?: string) => {
    const rawText = textToSend || chatInput;
    if (!rawText.trim() || chatLoading) return;

    const userMsg: Message = {
      role: "user",
      content: rawText,
      timestamp: new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })
    };

    const assistantMsg: Message = {
      role: "assistant",
      content: "",
      timestamp: new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })
    };

    setChatMessages((prev) => [...prev, userMsg, assistantMsg]);
    if (!textToSend) {
      setChatInput("");
    }
    setChatLoading(true);

    try {
      const payloadMessages = [...chatMessages, userMsg].map(m => ({
        role: m.role,
        content: m.content
      }));

      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          messages: payloadMessages,
          customApiKey: bailianApiKey,
          customModel: bailianModel
        })
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        const errText = data.error || "未知服务器异常";
        setChatMessages((prev) => prev.map((m, idx) => idx === prev.length - 1 ? {
          ...m,
          content: `❌ 出错啦: ${errText}`
        } : m));
        return;
      }

      if (!res.body) {
        throw new Error("响应流为空");
      }

      const finalText = await readStreamResponse(
        res,
        (text) => {
          setChatMessages((prev) => prev.map((m, idx) => idx === prev.length - 1 ? { ...m, content: text } : m));
        },
        (message) => {
          setChatMessages((prev) => prev.map((m, idx) => idx === prev.length - 1 ? { ...m, content: `❌ 出错啦: ${message}` } : m));
        }
      );

      if (!finalText) {
        setChatMessages((prev) => prev.map((m, idx) => idx === prev.length - 1 ? { ...m, content: "抱歉，本次没有生成有效回答，请重试。" } : m));
      }
    } catch (err: any) {
      setChatMessages((prev) => prev.map((m, idx) => idx === prev.length - 1 ? {
        ...m,
        content: `❌ 网络连接异常: ${err.message || err}。请确认本地服务运行正常。`
      } : m));
    } finally {
      setChatLoading(false);
    }
  };

  // Drag & drop state and handlers for avatar upload
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith("image/")) {
        const reader = new FileReader();
        reader.onload = (event) => {
          if (event.target?.result) {
            setEditAvatar(event.target.result as string);
          }
        };
        reader.readAsDataURL(file);
      } else {
        alert("请上传图片格式文件！");
      }
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (file.type.startsWith("image/")) {
        const reader = new FileReader();
        reader.onload = (event) => {
          if (event.target?.result) {
            setEditAvatar(event.target.result as string);
          }
        };
        reader.readAsDataURL(file);
      } else {
        alert("请上传图片格式文件！");
      }
    }
  };

  // Test Alibaba Cloud Bailian connection
  const handleTestBailianConnection = async () => {
    if (!editApiKey.trim()) {
      setTestStatus("error");
      setTestFeedback("请输入 阿里云百炼 API Key 后再进行连接测试！");
      return;
    }
    setTestStatus("testing");
    setTestFeedback("");
    try {
      const res = await fetch("/api/test-bailian", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          apiKey: editApiKey,
          model: editModel
        })
      });
      const data = await res.json();
      if (res.ok) {
        setTestStatus("success");
        setTestFeedback(data.message || "连接测试成功！通义千问已就绪。");
      } else {
        setTestStatus("error");
        setTestFeedback(data.error || "连接测试失败，请确认 API Key 是否正确或具有可用额度。");
      }
    } catch (err: any) {
      setTestStatus("error");
      setTestFeedback(`测试出现异常: ${err.message || err}`);
    }
  };

  // Test Alibaba Cloud Bailian Vision connection
  const handleTestVisionConnection = async () => {
    if (!editVisionApiKey.trim()) {
      setVisionTestStatus("error");
      setVisionTestFeedback("请输入视觉模型的 API Key 后再进行连接测试！");
      return;
    }
    setVisionTestStatus("testing");
    setVisionTestFeedback("");
    try {
      const res = await fetch("/api/test-vision", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          apiKey: editVisionApiKey,
          model: editVisionModel
        })
      });
      const data = await res.json();
      if (res.ok) {
        setVisionTestStatus("success");
        setVisionTestFeedback(data.message || "视觉模型连通测试成功！");
      } else {
        setVisionTestStatus("error");
        setVisionTestFeedback(data.error || "视觉模型连通测试失败，请确认 API Key 和模型是否正确。");
      }
    } catch (err: any) {
      setVisionTestStatus("error");
      setVisionTestFeedback(`测试出现异常: ${err.message || err}`);
    }
  };

  // Test Alibaba Cloud Bailian Image Generation connection
  const handleTestImageGenConnection = async () => {
    if (!editImageGenApiKey.trim()) {
      setImageGenTestStatus("error");
      setImageGenTestFeedback("请输入文生图 API Key 后再进行连接测试！");
      return;
    }
    setImageGenTestStatus("testing");
    setImageGenTestFeedback("");
    try {
      const res = await fetch("/api/test-imagegen", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          apiKey: editImageGenApiKey,
          model: editImageGenModel
        })
      });
      const data = await res.json();
      if (res.ok) {
        setImageGenTestStatus("success");
        setImageGenTestFeedback(data.message || "文生图模型连通测试成功！");
      } else {
        setImageGenTestStatus("error");
        setImageGenTestFeedback(data.error || "文生图模型连通测试失败，请确认 API Key 和模型是否正确。");
      }
    } catch (err: any) {
      setImageGenTestStatus("error");
      setImageGenTestFeedback(`测试出现异常: ${err.message || err}`);
    }
  };

  const handleTestVideoGenConnection = async () => {
    if (!editVideoGenApiKey.trim()) {
      setVideoGenTestStatus("error");
      setVideoGenTestFeedback("请输入文生视频 API Key 后再进行连接测试！");
      return;
    }
    setVideoGenTestStatus("testing");
    setVideoGenTestFeedback("");
    try {
      const res = await fetch("/api/test-videogen", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          apiKey: editVideoGenApiKey,
          model: editVideoGenModel
        })
      });
      const data = await res.json();
      if (res.ok) {
        setVideoGenTestStatus("success");
        setVideoGenTestFeedback(data.message || "文生视频模型连通测试成功！");
      } else {
        setVideoGenTestStatus("error");
        setVideoGenTestFeedback(data.error || "文生视频模型连通测试失败，请确认 API Key 和模型是否正确。");
      }
    } catch (err: any) {
      setVideoGenTestStatus("error");
      setVideoGenTestFeedback(`测试出现异常: ${err.message || err}`);
    }
  };

  // Save entire user profile
  const handleSaveProfile = () => {
    try {
      // Persist to user-scoped localStorage
      localStorage.setItem(getScopedKey("profile_name"), editName);
      localStorage.setItem(getScopedKey("profile_gender"), editGender);
      localStorage.setItem(getScopedKey("profile_age"), editAge);
      localStorage.setItem(getScopedKey("profile_email"), editEmail);
      localStorage.setItem(getScopedKey("profile_school"), editSchool);
      localStorage.setItem(getScopedKey("profile_major"), editMajor);
      localStorage.setItem(getScopedKey("profile_role"), editRole);
      localStorage.setItem(getScopedKey("profile_avatar"), editAvatar);
      localStorage.setItem(getScopedKey("bailian_api_key"), editApiKey);
      localStorage.setItem(getScopedKey("bailian_model"), editModel);
      localStorage.setItem(getScopedKey("bailian_vision_api_key"), editVisionApiKey);
      localStorage.setItem(getScopedKey("bailian_vision_model"), editVisionModel);
      localStorage.setItem(getScopedKey("bailian_imagegen_api_key"), editImageGenApiKey);
      localStorage.setItem(getScopedKey("bailian_imagegen_model"), editImageGenModel);
      localStorage.setItem(getScopedKey("bailian_videogen_api_key"), editVideoGenApiKey);
      localStorage.setItem(getScopedKey("bailian_videogen_model"), editVideoGenModel);

      // Sync parent state
      setProfileName(editName);
      setProfileGender(editGender);
      setProfileAge(editAge);
      setProfileEmail(editEmail);
      setProfileSchool(editSchool);
      setProfileMajor(editMajor);
      setProfileRole(editRole);
      setProfileAvatar(editAvatar);
      setBailianApiKey(editApiKey);
      setBailianModel(editModel);
      setBailianVisionApiKey(editVisionApiKey);
      setBailianVisionModel(editVisionModel);
      setBailianImageGenApiKey(editImageGenApiKey);
      setBailianImageGenModel(editImageGenModel);
      setBailianVideoGenApiKey(editVideoGenApiKey);
      setBailianVideoGenModel(editVideoGenModel);

      setSaveStatus("success");
      setTimeout(() => setSaveStatus("idle"), 3000);
    } catch (err) {
      setSaveStatus("error");
    }
  };

  // Submit Handler for user authentication login form
  const handleLoginSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    // Check against registered users in localStorage
    const savedUsersStr = localStorage.getItem("registered_users");
    const savedUsers = savedUsersStr ? JSON.parse(savedUsersStr) : [];
    const matchedUser = savedUsers.find((u: any) => u.account === loginAccount && u.password === loginPassword);

    if ((loginAccount.trim() === "123456" && loginPassword === "123456") || matchedUser) {
      const account = matchedUser ? matchedUser.account : "123456";
      setIsLoggedIn(true);
      localStorage.setItem("is_logged_in", "true");
      localStorage.setItem("current_account", account);
      
      // Load user-scoped data
      loadUserData(account);

      setLoginError("");
      setActiveTab("profile");
      setLoginAccount("");
      setLoginPassword("");
    } else {
      setLoginError("⚠️ 账号或密码错误！若未注册，请先点击下方立即注册。");
    }
  };

  // Submit Handler for registration
  const handleRegisterSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (regPassword !== regConfirmPassword) {
      setRegError("⚠️ 两次输入的密码不一致！");
      return;
    }
    if (regPassword.length < 6) {
      setRegError("⚠️ 密码长度不能少于6位！");
      return;
    }

    // Save to localStorage list of users
    const savedUsersStr = localStorage.getItem("registered_users");
    const savedUsers = savedUsersStr ? JSON.parse(savedUsersStr) : [];
    
    // Check if user already exists
    if (savedUsers.find((u: any) => u.account === regAccount) || regAccount === "123456") {
      setRegError("⚠️ 该账号已存在，请直接登录。");
      return;
    }

    const newUser = { account: regAccount, password: regPassword };
    savedUsers.push(newUser);
    localStorage.setItem("registered_users", JSON.stringify(savedUsers));

    // Auto login after registration
    setIsLoggedIn(true);
    localStorage.setItem("is_logged_in", "true");
    localStorage.setItem("current_account", regAccount);
    
    // Load scoped data (new account defaults)
    loadUserData(regAccount);
    
    setRegError("");
    setActiveTab("profile");
    setRegAccount("");
    setRegPassword("");
    setRegConfirmPassword("");
  };

  // Fast preset-clicks triggers
  const handlePresetQuestion = (q: string) => {
    setChatOpen(true);
    handleSendMessage(q);
  };

  // Filtering list logic for models
  const filteredModels = MODELS_DATA.filter((m) => {
    const matchCategory = selectedCategory === "all" || m.category === selectedCategory;
    const matchSearch =
      globalSearch === "" ||
      m.name.toLowerCase().includes(globalSearch.toLowerCase()) ||
      m.summary.toLowerCase().includes(globalSearch.toLowerCase()) ||
      m.categoryName.toLowerCase().includes(globalSearch.toLowerCase());
    return matchCategory && matchSearch;
  });

  // Filtering list logic for competitions
  const filteredCompetitions = COMPETITIONS_DATA.filter((c) => {
    const matchDifficulty = selectedDifficulty === null || c.difficulty === selectedDifficulty;
    const matchMonth = selectedMonth === null || c.month === selectedMonth;
    const matchSearch =
      globalSearch === "" ||
      c.name.toLowerCase().includes(globalSearch.toLowerCase()) ||
      c.alias.toLowerCase().includes(globalSearch.toLowerCase()) ||
      c.description.toLowerCase().includes(globalSearch.toLowerCase());
    return matchDifficulty && matchMonth && matchSearch;
  });

  // Calculate current completion status dynamically for fun visual progression
  const currentPath = PATHWAYS_DATA.find((p) => p.id === selectedPathId) || PATHWAYS_DATA[0];

  return (
    <div id="app-root" className="flex h-screen bg-[#F6F8FA] text-[#24292F] font-sans antialiased overflow-hidden">
      
      {/* Sidebar Navigation */}
      <aside id="app-sidebar" className="w-72 bg-white border-r border-[#D0D7DE] flex flex-col shrink-0 shadow-sm">
        <div className="p-6 flex flex-col h-full">
          {/* Logo Brand banner */}
          <div className="flex items-center gap-4.5 mb-8 select-none">
            <div id="mathmodel-hub-logo" className="w-[84px] h-[84px] shrink-0 bg-[#F6F8FA] rounded-xl shadow-md border border-[#D0D7DE] overflow-hidden flex items-center justify-center p-1.5 hover:scale-105 hover:shadow-lg transition-all duration-300">
              <img
                src={mainLogo}
                alt="MathHub"
                className="w-full h-full object-contain select-none"
                referrerPolicy="no-referrer"
              />
            </div>
            <div>
              <h1 className="text-[31.5px] font-bold tracking-tight text-[#1F2328] font-serif leading-tight" style={{ fontFamily: "'Times New Roman', Times, serif", letterSpacing: "0.2px" }}>MathHub</h1>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="space-y-1.5 flex-1">
            <button
              onClick={() => { setActiveTab("home"); setGlobalSearch(""); }}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg border-l-4 transition-all duration-200 font-medium ${
                activeTab === "home"
                  ? "bg-[#0969DA]/10 text-[#0969DA] border-[#0969DA] shadow-sm"
                  : "text-[#57606A] hover:text-[#1F2328] hover:bg-[#F6F8FA] border-transparent"
              }`}
            >
              <Compass className="w-5 h-5 shrink-0" />
              <span>社区首页</span>
            </button>

            <button
              onClick={() => { setActiveTab("models"); setGlobalSearch(""); }}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg border-l-4 transition-all duration-200 font-medium ${
                activeTab === "models"
                  ? "bg-[#0969DA]/10 text-[#0969DA] border-[#0969DA] shadow-sm"
                  : "text-[#57606A] hover:text-[#1F2328] hover:bg-[#F6F8FA] border-transparent"
              }`}
            >
              <BookOpen className="w-5 h-5 shrink-0" />
              <span>数学模型库</span>
              <span className="ml-auto text-[10px] bg-[#EFF2F5] border border-[#D0D7DE] text-[#57606A] px-1.5 py-0.5 rounded-md font-mono">
                {MODELS_DATA.length}
              </span>
            </button>

            <button
              onClick={() => { setActiveTab("competitions"); setGlobalSearch(""); }}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg border-l-4 transition-all duration-200 font-medium ${
                activeTab === "competitions"
                  ? "bg-[#0969DA]/10 text-[#0969DA] border-[#0969DA] shadow-sm"
                  : "text-[#57606A] hover:text-[#1F2328] hover:bg-[#F6F8FA] border-transparent"
              }`}
            >
              <Award className="w-5 h-5 shrink-0" />
              <span>数模竞赛汇总</span>
              <span className="ml-auto text-[10px] bg-[#EFF2F5] border border-[#D0D7DE] text-[#57606A] px-1.5 py-0.5 rounded-md font-mono">
                {COMPETITIONS_DATA.length}
              </span>
            </button>

            <button
              onClick={() => { setActiveTab("pathways"); setGlobalSearch(""); }}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg border-l-4 transition-all duration-200 font-medium ${
                activeTab === "pathways"
                  ? "bg-[#0969DA]/10 text-[#0969DA] border-[#0969DA] shadow-sm"
                  : "text-[#57606A] hover:text-[#1F2328] hover:bg-[#F6F8FA] border-transparent"
              }`}
            >
              <TrendingUp className="w-5 h-5 shrink-0" />
              <span>新手推荐路径</span>
            </button>

            <button
              onClick={() => { setActiveTab("agent"); setGlobalSearch(""); }}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg border-l-4 transition-all duration-200 font-medium ${
                activeTab === "agent"
                  ? "bg-[#0969DA]/10 text-[#0969DA] border-[#0969DA] shadow-sm"
                  : "text-[#57606A] hover:text-[#1F2328] hover:bg-[#F6F8FA] border-transparent"
              }`}
            >
              <Bot className="w-5 h-5 shrink-0" />
              <span>数模 AI 教练</span>
              <span className="ml-auto text-[10px] bg-blue-50 border border-blue-200 text-blue-600 px-1.5 py-0.5 rounded font-bold">
                Agent
              </span>
            </button>

            <button
              onClick={() => { setActiveTab("assessment"); setGlobalSearch(""); }}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg border-l-4 transition-all duration-200 font-medium ${
                activeTab === "assessment"
                  ? "bg-[#0969DA]/10 text-[#0969DA] border-[#0969DA] shadow-sm"
                  : "text-[#57606A] hover:text-[#1F2328] hover:bg-[#F6F8FA] border-transparent"
              }`}
            >
              <ClipboardList className="w-5 h-5 shrink-0" />
              <span>数模能力测评</span>
              <span className="ml-auto text-[10px] bg-red-50 border border-red-200 text-red-600 px-1.5 py-0.5 rounded font-bold animate-pulse">
                Hot
              </span>
            </button>

            <button
              onClick={() => { setActiveTab(isLoggedIn ? "profile" : "login"); setGlobalSearch(""); }}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg border-l-4 transition-all duration-200 font-medium ${
                activeTab === "profile"
                  ? "bg-[#0969DA]/10 text-[#0969DA] border-[#0969DA] shadow-sm"
                  : "text-[#57606A] hover:text-[#1F2328] hover:bg-[#F6F8FA] border-transparent"
              }`}
            >
              <User className="w-5 h-5 shrink-0" />
              <span>个人主页</span>
            </button>

            <button
              onClick={() => { setActiveTab(isLoggedIn ? "settings" : "login"); setGlobalSearch(""); }}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg border-l-4 transition-all duration-200 font-medium ${
                activeTab === "settings"
                  ? "bg-[#0969DA]/10 text-[#0969DA] border-[#0969DA] shadow-sm"
                  : "text-[#57606A] hover:text-[#1F2328] hover:bg-[#F6F8FA] border-transparent"
              }`}
            >
              <Settings className="w-5 h-5 shrink-0" />
              <span>配置中心</span>
              {isLoggedIn && bailianApiKey && (
                <span className="ml-auto text-[10px] bg-emerald-50 border border-emerald-250 text-emerald-700 px-1.5 py-0.5 rounded font-mono font-bold animate-pulse">
                  Qwen
                </span>
              )}
            </button>
          </nav>
        </div>
      </aside>

      {/* Main Content Pane */}
      <main className="flex-1 flex flex-col min-w-0 bg-[#F6F8FA] h-full overflow-hidden">
        {/* Top Header Panel */}
        <header className="h-16 border-b border-[#D0D7DE] bg-white px-8 flex items-center justify-between shrink-0 shadow-sm z-10">
          <div className="flex-1 max-w-lg">
            <div className="relative">
              <input
                type="text"
                value={globalSearch}
                onChange={(e) => setGlobalSearch(e.target.value)}
                placeholder="搜索数学模型、应用算法或数模比赛..."
                className="w-full bg-[#F6F8FA] border border-[#D0D7DE] rounded-full pl-10 pr-4 py-1.5 text-sm text-[#24292F] placeholder-[#57606A] focus:outline-none focus:bg-white focus:border-[#0969DA] focus:ring-1 focus:ring-[#0969DA]"
              />
              <Search className="w-4 h-4 text-[#57606A] absolute left-3.5 top-2.5" />
              {globalSearch && (
                <button
                  onClick={() => setGlobalSearch("")}
                  className="absolute right-3 top-2.5 text-xs text-[#57606A] hover:text-[#24292F] font-semibold cursor-pointer"
                >
                  清除
                </button>
              )}
            </div>
          </div>

          {/* Quick Stats bar inside header */}
          <div className="flex items-center gap-6">
            <button
              onClick={() => setChatOpen(!chatOpen)}
              className="flex items-center gap-2 px-3 py-1 bg-gradient-to-r from-blue-5/80 to-teal-5/80 hover:from-blue-100 hover:to-teal-100 text-[#0969DA] border border-[#0969DA]/20 text-xs rounded-full cursor-pointer transition-all font-semibold"
            >
              <Sparkles className="w-3.5 h-3.5 animate-pulse text-[#0969DA]" />
              <span>AI 提问助手</span>
            </button>
            <div className="h-4 w-px bg-[#D0D7DE]"></div>
            <div 
              className="relative"
              onMouseEnter={() => {
                if (dropdownTimerRef.current) clearTimeout(dropdownTimerRef.current);
                setShowDropdown(true);
              }}
              onMouseLeave={() => {
                dropdownTimerRef.current = setTimeout(() => {
                  setShowDropdown(false);
                }, 150);
              }}
            >
              <button
                id="header-profile-button"
                onClick={() => { 
                  if (isLoggedIn) {
                    setActiveTab("profile"); 
                    setGlobalSearch(""); 
                  } else {
                    setActiveTab("login");
                    setGlobalSearch("");
                  }
                }}
                className="flex items-center gap-3 hover:opacity-85 transition-all text-left cursor-pointer"
                title={isLoggedIn ? "查看修改个人资料与百炼 API Key 算力中心" : "未登录，请点击登录"}
              >
                <div className="text-right">
                  <p className="text-xs text-[#24292F] font-semibold font-mono">
                    {isLoggedIn ? profileEmail : "游客 (未登录)"}
                  </p>
                  <p className="text-[10px] text-emerald-600 flex items-center justify-end gap-1 font-semibold">
                    <span className={`w-1.5 h-1.5 rounded-full ${isLoggedIn ? 'bg-emerald-500 animate-pulse' : 'bg-gray-400'}`}></span>
                    {isLoggedIn ? (bailianApiKey ? "通义千问已就绪" : "已登录") : "待登录"}
                  </p>
                </div>
                {isLoggedIn && profileAvatar ? (
                  <img
                    src={profileAvatar}
                    alt={profileName}
                    className="w-8 h-8 rounded-full object-cover border border-[#D0D7DE] shadow-xs"
                    referrerPolicy="no-referrer"
                  />
                ) : (
                  <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-[#0969DA] to-[#388BFD] flex items-center justify-center font-bold text-xs text-white shadow-xs">
                    {isLoggedIn ? profileName.substring(0, 2).toUpperCase() : "?"}
                  </div>
                )}
              </button>

              {/* Floating Dropdown Frame */}
              {showDropdown && (
                <div className="absolute right-0 w-44 pt-2 z-50 animate-fade-in">
                  {/* Visual Pointer / Triangle Arrow */}
                  <div className="absolute top-[3px] right-4 w-3.5 h-3.5 bg-white border-t border-l border-[#D0D7DE] rotate-45 z-[-1] shadow-[-2px_-2px_5px_rgba(0,0,0,0.02)]"></div>
                  
                  <div className="bg-white border border-[#D0D7DE] rounded-xl shadow-xl py-2 overflow-hidden ring-1 ring-black/5">
                    {isLoggedIn ? (
                      <>
                        <div className="px-4 py-1.5 border-b border-[#F6F8FA] mb-1">
                          <p className="text-xs font-bold text-gray-900 truncate">{profileName}</p>
                        </div>
                        <button
                          onClick={() => {
                            setActiveTab("profile");
                            setGlobalSearch("");
                            setShowDropdown(false);
                          }}
                          className="w-full text-left px-4 py-2 text-xs text-[#24292F] hover:bg-[#F6F8FA] hover:text-[#0969DA] font-semibold flex items-center gap-2 cursor-pointer transition-all"
                        >
                          <User className="w-4 h-4 text-gray-500" />
                          <span>个人主页</span>
                        </button>
                        <div className="h-px bg-[#EFF2F5] my-1"></div>
                        <button
                          onClick={clearUserData}
                          className="w-full text-left px-4 py-2 text-xs text-rose-600 hover:bg-rose-50 font-bold flex items-center gap-2 cursor-pointer transition-all"
                        >
                          <LogOut className="w-4 h-4 text-rose-600" />
                          <span>退出登录</span>
                        </button>
                      </>
                    ) : (
                      <>
                        <button
                          onClick={() => {
                            setActiveTab("login");
                            setGlobalSearch("");
                            setShowDropdown(false);
                          }}
                          className="w-full text-left px-4 py-2.5 text-xs text-[#0969DA] hover:bg-blue-50 font-bold flex items-center gap-2 cursor-pointer transition-all"
                        >
                          <LogIn className="w-4 h-4 text-[#0969DA]" />
                          <span>立即登录</span>
                        </button>
                      </>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </header>

        {/* Dynamic Display Screens wrapper */}
        <div id="main-content-scroll-container" className="flex-1 p-8 overflow-y-auto">
          
          {/* SEARCH HIGHLIGHT IF SEARCH IS ACTIVE */}
          {globalSearch && (
            <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg flex items-center justify-between">
              <div>
                <p className="text-xs text-blue-600 font-semibold">正在应用全局搜索筛选：</p>
                <p className="text-lg font-bold text-gray-900 font-mono">“{globalSearch}”</p>
              </div>
              <button
                onClick={() => setGlobalSearch("")}
                className="px-3 py-1 bg-white border border-[#D0D7DE] text-xs text-[#57606A] hover:text-[#24292F] hover:bg-[#F6F8FA] rounded font-semibold cursor-pointer shadow-sm"
              >
                关闭过滤
              </button>
            </div>
          )}

          {/* SCREEN: MODELING AGENT PANEL (UPGRADED TO WORKBENCH) */}
          {activeTab === "agent" && (
            <div id="agent-panel" className="animate-fade-in h-[calc(100vh-140px)] flex gap-4 overflow-hidden">
              {/* Left: Session History Sidebar */}
              <div className="w-56 flex-shrink-0 flex flex-col gap-2">
                  <button
                    onClick={createNewAgentSession}
                    className="w-full flex items-center gap-2 px-3 py-2.5 bg-[#F6F8FA] hover:bg-blue-50 border border-[#D0D7DE] hover:border-blue-300 rounded-xl text-xs font-bold text-[#57606A] hover:text-[#0969DA] transition-all group"
                  >
                    <Plus className="w-4 h-4 text-blue-400 group-hover:text-[#0969DA] shrink-0" />
                    发起新会话
                  </button>
                  <div className="flex-1 overflow-y-auto space-y-1">
                    {agentSessions.map(s => (
                      <div
                        key={s.id}
                        className={`group flex items-center gap-1.5 px-3 py-2 rounded-xl cursor-pointer transition-all text-xs ${
                          s.id === activeSessionId
                            ? "bg-blue-50 border border-blue-200 text-blue-900 font-bold"
                            : "text-[#57606A] hover:bg-gray-50 border border-transparent font-medium"
                        }`}
                      >
                        <MessageSquare className={`w-3.5 h-3.5 shrink-0 ${s.id === activeSessionId ? "text-blue-500" : "text-gray-400"}`} />
                        <span
                          className="flex-1 truncate"
                          onClick={() => switchAgentSession(s.id)}
                        >
                          {s.title}
                        </span>
                        <button
                          onClick={(e) => { e.stopPropagation(); deleteAgentSession(s.id); }}
                          className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-50 rounded-lg text-red-400 hover:text-red-600 transition-all shrink-0"
                          title="删除会话"
                        >
                          <Trash2 className="w-3 h-3" />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Chat Area — full width */}
                <div className="flex-1 bg-white border border-[#D0D7DE] rounded-2xl shadow-sm overflow-hidden flex flex-col relative">
                    <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-[#F6F8FA]/20">
                    {agentMessages.map((msg, idx) => (
                      <div key={idx} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"} items-start gap-4`}>
                        {msg.role === "assistant" && (
                          <div className="w-9 h-9 rounded-xl bg-[#0969DA] flex items-center justify-center text-white shrink-0 mt-1 shadow-md">
                            <Bot className="w-5 h-5" />
                          </div>
                        )}
                        <div className={`max-w-[85%] space-y-2 ${msg.role === "user" ? "items-end" : "items-start"}`}>
                          <div className={`p-5 rounded-2xl text-[14px] leading-relaxed shadow-sm ${
                            msg.role === "user" 
                              ? "bg-[#0969DA] text-white rounded-tr-none" 
                              : "bg-white border border-[#D0D7DE] text-gray-800 rounded-tl-none"
                          }`}>
                            {msg.role === "assistant" ? (
                              <div>
                                {msg.content && (
                                  <div className="prose prose-sm max-w-none prose-blue markdown-body mb-2">
                                    {renderMarkdownWithMath(msg.content)}
                                  </div>
                                )}
                                {msg.imageUrl && (
                                  <div className="mt-2 rounded-xl overflow-hidden border border-[#D0D7DE]">
                                    {msg.mediaType === "video" ? (
                                      <video
                                        src={msg.imageUrl}
                                        controls
                                        className="max-w-full"
                                        style={{ maxHeight: "360px" }}
                                      />
                                    ) : (
                                      <img
                                        src={msg.imageUrl}
                                        alt="AI 生成内容"
                                        className="max-w-full rounded-xl cursor-pointer hover:opacity-90 transition-opacity"
                                        style={{ maxHeight: "360px" }}
                                        onClick={() => window.open(msg.imageUrl, "_blank")}
                                      />
                                    )}
                                  </div>
                                )}
                                {msg.imageUrl && (
                                  <a
                                    href={msg.imageUrl}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="inline-flex items-center gap-1 mt-2 text-[11px] text-[#0969DA] hover:underline font-semibold"
                                  >
                                    <ExternalLink className="w-3 h-3" />
                                    {msg.mediaType === "video" ? "在新窗口打开视频" : "在新窗口查看大图"}
                                  </a>
                                )}
                              </div>
                            ) : (
                              msg.content
                            )}
                          </div>
                          <div className={`text-[10px] text-[#8C959F] px-1 font-mono ${msg.role === "user" ? "text-right" : "text-left"}`}>
                            {msg.timestamp}
                          </div>
                        </div>
                        {msg.role === "user" && (
                          profileAvatar ? (
                            <img
                              src={profileAvatar}
                              alt={profileName}
                              className="w-9 h-9 rounded-xl object-cover shrink-0 mt-1 shadow-md"
                              referrerPolicy="no-referrer"
                            />
                          ) : (
                            <div className="w-9 h-9 rounded-xl bg-gray-900 flex items-center justify-center text-white shrink-0 mt-1 shadow-md">
                              <User className="w-5 h-5" />
                            </div>
                          )
                        )}
                      </div>
                    ))}
                    {isAgentTyping && (
                      <div className="flex justify-start items-center gap-4 animate-pulse">
                        <div className="w-9 h-9 rounded-xl bg-blue-100 flex items-center justify-center text-blue-600 shrink-0">
                          <Bot className="w-5 h-5" />
                        </div>
                        <div className="bg-white border border-[#D0D7DE] px-5 py-3 rounded-2xl text-xs font-bold text-[#57606A] italic shadow-sm">
                          {agentModelingStage === "PLANNING" ? "正在规划建模路径..." : "正在分析模型输出..."}
                        </div>
                      </div>
                    )}
                    <div ref={agentBottomRef} />
                  </div>

                  {/* Enhanced Input Area */}
                  <div className="p-6 bg-white border-t border-[#D0D7DE]">
                    {agentAttachedFile && (
                      <div className="mb-4 p-3 bg-blue-50 border border-blue-100 rounded-xl animate-in slide-in-from-bottom-2">
                        <div className="flex items-center gap-2 mb-1">
                          {getFileIcon(agentAttachedFile)}
                          <p className="text-xs font-bold text-blue-900 truncate tracking-tight flex-1">{agentAttachedFile.name}</p>
                          <span className="text-[10px] text-blue-400">{(agentAttachedFile.size / 1024).toFixed(1)} KB</span>
                          <button
                            onClick={removeAgentFile}
                            className="p-1 hover:bg-blue-100 rounded-lg text-blue-400 hover:text-blue-600 transition-colors"
                          >
                            <X className="w-4 h-4" />
                          </button>
                        </div>
                        {agentFilePreview && (
                          <img
                            src={agentFilePreview}
                            alt="Preview"
                            className="mt-2 max-h-40 rounded-lg border border-blue-200 object-contain w-auto"
                          />
                        )}
                      </div>
                    )}
                    <form onSubmit={handleAgentSendMessage} className="relative flex items-center gap-4">
                      <input
                        type="file"
                        ref={agentFileRef}
                        onChange={handleAgentFileChange}
                        className="hidden"
                        accept=".pdf,.doc,.docx,.png,.jpg,.jpeg,.xlsx,.csv"
                      />
                      <button
                        type="button"
                        onClick={() => agentFileRef.current?.click()}
                        className={`p-4 border rounded-xl transition-all cursor-pointer group ${
                          agentMode === "image"
                            ? "bg-gray-50 border-gray-200 text-gray-300 cursor-not-allowed"
                            : "bg-[#F6F8FA] border-[#D0D7DE] text-gray-500 hover:text-blue-600 hover:border-blue-300"
                        }`}
                        title={agentMode === "image" ? "文生图模式不支持上传附件" : "上传数据或题目附件"}
                        disabled={agentMode === "image"}
                      >
                        <Paperclip className="w-5 h-5 group-hover:scale-110 transition-transform" />
                      </button>
                      <div className="flex-1 relative">
                        <input
                          type="text"
                          className="w-full bg-[#F6F8FA] border border-[#D0D7DE] rounded-xl px-4 py-4 pr-12 text-sm focus:outline-none focus:bg-white focus:border-[#0969DA] focus:ring-4 focus:ring-blue-50 transition-all placeholder:text-gray-400"
                          placeholder={agentMode === "image" ? "描述你想生成的图片内容..." : (agentAttachedFile ? "为该文件添加说明..." : "描述建模问题或输入你的困惑...")}
                          value={agentInput}
                          onChange={(e) => setAgentInput(e.target.value)}
                          disabled={isAgentTyping}
                        />
                        <div className="absolute right-4 top-1/2 -translate-y-1/2 flex items-center gap-2">
                          <span className="text-[10px] font-mono text-gray-400 bg-gray-100 px-1.5 py-0.5 rounded border border-gray-200">Enter</span>
                        </div>
                      </div>
                      <button
                        type="submit"
                        disabled={isAgentTyping || (!agentInput.trim() && !agentAttachedFile)}
                        className={`p-4 text-white rounded-xl transition-all shadow-xl disabled:shadow-none cursor-pointer transform active:scale-95 ${
                          agentMode === "image"
                            ? "bg-green-600 hover:bg-green-700 disabled:bg-green-300 disabled:cursor-not-allowed shadow-green-100"
                            : "bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed shadow-blue-100"
                        }`}
                      >
                        <Send className="w-5 h-5" />
                      </button>
                    </form>
                    {/* 文生图/文生视频模式切换 */}
                    <div className="mt-3 flex items-center gap-2">
                      <span className="text-[10px] text-gray-400 font-medium">模式：</span>
                      <div className="flex rounded-lg border border-gray-200 overflow-hidden text-[11px] font-bold">
                        <button
                          onClick={() => setAgentMode("text")}
                          className={`px-3 py-1.5 transition-all flex items-center gap-1 ${
                            agentMode === "text"
                              ? "bg-blue-600 text-white"
                              : "bg-gray-50 text-gray-500 hover:bg-gray-100"
                          }`}
                        >
                          <MessageSquare className="w-3.5 h-3.5" />
                          文本
                        </button>
                        <button
                          onClick={() => {
                            if (agentMode !== "image") {
                              if (!bailianImageGenApiKey) {
                                alert("切换到文生图模式需要先配置文生图 API Key，请在设置中配置。");
                                return;
                              }
                              setAgentMode("image");
                            }
                          }}
                          className={`px-3 py-1.5 transition-all flex items-center gap-1 ${
                            agentMode === "image"
                              ? "bg-green-600 text-white"
                              : "bg-gray-50 text-gray-500 hover:bg-gray-100"
                          }`}
                        >
                          <ImageIcon className="w-3.5 h-3.5" />
                          文生图
                        </button>
                        <button
                          onClick={() => {
                            if (agentMode !== "video") {
                              if (!bailianVideoGenApiKey) {
                                alert("切换到文生视频模式需要先配置文生视频 API Key，请在设置中配置。");
                                return;
                              }
                              setAgentMode("video");
                            }
                          }}
                          className={`px-3 py-1.5 transition-all flex items-center gap-1 ${
                            agentMode === "video"
                              ? "bg-orange-600 text-white"
                              : "bg-gray-50 text-gray-500 hover:bg-gray-100"
                          }`}
                        >
                          <Video className="w-3.5 h-3.5" />
                          文生视频
                        </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* SCREEN 5: ASSESSMENT PANEL */}
          {activeTab === "assessment" && (
            <div className="animate-fade-in max-w-4xl mx-auto pb-12">
              {assessmentStep === "start" && (
                <div className="bg-white rounded-2xl border border-[#D0D7DE] p-12 text-center space-y-8 shadow-sm">
                  <div className="w-20 h-20 bg-blue-50 rounded-2xl flex items-center justify-center mx-auto mb-6">
                    <ClipboardList className="w-10 h-10 text-blue-600" />
                  </div>
                  <div className="space-y-4">
                    <h2 className="text-3xl font-extrabold text-gray-900">数学建模能力深度测评</h2>
                    <p className="text-[#57606A] text-lg max-w-lg mx-auto leading-relaxed">
                      基于历年竞赛核心考点，从<b>六维模型</b>全方位解读你的建模素质。AI 助教将为你生成个性化成长报告。
                    </p>
                  </div>
                  
                  <div className="grid grid-cols-3 gap-4 text-left">
                    <div className="p-4 bg-[#F6F8FA] rounded-xl border border-[#D0D7DE]">
                      <h4 className="font-bold text-gray-900 text-sm mb-1 uppercase tracking-tight">12 道核心题</h4>
                      <p className="text-[11px] text-[#57606A]">涵盖模型选择、算法实现与论文表达等关键环节</p>
                    </div>
                    <div className="p-4 bg-[#F6F8FA] rounded-xl border border-[#D0D7DE]">
                      <h4 className="font-bold text-gray-900 text-sm mb-1 uppercase tracking-tight">六维雷达图</h4>
                      <p className="text-[11px] text-[#57606A]">直观展现你的建模优势与能力短板</p>
                    </div>
                    <div className="p-4 bg-[#F6F8FA] rounded-xl border border-[#D0D7DE]">
                      <h4 className="font-bold text-gray-900 text-sm mb-1 uppercase tracking-tight">AI 诊断报告</h4>
                      <p className="text-[11px] text-[#57606A]">基于大模型的个性化备赛策略与学习建议</p>
                    </div>
                  </div>

                  <button
                    onClick={handleStartAssessment}
                    className="px-10 py-4 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-xl shadow-lg shadow-blue-200 transition-all cursor-pointer transform hover:scale-105 active:scale-95"
                  >
                    立即开启你的能力测评
                  </button>
                  <p className="text-[10px] text-[#8C959F] italic">预计耗时：5-8 分钟 • 测试结果将计入你的数模成长档案</p>
                </div>
              )}

              {assessmentStep === "quiz" && (
                <div className="space-y-6">
                  {/* Progress Header */}
                  <div className="flex justify-between items-center px-2">
                    <div>
                      <span className="text-xs font-bold text-blue-600 uppercase tracking-widest">Question {currentQuestionIdx + 1} / {ASSESSMENT_QUESTIONS.length}</span>
                      <h3 className="text-sm font-semibold text-gray-500">正在评估：{
                        ASSESSMENT_QUESTIONS[currentQuestionIdx].dimension === "abstraction" ? "问题抽象能力" :
                        ASSESSMENT_QUESTIONS[currentQuestionIdx].dimension === "selection" ? "模型选择能力" :
                        ASSESSMENT_QUESTIONS[currentQuestionIdx].dimension === "foundation" ? "数学基础能力" :
                        ASSESSMENT_QUESTIONS[currentQuestionIdx].dimension === "algorithm" ? "算法与求解能力" :
                        ASSESSMENT_QUESTIONS[currentQuestionIdx].dimension === "programming" ? "编程与实现能力" : "结果分析与表达能力"
                      }</h3>
                    </div>
                    <div className="text-right">
                      <span className="text-xs font-mono font-bold text-gray-400">Progression</span>
                      <div className="w-32 h-1.5 bg-gray-200 rounded-full mt-1 overflow-hidden">
                        <div 
                          className="h-full bg-blue-500 transition-all duration-300" 
                          style={{ width: `${((currentQuestionIdx + 1) / ASSESSMENT_QUESTIONS.length) * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>

                  {/* Question Card */}
                  <div className="bg-white rounded-2xl border border-[#D0D7DE] p-10 shadow-sm space-y-8">
                    <h2 className="text-2xl font-bold text-gray-900 leading-snug">
                      {ASSESSMENT_QUESTIONS[currentQuestionIdx].question}
                    </h2>
                    
                    <div className="space-y-3">
                      {ASSESSMENT_QUESTIONS[currentQuestionIdx].options.map((option, idx) => (
                        <button
                          key={idx}
                          onClick={() => handleAnswer(ASSESSMENT_QUESTIONS[currentQuestionIdx].id, idx)}
                          className="w-full text-left p-5 rounded-xl border border-[#D0D7DE] hover:border-blue-300 hover:bg-blue-50 transition-all font-medium text-gray-800 flex items-center gap-4 group cursor-pointer active:bg-blue-100"
                        >
                          <span className="w-8 h-8 rounded-lg bg-[#F6F8FA] group-hover:bg-white border border-[#D0D7DE] flex items-center justify-center text-xs font-bold text-gray-500 group-hover:text-blue-600 transition-colors">
                            {String.fromCharCode(65 + idx)}
                          </span>
                          <span>{option}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {assessmentStep === "result" && (
                <div className="space-y-8">
                  <div className="grid grid-cols-12 gap-8">
                    {/* Left: Radar Chart */}
                    <div className="col-span-5 bg-white rounded-2xl border border-[#D0D7DE] p-6 shadow-sm flex flex-col items-center">
                      <h3 className="text-xs font-extrabold text-[#57606A] uppercase tracking-widest mb-6">能力雷达图</h3>
                      <div className="w-full h-[300px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <RadarChart cx="50%" cy="50%" outerRadius="80%" data={assessmentResults}>
                            <PolarGrid stroke="#D0D7DE" />
                            <PolarAngleAxis dataKey="subject" tick={{ fill: '#57606A', fontSize: 11, fontWeight: 700 }} />
                            <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                            <Radar
                              name="建模能力"
                              dataKey="A"
                              stroke="#0969DA"
                              fill="#0969DA"
                              fillOpacity={0.6}
                            />
                          </RadarChart>
                        </ResponsiveContainer>
                      </div>
                      <p className="text-[10px] text-[#8C959F] text-center mt-4">
                        * 分数基于你的答题表现与模型权重计算得出
                      </p>
                    </div>

                    {/* Right: Score Breakdown */}
                    <div className="col-span-7 space-y-6">
                      <div className="bg-white rounded-2xl border border-[#D0D7DE] p-8 shadow-sm flex items-center justify-between">
                        <div>
                          <p className="text-[11px] font-bold text-[#57606A] uppercase tracking-widest">总体评价</p>
                          <h2 className="text-3xl font-extrabold text-gray-900 mt-1">
                            {(() => {
                              const avg = assessmentResults ? assessmentResults.reduce((acc: any, curr: any) => acc + curr.A, 0) / 6 : 0;
                              if (avg >= 85) return "卓越不凡的建模者";
                              if (avg >= 70) return "潜力巨大的进阶者";
                              if (avg >= 50) return "稳扎稳打的探索者";
                              return "初出茅庐的开拓者";
                            })()}
                          </h2>
                        </div>
                        <div className="bg-blue-600 text-white w-20 h-20 rounded-full flex flex-col items-center justify-center shadow-lg">
                          <span className="text-2xl font-black">{Math.round(assessmentResults ? assessmentResults.reduce((acc: any, curr: any) => acc + curr.A, 0) / 6 : 0)}</span>
                          <span className="text-[8px] uppercase font-bold opacity-80">综合分</span>
                        </div>
                      </div>

                      <div className="bg-white rounded-2xl border border-[#D0D7DE] p-8 shadow-sm space-y-6 relative overflow-hidden">
                        <div className="flex items-center gap-2 mb-2">
                          <Sparkles className="w-4 h-4 text-amber-500" />
                          <h3 className="text-sm font-bold text-gray-900">AI 教练深度诊断报告</h3>
                        </div>

                        {isAnalyzing ? (
                          <div className="space-y-3">
                            <div className="h-4 bg-[#F6F8FA] rounded-md w-full animate-pulse"></div>
                            <div className="h-4 bg-[#F6F8FA] rounded-md w-3/4 animate-pulse"></div>
                            <div className="h-4 bg-[#F6F8FA] rounded-md w-5/6 animate-pulse"></div>
                            <div className="h-4 bg-[#F6F8FA] rounded-md w-1/2 animate-pulse"></div>
                            <p className="text-xs text-[#8C959F] italic font-medium">正在生成个性化学习建议...</p>
                          </div>
                        ) : (
                          <div className="text-sm text-[#24292F] leading-relaxed whitespace-pre-wrap font-medium bg-[#F6F8FA]/50 p-4 rounded-xl border border-blue-50/50">
                            {renderMarkdownWithMath(aiAnalysisPrompt)}
                          </div>
                        )}
                        
                        {!isAnalyzing && (
                          <div className="flex gap-4 mt-6">
                            <button
                              onClick={() => { setActiveTab("pathways"); setGlobalSearch(""); }}
                              className="flex-1 py-2.5 bg-blue-600 text-white text-xs font-bold rounded-lg hover:bg-blue-700 transition-all cursor-pointer shadow-md"
                            >
                              前往推荐学习路径
                            </button>
                            <button
                              onClick={handleStartAssessment}
                              className="flex-1 py-2.5 bg-white border border-[#D0D7DE] text-gray-700 text-xs font-bold rounded-lg hover:bg-[#F6F8FA] transition-all cursor-pointer"
                            >
                              重新测试
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
          {activeTab === "home" && (
            <div className="space-y-8 animate-fade-in">
              {/* Grand Welcome Card Grid */}
              <div className="grid grid-cols-12 gap-6">
                <div className="col-span-8">
                  <div className={`h-[320px] rounded-2xl relative overflow-hidden flex flex-col justify-between shadow-md transition-all duration-1000 bg-gradient-to-br group ${heroContent[heroIndex].bg}`}>
                    {/* Navigation Arrows */}
                    <button 
                      onClick={(e) => {
                        e.stopPropagation();
                        setHeroIndex((prev) => (prev - 1 + heroContent.length) % heroContent.length);
                      }}
                      className="absolute left-4 top-1/2 -translate-y-1/2 z-20 w-10 h-10 rounded-full bg-black/20 hover:bg-black/40 text-white flex items-center justify-center backdrop-blur-sm opacity-0 group-hover:opacity-100 transition-all duration-300 cursor-pointer border border-white/10"
                    >
                      <ChevronLeft className="w-6 h-6" />
                    </button>
                    <button 
                      onClick={(e) => {
                        e.stopPropagation();
                        setHeroIndex((prev) => (prev + 1) % heroContent.length);
                      }}
                      className="absolute right-4 top-1/2 -translate-y-1/2 z-20 w-10 h-10 rounded-full bg-black/20 hover:bg-black/40 text-white flex items-center justify-center backdrop-blur-sm opacity-0 group-hover:opacity-100 transition-all duration-300 cursor-pointer border border-white/10"
                    >
                      <ChevronRight className="w-6 h-6" />
                    </button>

                    <AnimatePresence mode="wait">
                      <motion.div
                        key={heroIndex}
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                        transition={{ duration: 0.5, ease: "easeOut" }}
                        className="p-8 h-full flex flex-col justify-between relative z-10"
                      >
                        <div className="space-y-4">
                          <div className="inline-flex items-center gap-2 px-3 py-1 bg-white/15 rounded-full text-[11px] font-bold text-white shadow-sm border border-white/20">
                            {heroContent[heroIndex].icon}
                            <span className="tracking-wide uppercase">{heroContent[heroIndex].tag}</span>
                          </div>
                          <h2 className="text-3xl font-extrabold tracking-tight text-white leading-tight">
                            {heroContent[heroIndex].title}
                          </h2>
                          <p className="text-white/90 text-sm max-w-lg leading-relaxed font-medium">
                            {heroContent[heroIndex].subtitle}
                          </p>
                        </div>

                        <div className="flex gap-4 mt-8">
                          <button
                            onClick={() => setActiveTab("pathways")}
                            className="bg-white text-blue-900 font-bold text-xs px-5 py-2.5 rounded-lg shadow-lg hover:bg-blue-50 transition-all flex items-center gap-2 cursor-pointer active:scale-95"
                          >
                            <span>智能学习路线</span>
                            <ArrowRight className="w-3.5 h-3.5" />
                          </button>
                          <button
                            onClick={() => setActiveTab("models")}
                            className="bg-white/10 border border-white/25 text-white font-bold text-xs px-5 py-2.5 rounded-lg hover:bg-white/20 transition-all cursor-pointer backdrop-blur-sm active:scale-95"
                          >
                            快速探索基础模型
                          </button>
                        </div>

                        {/* Pagination indicators inside hero */}
                        <div className="absolute bottom-6 right-8 flex gap-2">
                          {heroContent.map((_, idx) => (
                            <button
                              key={idx}
                              onClick={(e) => {
                                e.stopPropagation();
                                setHeroIndex(idx);
                              }}
                              className={`h-1.5 rounded-full transition-all duration-300 ${
                                heroIndex === idx ? "w-6 bg-white" : "w-1.5 bg-white/40 hover:bg-white/60"
                              }`}
                            />
                          ))}
                        </div>
                      </motion.div>
                    </AnimatePresence>

                    {/* Background abstract mathematics trace decoration */}
                    <AnimatePresence mode="wait">
                      <motion.div 
                        key={`accent-${heroIndex}`}
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 0.12, scale: 1 }}
                        exit={{ opacity: 0, scale: 1.1 }}
                        className="absolute right-0 bottom-0 text-[10rem] font-mono select-none pointer-events-none translate-x-10 translate-y-10 text-white italic font-bold"
                      >
                        {heroContent[heroIndex].accent}
                      </motion.div>
                    </AnimatePresence>
                  </div>
                </div>

                {/* Main Upcoming Contests Countdown widget */}
                <div className="col-span-4">
                  <div className="h-full bg-white border border-[#D0D7DE] rounded-2xl p-6 flex flex-col justify-between shadow-sm">
                    <div>
                      <div className="flex justify-between items-center mb-6">
                        <p className="text-[#57606A] text-xs uppercase font-extrabold tracking-widest font-mono">
                          🏆 即将到来的焦点赛事
                        </p>
                        <span className="w-2 h-2 rounded-full bg-blue-600 animate-ping"></span>
                      </div>

                      <div className="space-y-5">
                        <div className="flex gap-4">
                          <div className="w-12 h-12 bg-[#F6F8FA] rounded-xl flex flex-col items-center justify-center border border-[#D0D7DE] shrink-0">
                            <span className="text-[9px] uppercase font-bold text-red-500">FEB</span>
                            <span className="text-base font-extrabold text-gray-850 font-mono">05</span>
                          </div>
                          <div>
                            <h4 className="text-sm font-bold text-[#24292F]">美国数学建模竞赛 (MCM/ICM)</h4>
                            <p className="text-xs text-[#57606A] leading-normal">报名截止：1月底 | 极其重视英文写作与图表</p>
                          </div>
                        </div>

                        <div className="flex gap-4">
                          <div className="w-12 h-12 bg-[#F6F8FA] rounded-xl flex flex-col items-center justify-center border border-[#D0D7DE] shrink-0">
                            <span className="text-[9px] uppercase font-bold text-blue-600">SEP</span>
                            <span className="text-base font-extrabold text-gray-850 font-mono">10</span>
                          </div>
                          <div>
                            <h4 className="text-sm font-bold text-[#24292F]">全国大学生数学建模竞赛 (国赛)</h4>
                            <p className="text-xs text-[#57606A] leading-normal">保研核武器 | A/B/C 题多阶段选拔</p>
                          </div>
                        </div>
                      </div>
                    </div>

                    <button
                      onClick={() => setActiveTab("competitions")}
                      className="mt-6 w-full py-2.5 bg-[#F6F8FA] hover:bg-[#EFF2F5] text-[11px] text-blue-600 rounded-lg text-center font-bold border border-[#D0D7DE] hover:border-[#8C959F] transition-all cursor-pointer shadow-xs"
                    >
                      查看全部主流赛事详情 →
                    </button>
                  </div>
                </div>
              </div>

              {/* Home Quick Intro - "What is Math Modeling?" */}
              <div className="bg-white border border-[#D0D7DE] rounded-2xl p-8 space-y-6 shadow-sm">
                <div>
                  <h3 className="text-lg font-bold text-gray-900 flex items-center gap-2">
                    <span className="w-1.5 h-6 bg-blue-600 rounded-full"></span>
                    💡 什么是数学建模？
                  </h3>
                  <p className="text-sm text-[#57606A] mt-2 leading-relaxed">
                    数学建模就是根据实际问题建立数学模型，并求解模型以分析解决现实生活问题的全过程。简单来说就是：
                    <strong className="text-blue-600">“把生活问题翻译成数学公式，编写程序求解，并写成科研论文”</strong>。
                  </p>
                </div>

                <div className="grid grid-cols-3 gap-6">
                  <div className="p-4 bg-[#F8F9FA] rounded-xl border border-[#D0D7DE]">
                    <div className="w-8 h-8 rounded-lg bg-blue-100 text-blue-700 flex items-center justify-center font-extrabold text-sm mb-3">1</div>
                    <h4 className="text-sm font-bold text-gray-950 mb-1">翻译公式 (模型构建)</h4>
                    <p className="text-xs text-[#57606A] leading-relaxed">分析实际自变量与因变量关系，选用最优化的规划、统计预测或者决策评价模型。</p>
                  </div>
                  <div className="p-4 bg-[#F8F9FA] rounded-xl border border-[#D0D7DE]">
                    <div className="w-8 h-8 rounded-lg bg-teal-100 text-teal-700 flex items-center justify-center font-extrabold text-sm mb-3">2</div>
                    <h4 className="text-sm font-bold text-gray-950 mb-1">编程求解 (代码计算)</h4>
                    <p className="text-xs text-[#57606A] leading-relaxed">主要通过 Python / MATLAB 等工具包跑通数据计算、缺失差值与迭代寻优算法。</p>
                  </div>
                  <div className="p-4 bg-[#F8F9FA] rounded-xl border border-[#D0D7DE]">
                    <div className="w-8 h-8 rounded-lg bg-purple-100 text-purple-700 flex items-center justify-center font-extrabold text-sm mb-3">3</div>
                    <h4 className="text-sm font-bold text-gray-950 mb-1">论文表达 (图表写作)</h4>
                    <p className="text-xs text-[#57606A] leading-relaxed">将计算结果转化为精美的趋势折线图、热力图，并写出包含科学洞察的 LaTeX 学术论文。</p>
                  </div>
                </div>
              </div>

              {/* Recommended Core Models Grid */}
              <section className="space-y-4">
                <div className="flex justify-between items-center">
                  <h3 className="text-lg font-bold text-gray-900 flex items-center gap-2">
                    <span className="w-1.5 h-6 bg-purple-500 rounded-full"></span>
                    🔥 热门入门模型推荐
                  </h3>
                  <button
                    onClick={() => setActiveTab("models")}
                    className="text-xs text-blue-600 hover:text-blue-700 font-semibold hover:underline flex items-center gap-1 cursor-pointer"
                  >
                    <span>浏览全部模型库</span>
                    <ChevronRight className="w-3.5 h-3.5" />
                  </button>
                </div>

                <div className="grid grid-cols-4 gap-4 animate-fade-in-up">
                  {MODELS_DATA.slice(0, 4).map((model) => (
                    <div
                      key={model.id}
                      onClick={() => {
                        setSelectedModelId(model.id);
                        setActiveTab("models");
                      }}
                      className="bg-white border border-[#D0D7DE] p-5 rounded-xl hover:border-blue-500 hover:shadow-md transition-all duration-200 cursor-pointer group"
                    >
                      <div className="flex justify-between items-start mb-3">
                        <span className="text-[10px] bg-blue-50 text-blue-700 font-semibold px-2 py-0.5 rounded border border-blue-100">
                          {model.categoryName}
                        </span>
                        <ChevronRight className="w-4 h-4 text-[#57606A] group-hover:text-blue-600 group-hover:translate-x-1 transition-all" />
                      </div>
                      <h4 className="text-sm font-bold text-gray-900 group-hover:text-blue-600 mb-1 leading-snug">{model.name}</h4>
                      <p className="text-xs text-[#57606A] line-clamp-3 leading-relaxed mt-1.5">{model.summary}</p>
                    </div>
                  ))}
                </div>
              </section>

              {/* Frequently Asked Prompt Presets */}
              <section className="bg-white border border-[#D0D7DE] rounded-2xl p-6 shadow-sm">
                <h4 className="text-sm font-bold text-gray-950 mb-3 flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-amber-500" />
                  初学者常见困惑 (点击即可调起 AI Copilot 实时解答)：
                </h4>
                <div className="flex flex-wrap gap-2.5">
                  {[
                    "“第一次参加数学建模，三人队伍中怎么分配职责比较好？”",
                    "“美赛和国赛最大的区别在什么地方？”",
                    "“对毫无基础的新手，推荐学习 Python 还是 MATLAB？”",
                    "“建模论文中的摘要，有哪些必须要写的要素？”"
                  ].map((preset, index) => (
                    <button
                      key={index}
                      onClick={() => handlePresetQuestion(preset)}
                      className="text-xs bg-[#F6F8FA] hover:bg-blue-50 hover:text-blue-600 hover:border-blue-400 border border-[#D0D7DE] text-[#57606A] px-3.5 py-2 rounded-lg cursor-pointer transition-all duration-150 font-medium"
                    >
                      {preset}
                    </button>
                  ))}
                </div>
              </section>

            </div>
          )}

          {/* SCREEN 2: MODELS LIBRARY SCREEN */}
          {activeTab === "models" && (
            <div className="space-y-6">
              {!selectedModelId ? (
                <div className="space-y-6 animate-fade-in text-left">
                  {/* Top Filters Block */}
              <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-[#D0D7DE] pb-5">
                <div>
                  <h2 className="text-2xl font-bold text-gray-950 flex items-center gap-2">
                    <BookOpen className="w-6 h-6 text-[#0969DA]" />
                    <span>系统化数学模型学习库</span>
                  </h2>
                  <p className="text-xs text-[#57606A] mt-1">
                    覆盖数类核心经典的数学模型。深入剖析基本原理、数学模型公式、适用场景，并一键复制 Python/MATLAB 求解模板。
                  </p>
                </div>

                <div className="flex flex-wrap gap-2">
                  {[
                    { id: "all", label: "全部模型" },
                    { id: "preprocess", label: "预处理与特征工程" },
                    { id: "prediction", label: "预测模型" },
                    { id: "evaluation", label: "综合评价" },
                    { id: "difference-analysis", label: "差异性分析" },
                    { id: "correlation-analysis", label: "相关性分析" },
                    { id: "machine-learning", label: "机器学习分类" },
                    { id: "machine-learning-regression", label: "机器学习回归" },
                    { id: "stats-analysis", label: "统计分析" },
                    { id: "econometrics", label: "计量经济模型" },
                    { id: "programming-solvers", label: "规划求解" }
                  ].map((cat) => (
                    <button
                      key={cat.id}
                      onClick={() => setSelectedCategory(cat.id as any)}
                      className={`text-xs px-3.5 py-1.5 rounded-full border transition-all cursor-pointer font-semibold ${
                        selectedCategory === cat.id
                          ? "bg-[#0969DA] text-white border-[#0969DA] shadow-sm"
                          : "bg-white border-[#D0D7DE] text-[#57606A] hover:bg-[#F6F8FA] hover:text-[#1F2328]"
                      }`}
                    >
                      {cat.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Quick info if filter produced nothing */}
              {filteredModels.length === 0 ? (
                <div className="p-8 text-center bg-white border border-[#D0D7DE] rounded-xl text-[#57606A]">
                  未找到与当前检索和分类相对应的数学模型，请重置过滤输入。
                </div>
              ) : (
                /* Primary Models dynamic card grid */
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
                  {filteredModels.map((m) => {
                    return (
                      <div
                        key={m.id}
                        id={`model-card-${m.id}`}
                        onClick={() => {
                          setSelectedModelId(m.id);
                          // Smooth scroll main content to top so the details start at the top of the viewport
                          document.getElementById("main-content-scroll-container")?.scrollTo({ top: 0, behavior: "smooth" });
                        }}
                        className="bg-white border border-[#D0D7DE] hover:border-blue-500 rounded-xl overflow-hidden cursor-pointer transition-all duration-200 flex flex-col justify-between shadow-xs hover:shadow-md group"
                      >
                        <div className="p-5 space-y-4">
                          <div className="flex justify-between items-center">
                            <span className={`text-[10px] px-2 py-0.5 rounded border font-mono font-bold ${
                              m.category === "preprocess"
                                ? "bg-purple-50 text-purple-700 border-purple-200"
                                : m.category === "prediction"
                                ? "bg-amber-50 text-amber-700 border-amber-200"
                                : m.category === "evaluation"
                                ? "bg-teal-50 text-teal-700 border-teal-200"
                                : m.category === "difference-analysis"
                                ? "bg-indigo-50 text-indigo-700 border-indigo-200"
                                : m.category === "correlation-analysis"
                                ? "bg-sky-50 text-sky-700 border-sky-200"
                                : m.category === "machine-learning"
                                ? "bg-rose-50 text-rose-700 border-rose-200"
                                : m.category === "machine-learning-regression"
                                ? "bg-fuchsia-50 text-fuchsia-700 border-fuchsia-200"
                                : m.category === "stats-analysis"
                                ? "bg-cyan-50 text-cyan-700 border-cyan-200"
                                : m.category === "econometrics"
                                ? "bg-amber-50 text-amber-700 border-amber-200"
                                : "bg-emerald-50 text-emerald-700 border-emerald-200"
                            }`}>
                              {m.categoryName}
                            </span>
                            <span className="text-[11px] text-[#57606A] group-hover:text-blue-600 font-semibold transition-colors">查看详情 🔍</span>
                          </div>

                          <h3 className="text-base font-bold text-gray-950 group-hover:text-[#0969DA] font-sans transition-colors leading-snug">
                            {m.name}
                          </h3>

                          <p className="text-xs text-[#57606A] leading-relaxed line-clamp-3">
                            {m.summary}
                          </p>
                        </div>

                        {/* Bottom features snapshot summary */}
                        <div className="px-5 py-3 bg-[#F6F8FA] group-hover:bg-blue-50/20 border-t border-[#D0D7DE] group-hover:border-blue-200 flex items-center justify-between text-[11px] text-[#57606A] transition-all">
                          <span className="flex items-center gap-1 font-semibold">
                            <CheckCircle2 className="w-3.5 h-3.5 text-blue-500" />
                            {m.scenarios.length} 大场景
                          </span>
                          <span className="text-[#0969DA] font-extrabold group-hover:translate-x-0.5 transition-transform">点看原理 & 算力模板 →</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
                </div>
              ) : (() => {
                const model = MODELS_DATA.find((m) => m.id === selectedModelId);
                  if (!model) return null;

                  return (
                    <div className="space-y-6 animate-fade-in text-left">
                      {/* Top back selector */}
                      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#D0D7DE] pb-4">
                        <button
                          onClick={() => setSelectedModelId(null)}
                          className="group flex items-center gap-2 px-3.5 py-1.5 bg-white hover:bg-[#F6F8FA] text-xs font-bold text-[#0969DA] rounded-lg border border-[#D0D7DE] transition-all cursor-pointer shadow-xs hover:shadow-xs"
                        >
                          <ArrowLeft className="w-4 h-4 group-hover:-translate-x-0.5 transition-all" />
                          <span>返回系统化数学模型库</span>
                        </button>
                        <div className="text-xs text-[#57606A] font-semibold flex items-center gap-2">
                          <span>当前查阅详情：</span>
                          <span className="font-bold text-[#0969DA] bg-blue-50/75 px-2.5 py-1 rounded border border-blue-200 font-sans">
                            {model.name}
                          </span>
                        </div>
                      </div>

                      <div className="bg-white border border-[#D0D7DE] rounded-2xl overflow-hidden p-6 md:p-8 space-y-8 shadow-sm">
                        
                        {/* Header title panel */}
                        <div className="flex flex-col sm:flex-row justify-between sm:items-start gap-4 border-b border-[#D0D7DE] pb-5">
                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <span className={`text-xs px-2.5 py-1 rounded-full border font-mono font-bold ${
                                model.category === "preprocess"
                                  ? "bg-purple-50 text-purple-700 border-purple-200"
                                  : model.category === "prediction"
                                  ? "bg-amber-50 text-amber-700 border-amber-200"
                                  : model.category === "evaluation"
                                  ? "bg-teal-50 text-teal-700 border-teal-200"
                                  : model.category === "difference-analysis"
                                  ? "bg-indigo-50 text-indigo-700 border-indigo-200"
                                  : model.category === "correlation-analysis"
                                  ? "bg-sky-50 text-sky-700 border-sky-200"
                                  : model.category === "machine-learning"
                                  ? "bg-rose-50 text-rose-700 border-rose-200"
                                  : model.category === "machine-learning-regression"
                                  ? "bg-fuchsia-50 text-fuchsia-700 border-fuchsia-200"
                                  : model.category === "stats-analysis"
                                  ? "bg-cyan-50 text-cyan-700 border-cyan-200"
                                  : model.category === "econometrics"
                                  ? "bg-amber-50 text-amber-700 border-amber-200"
                                  : "bg-emerald-50 text-emerald-700 border-emerald-200"
                              }`}>
                                {model.categoryName}
                              </span>
                              <span className="text-xs text-[#57606A] font-medium">模型检索 ID: {model.id}</span>
                            </div>
                            <h3 className="text-2xl font-extrabold text-gray-950">{model.name}</h3>
                          </div>
                        </div>

                    {/* Left & Right layout content */}
                    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                      {/* Left: Principal concepts and scenes */}
                      <div className="lg:col-span-7 space-y-6">
                        
                        {/* 1. Summary Card */}
                        <div className="space-y-2">
                          <h4 className="text-xs font-extrabold text-[#57606A] uppercase tracking-wider flex items-center gap-1.5">
                            <Info className="w-4 h-4 text-[#0969DA]" />
                            📌 什么是这个模型？ (简介)
                          </h4>
                          <div className="p-4 bg-[#F8F9FA] rounded-xl border border-[#D0D7DE] text-xs leading-relaxed text-[#24292F]">
                            {model.summary}
                          </div>
                        </div>

                        {/* 2. Math principles + equations */}
                        <div className="space-y-2">
                          <h4 className="text-xs font-extrabold text-[#57606A] uppercase tracking-wider flex items-center gap-1.5">
                            <BrainCircuit className="w-4 h-4 text-emerald-600" />
                            🧠 基本原理及公式表达
                          </h4>
                          <div className="p-4 bg-[#F1F8FF] rounded-xl border border-[#BEE4FA] text-xs font-mono text-blue-900 whitespace-pre-wrap leading-relaxed">
                            {model.principles}
                          </div>
                        </div>

                        {/* 3. Applicability and restrictions */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          <div className="p-4 bg-[#F8F9FA] rounded-xl border border-[#D0D7DE] space-y-3">
                            <h5 className="text-xs font-bold text-gray-950 flex items-center gap-1.5">
                              <CheckCircle2 className="w-4 h-4 text-teal-600" />
                              📊 适用建模场景
                            </h5>
                            <ul className="space-y-1.5 text-xs text-[#24292F]">
                              {model.scenarios.map((sc, i) => (
                                <li key={i} className="flex gap-2 items-start">
                                  <span className="text-teal-600">•</span>
                                  <span>{sc}</span>
                                </li>
                              ))}
                            </ul>
                          </div>

                          <div className="p-4 bg-[#F8F9FA] rounded-xl border border-[#D0D7DE] space-y-3">
                            <h5 className="text-xs font-bold text-gray-950 flex items-center gap-1.5">
                              <AlertTriangle className="w-4 h-4 text-rose-600" />
                              ⚠️ 使用约束与条件限制
                            </h5>
                            <ul className="space-y-1.5 text-xs text-[#24292F]">
                              {model.limitations.map((lim, i) => (
                                <li key={i} className="flex gap-2 items-start">
                                  <span className="text-xs text-rose-600">•</span>
                                  <span>{lim}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        </div>

                        {/* 4. Practical Case Study */}
                        <div className="p-5 bg-gradient-to-r from-[#F6F8FA] to-[#EFF2F5] rounded-xl border border-[#D0D7DE] space-y-4">
                          <h4 className="text-xs font-extrabold text-[#0969DA] uppercase tracking-wider flex items-center gap-1.5">
                            <Lightbulb className="w-4 h-4 text-amber-500" />
                            🧪 极简经典案例：{model.caseStudy.title}
                          </h4>
                          
                          <div className="space-y-2">
                            <p className="text-xs text-[#24292F] leading-relaxed"><strong>问题描述：</strong>{model.caseStudy.description}</p>
                            <div className="bg-white p-4 rounded border border-[#D0D7DE] text-[11px] text-[#57606A] whitespace-pre-wrap leading-relaxed">
                              {model.caseStudy.solution}
                            </div>
                          </div>
                        </div>

                      </div>

                      {/* Right: Executable Code block template */}
                      <div className="lg:col-span-5 space-y-4">
                        <div className="flex justify-between items-center">
                          <h4 className="text-xs font-extrabold text-[#57606A] uppercase tracking-wider flex items-center gap-1.5">
                            <Terminal className="w-4 h-4 text-purple-600" />
                            💻 Python / MATLAB 求解模板
                          </h4>

                          {/* Quick selection tabs */}
                          <div className="flex bg-[#EFF2F5] p-0.5 rounded border border-[#D0D7DE]">
                            <button
                              onClick={() => setActiveCodeTab("python")}
                              className={`text-[10px] px-2 py-0.5 rounded font-mono font-semibold ${
                                activeCodeTab === "python"
                                  ? "bg-[#0969DA] text-white"
                                  : "text-[#57606A] hover:text-[#24292F]"
                              }`}
                            >
                              Python
                            </button>
                            <button
                              onClick={() => setActiveCodeTab("matlab")}
                              className={`text-[10px] px-2 py-0.5 rounded font-mono font-semibold ${
                                activeCodeTab === "matlab"
                                  ? "bg-[#0969DA] text-white"
                                  : "text-[#57606A] hover:text-[#24292F]"
                              }`}
                            >
                              MATLAB
                            </button>
                          </div>
                        </div>

                        {/* Active text of code */}
                        {(() => {
                          const activeCodeText =
                            activeCodeTab === "python"
                              ? model.code.python
                              : model.code.matlab;

                          return (
                            <div className="relative bg-[#0D1117] rounded-xl border border-[#D0D7DE] overflow-hidden flex flex-col shadow-inner">
                              {/* Action header on top code pane */}
                              <div className="px-4 py-2 bg-[#161B22] border-b border-[#30363D] flex justify-between items-center">
                                <span className="text-[10px] text-[#7D8590] font-mono">
                                  {activeCodeTab === "python" ? "main.py" : "solve.m"}
                                </span>
                                
                                <button
                                  onClick={() => copyToClipboard(activeCodeText || "")}
                                  className="text-[10px] text-blue-400 hover:text-white flex items-center gap-1 cursor-pointer"
                                >
                                  {copiedCodeFlag ? (
                                    <>
                                      <Check className="w-3 h-3 text-emerald-400" />
                                      <span className="text-emerald-400">已复制!</span>
                                    </>
                                  ) : (
                                    <>
                                      <Copy className="w-3 h-3" />
                                      <span>复制代码</span>
                                    </>
                                  )}
                                </button>
                              </div>

                              <pre className="p-4 overflow-x-auto text-[11px] text-purple-200 font-mono leading-relaxed whitespace-pre h-96">
                                {activeCodeText || "# 该自研模型目前尚未整理对应语言的模板求解模块。"}
                              </pre>
                            </div>
                          );
                        })()}

                        {/* Pre-formatted tips for run-time compilation */}
                        <div className="p-3.5 bg-blue-50 border border-blue-100 text-[11px] rounded-lg text-[#57606A] space-y-1.5 shadow-xs">
                          <p className="text-[#0969DA] font-extrabold flex items-center gap-1">
                            <Sparkles className="w-3.5 h-3.5 text-blue-500 animate-pulse" />
                            <span>代码运行前置提示：</span>
                          </p>
                          <p className="leading-relaxed">
                            - Python 模板需要提前跑通 <code className="text-blue-800 font-mono bg-blue-100/60 px-1 rounded">pip install scipy numpy statsmodels</code> 等数学库。<br />
                            - MATLAB 代码通常可直接黏贴入 Matlab 脚本窗口中一键执行，注意核实矩阵分号格式。
                          </p>
                        </div>

                        {/* Recommended videos / resources links */}
                        <div className="space-y-2">
                          <p className="text-xs text-[#57606A] font-bold">📎 推荐自修辅导资源：</p>
                          <div className="space-y-1.5">
                            {model.resources.map((res, is) => (
                              <a
                                key={is}
                                href={res.url}
                                target="_blank"
                                rel="noreferrer"
                                className="flex justify-between items-center p-2.5 bg-white hover:bg-[#F6F8FA] rounded border border-[#D0D7DE] text-xs text-blue-600 hover:text-blue-750 transition-all font-semibold shadow-xs"
                              >
                                <span>{res.title}</span>
                                <ExternalLink className="w-3.5 h-3.5 text-[#57606A]" />
                              </a>
                            ))}
                          </div>
                        </div>

                      </div>
                    </div>

                    {/* Copilot redirect widget */}
                    <div className="p-4 bg-gradient-to-r from-blue-50 to-teal-50/40 border border-blue-200 rounded-xl flex flex-col md:flex-row justify-between items-center gap-4">
                      <p className="text-xs text-gray-800 leading-normal">
                        <strong>遇到报错或公式不理解？</strong> 你可以点击向右下角 Copilot 助手提问：“解释 {model.name} 中代码求解各个因子的含义及求解优化怎么调参？”
                      </p>
                      <button
                        onClick={() => handlePresetQuestion(`为我深度剖析并解释一下 ${model.name} 的基本原理，它的求解局限性要怎么规避？`)}
                        className="px-4 py-1.5 bg-gradient-to-r from-[#0969DA] to-teal-600 hover:from-blue-700 hover:to-teal-700 text-white text-xs font-bold rounded-lg shadow-xs transition-all transform hover:scale-102 shrink-0"
                      >
                        AI 深度解答 🚀
                      </button>
                    </div>

                  </div>
                </div>
                );
              })()}

            </div>
          )}

          {/* SCREEN 3: COMPETITIONS LIST */}
          {activeTab === "competitions" && (
            <div className="space-y-6">
              
              {/* Header section */}
              <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-[#D0D7DE] pb-5">
                <div>
                  <h2 className="text-2xl font-bold text-gray-950 flex items-center gap-2">
                    <Award className="w-6 h-6 text-amber-500 animate-pulse" />
                    <span>全球与国内主流数学建模比赛</span>
                  </h2>
                  <p className="text-xs text-[#57606A] mt-1">
                    整理汇总数模大格局的核心三大赛（国赛、美赛、华为杯）以及大厂高校邀请挑战赛。提前制定备战日历、查看参赛门槛与保研认可度分析。
                  </p>
                </div>

                {/* Filter buttons on difficulty */}
                <div className="flex gap-2">
                  <span className="text-xs text-[#57606A] self-center font-semibold">筛选难度：</span>
                  {[
                    { val: null, label: "全部难度" },
                    { val: 3, label: "3星 (初试)" },
                    { val: 4, label: "4星 (极客)" },
                    { val: 5, label: "5星 (极具挑战)" }
                  ].map((item) => (
                    <button
                      key={item.label}
                      onClick={() => setSelectedDifficulty(item.val)}
                      className={`text-[11px] px-2.5 py-1 rounded-md border cursor-pointer font-semibold ${
                        selectedDifficulty === item.val
                          ? "bg-[#0969DA] text-white border-[#0969DA]"
                          : "bg-white border-[#D0D7DE] text-[#57606A] hover:bg-[#F6F8FA] hover:text-[#1F2328]"
                      }`}
                    >
                      {item.label}
                    </button>
                  ))}
                </div>
              </div>
              
              {/* TIMELINE VIEW (搞个时间轴线，在上面标出各个竞赛) */}
              <div id="competitions-annual-timeline" className="bg-white border border-[#D0D7DE] rounded-xl p-5 space-y-4 shadow-sm">
                <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
                  <div className="flex items-center gap-2">
                    <Calendar className="w-4 h-4 text-blue-600" />
                    <h3 className="text-sm font-bold text-gray-950">2026 年数模竞赛全年备战时间轴</h3>
                    <span className="text-[10px] text-[#57606A] hidden md:inline">（横向滚动查看 1-12 月，点击带标志月份进行快速筛选）</span>
                  </div>
                  {selectedMonth !== null && (
                    <button
                      onClick={() => setSelectedMonth(null)}
                      className="text-xs text-blue-600 hover:text-blue-800 flex items-center gap-1 font-bold group cursor-pointer"
                    >
                      <span>清除月份筛选</span>
                      <X className="w-3.5 h-3.5 text-blue-600 group-hover:text-blue-800 transition-colors" />
                    </button>
                  )}
                </div>

                <div className="w-full overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-gray-200 scrollbar-track-transparent">
                  <div className="min-w-[950px] relative py-3 px-1 flex justify-between items-stretch gap-2">
                    {/* The horizontal connecting line */}
                    <div className="absolute top-[28px] left-6 right-6 h-[2px] bg-[#E1E4E8] z-0"></div>

                    {/* Month cells */}
                    {Array.from({ length: 12 }).map((_, idx) => {
                      const monthNum = idx + 1;
                      const monthComps = COMPETITIONS_DATA.filter(c => c.month === monthNum);
                      const isSelected = selectedMonth === monthNum;
                      const hasComps = monthComps.length > 0;

                      return (
                        <div
                          key={monthNum}
                          onClick={() => {
                            if (hasComps) {
                              setSelectedMonth(selectedMonth === monthNum ? null : monthNum);
                            }
                          }}
                          className={`flex-1 flex flex-col items-center select-none group min-w-[70px] transition-all ${
                            hasComps ? "cursor-pointer" : "opacity-35"
                          }`}
                        >
                          {/* Month Title */}
                          <p className={`text-xs font-bold font-mono transition-colors duration-200 mb-1.5 ${
                            isSelected
                              ? "text-blue-600 font-extrabold"
                              : "text-[#57606A] group-hover:text-[#1F2328]"
                          }`}>
                            {monthNum}月
                          </p>

                          {/* Node circle node */}
                          <div className={`w-4 h-4 rounded-full border-2 flex items-center justify-center transition-all duration-300 z-10 ${
                            isSelected
                              ? "bg-[#0969DA] border-[#0969DA] scale-125 shadow-[0_0_8px_rgba(9,105,218,0.4)]"
                              : hasComps
                              ? "bg-white border-blue-600 group-hover:bg-blue-600 group-hover:scale-110"
                              : "bg-white border-[#D0D7DE]"
                          }`}>
                            {isSelected && <div className="w-1.5 h-1.5 bg-white rounded-full"></div>}
                          </div>

                          {/* Competition mini cards wrapper */}
                          <div className="mt-3.5 flex flex-col gap-1 w-full px-0.5 items-center">
                            {monthComps.length > 0 ? (
                              monthComps.map(c => (
                                <div
                                  key={c.id}
                                  className={`text-[9px] px-1.5 py-0.5 rounded border text-center font-bold leading-tight w-full truncate transition-all ${
                                    isSelected
                                      ? "bg-blue-50 text-blue-700 border-blue-200"
                                      : "bg-[#F6F8FA] text-[#24292F] border-[#D0D7DE] hover:border-blue-500/50"
                                  }`}
                                  title={`${c.name} (${c.alias})`}
                                >
                                  <span className="mr-0.5" role="img" aria-label="competition logo">{c.logoChar}</span>
                                  <span>{c.alias}</span>
                                </div>
                              ))
                            ) : (
                              <span className="text-[10px] text-gray-300 italic font-mono">—</span>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>

              {/* ACTIVE FILTER STATUS BANNER */}
              {selectedMonth !== null && (
                <div className="flex items-center justify-between p-4 bg-blue-50 border border-blue-200 rounded-xl text-xs text-blue-700 font-semibold shadow-xs animate-fade-in">
                  <span className="flex items-center gap-2">
                    <Calendar className="w-4 h-4 text-blue-600" />
                    <span>已经筛选：正在查看 <strong>{selectedMonth}月</strong> 进行/启动的数学建模竞赛（共 <strong>{filteredCompetitions.length}</strong> 项）</span>
                  </span>
                  <button
                    onClick={() => setSelectedMonth(null)}
                    className="px-3 py-1 bg-white hover:bg-[#F6F8FA] text-xs font-bold text-gray-800 border border-[#D0D7DE] rounded-lg transition-all cursor-pointer shadow-xs"
                  >
                    显示全年比赛
                  </button>
                </div>
              )}

              {/* Competition cards List */}
              {filteredCompetitions.length === 0 ? (
                <div className="p-8 text-center bg-white border border-[#D0D7DE] rounded-xl text-[#57606A]">
                  未找到所对应难度和搜索关键字的赛事。请尝试切换其他过滤器。
                </div>
              ) : (
                <div className="space-y-6">
                  {filteredCompetitions.map((c) => (
                    <div
                      key={c.id}
                      className="bg-white border border-[#D0D7DE] hover:border-blue-500/60 rounded-xl overflow-hidden p-6 transition-all duration-200 shadow-xs hover:shadow-md"
                    >
                      {/* Flex layout for logo, title and difficulty stars */}
                      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-[#D0D7DE] pb-4 mb-4">
                        <div className="flex items-center gap-3">
                          <span className="text-3xl filter saturate-100">{c.logoChar}</span>
                          <div>
                            <div className="flex items-center gap-2">
                              <h3 className="text-lg font-bold text-gray-950">{c.name}</h3>
                              <span className="text-xs bg-blue-50 text-blue-700 px-2.5 py-0.5 rounded border border-blue-100 font-bold font-mono">
                                {c.alias}
                              </span>
                            </div>
                            <p className="text-xs text-[#57606A] font-medium">适宜群体：
                              <span className="text-blue-700 font-bold ml-1 bg-blue-50 px-1.5 py-0.5 rounded text-[10px]">
                                {c.targetAudience}
                              </span>
                            </p>
                          </div>
                        </div>

                        {/* Stars level difficulty panel */}
                        <div className="flex items-center gap-1 bg-[#F6F8FA] px-3 py-1 rounded-lg border border-[#D0D7DE]">
                          <span className="text-xs text-[#57606A] mr-1 font-semibold">硬核挑战指数:</span>
                          {Array.from({ length: 5 }).map((_, i) => (
                            <Star
                              key={i}
                              className={`w-3.5 h-3.5 ${
                                i < c.difficulty ? "text-amber-400 fill-amber-400 animate-pulse" : "text-gray-300"
                              }`}
                            />
                          ))}
                        </div>
                      </div>

                      {/* Timeline table columns */}
                      <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
                        
                        {/* 1. Dates and timeline column */}
                        <div className="md:col-span-4 bg-[#F8F9FA] p-4 rounded-xl border border-[#D0D7DE] space-y-4">
                          <h4 className="text-xs font-extrabold text-[#57606A] uppercase tracking-wider flex items-center gap-1.5">
                            <Calendar className="w-4 h-4 text-rose-500 animate-pulse" />
                            📅 极其关键时间线
                          </h4>
                          
                          <div className="space-y-3 font-mono text-xs text-[#24292F]">
                            <div>
                              <p className="text-[#57606A] text-[10px] font-semibold uppercase">1. 报名登记阶段</p>
                              <p className="font-extrabold text-[#24292F]">{c.timeline.signup}</p>
                            </div>
                            <div>
                              <p className="text-[#57606A] text-[10px] font-semibold uppercase">2. 绝密拉练竞赛期</p>
                              <p className="font-extrabold text-teal-700">{c.timeline.contest}</p>
                            </div>
                            <div>
                              <p className="text-[#57606A] text-[10px] font-semibold uppercase">3. 榜单及审核结果</p>
                              <p className="font-medium text-gray-600">{c.timeline.results}</p>
                            </div>
                          </div>
                        </div>

                        {/* 2. Detailed description card with rich content */}
                        <div className="md:col-span-8 space-y-4">
                          <div>
                            <h4 className="text-xs font-bold text-gray-950">🏆 比赛官方简况</h4>
                            <p className="text-xs text-[#57606A] mt-1.5 leading-normal font-medium">{c.description}</p>
                          </div>

                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-2">
                            <div className="p-3 bg-blue-50/60 rounded-lg border border-blue-100">
                              <h5 className="text-[11px] font-extrabold text-blue-700 flex items-center gap-1">
                                <Users className="w-3.5 h-3.5 text-blue-500" />
                                👥 队员构成及门槛限制
                              </h5>
                              <p className="text-[11px] text-[#57606A] mt-1 leading-normal font-medium">{c.requirements}</p>
                            </div>

                            <div className="p-3 bg-amber-50/50 rounded-lg border border-amber-100">
                              <h5 className="text-[11px] font-extrabold text-amber-700 flex items-center gap-1">
                                <TrendingUp className="w-3.5 h-3.5" />
                                📈 含金量及保研/求职评级
                              </h5>
                              <p className="text-[11px] text-[#57606A] mt-1 leading-normal font-medium">{c.valueAnalysis}</p>
                            </div>
                          </div>

                          {/* Action area links */}
                          <div className="flex flex-wrap gap-4 pt-1 items-center">
                            <a
                              href={c.pastPapersUrl}
                              target="_blank"
                              rel="noreferrer"
                              className="px-4 py-2 bg-gradient-to-r from-blue-600 to-[#0969DA] hover:from-blue-700 hover:to-blue-700 text-white rounded-lg text-xs font-bold flex items-center gap-1.5 shadow-md flex items-center gap-1.5 shadow-md shrink-0 cursor-pointer"
                            >
                              <span>前往浏览历年真题及优秀论文</span>
                              <ExternalLink className="w-3.5 h-3.5" />
                            </a>

                            <button
                              onClick={() => handlePresetQuestion(`帮我专门定制一套参加 ${c.alias} 的 3 人队伍分工计划和具体赛前攻克手段`)}
                              className="px-4 py-2 bg-white hover:bg-[#F6F8FA] text-[#24292F] border border-[#D0D7DE] rounded-lg text-xs font-bold shadow-xs cursor-pointer transition-all hover:border-blue-500"
                            >
                              生成针对性备赛方案 🦾
                            </button>
                          </div>

                        </div>

                      </div>
                    </div>
                  ))}
                </div>
              )}

            </div>
          )}

          {/* SCREEN 4: LEARNING PATHWAYS */}
          {activeTab === "pathways" && (
            <div className="space-y-6">
              
              {/* Header and selection tabs for routes */}
              <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-[#D0D7DE] pb-5">
                <div>
                  <h2 className="text-2xl font-bold text-gray-950 flex items-center gap-2">
                    <Compass className="w-6 h-6 text-teal-600 animate-spin-slow" />
                    <span>分级数学建模成长推荐路径</span>
                  </h2>
                  <p className="text-xs text-[#57606A] mt-1">
                    告别盲目自学。我们携手数智建模专家编制的各阶段筑基路线，提供可量化的目标技能树。
                  </p>
                </div>

                {/* pathway switchers toggle info */}
                <div className="flex bg-[#F6F8FA] p-1 rounded-xl border border-[#D0D7DE]">
                  {PATHWAYS_DATA.map((p) => (
                    <button
                      key={p.id}
                      onClick={() => setSelectedPathId(p.id)}
                      className={`text-xs px-4 py-2 rounded-lg font-bold transition-all cursor-pointer ${
                        selectedPathId === p.id
                          ? "bg-[#0969DA] text-white shadow"
                          : "text-[#57606A] hover:text-[#1F2328]"
                      }`}
                    >
                      {p.name}
                    </button>
                  ))}
                </div>
              </div>

              {/* Displaying active selected route details */}
              {(() => {
                const pathObj = PATHWAYS_DATA.find((p) => p.id === selectedPathId) || PATHWAYS_DATA[0];
                return (
                  <div className="space-y-8 animate-fade-in">
                    
                    {/* Upper summary introduction metadata box */}
                    <div className="bg-gradient-to-r from-[#F6F8FA] to-white border border-[#D0D7DE] rounded-xl p-6 shadow-xs">
                      <p className="text-xs text-[#0969DA] uppercase tracking-wider font-extrabold font-mono mb-1">
                        🎯 当前路线概要定位
                      </p>
                      <h3 className="text-lg font-bold text-gray-950 mb-2">{pathObj.name}</h3>
                      <p className="text-xs text-[#57606A] leading-relaxed max-w-4xl font-semibold">{pathObj.description}</p>
                    </div>

                    {/* Sequential timeline progress steps wrapper */}
                    <div className="relative border-l-2 border-[#D0D7DE] ml-6 pl-8 space-y-8">
                      {pathObj.steps.map((step, idx) => (
                        <div key={idx} className="relative group">
                          {/* Left bullet marker node */}
                          <div className="absolute -left-14 top-0.5 w-12 h-12 rounded-full border-4 border-[#F6F8FA] bg-white group-hover:border-blue-500 flex items-center justify-center font-bold text-[#57606A] group-hover:text-blue-650 transition-all font-mono shadow-xs">
                            {idx + 1}
                          </div>

                          <div className="bg-white border border-[#D0D7DE] hover:border-blue-500/50 rounded-xl p-6 space-y-4 transition-all shadow-xs hover:shadow-md">
                            
                            {/* Title line */}
                            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[#D0D7DE] pb-3">
                              <h4 className="text-base font-bold text-gray-950 group-hover:text-blue-700 transition-colors">
                                {step.title}
                              </h4>
                              <span className="text-[11px] font-mono font-extrabold text-teal-700 bg-teal-50 px-2.5 py-1 rounded border border-teal-100">
                                {step.duration}
                              </span>
                            </div>

                            {/* Core description contents */}
                            <p className="text-xs text-[#57606A] leading-relaxed font-semibold">{step.description}</p>

                            {/* Skills badges tags bullet checklist */}
                            <div className="space-y-2">
                              <h5 className="text-[11px] font-extrabold text-gray-950 uppercase tracking-wider">🛠️ 攻克核心技能极：</h5>
                              <div className="flex flex-wrap gap-1.5">
                                {step.skills.map((sk, index) => (
                                  <span
                                    key={index}
                                    className="text-[10px] bg-blue-50 text-blue-700 px-2 py-0.5 rounded border border-blue-100 font-bold font-mono"
                                  >
                                    {sk}
                                  </span>
                                ))}
                              </div>
                            </div>

                            {/* Action links to required models */}
                            {step.recommendedModels.length > 0 && (
                              <div className="pt-2 flex flex-col sm:flex-row sm:items-center gap-3">
                                <span className="text-[11px] text-[#57606A] shrink-0 font-extrabold">🎯 对应推荐自研模型要领点:</span>
                                <div className="flex flex-wrap gap-2">
                                  {step.recommendedModels.map((rm) => {
                                    const modelObj = MODELS_DATA.find((m) => m.id === rm);
                                    if (!modelObj) return null;
                                    return (
                                      <button
                                        key={rm}
                                        onClick={() => {
                                          setSelectedModelId(modelObj.id);
                                          setActiveTab("models");
                                          // scroll to the expanded item as nice UX helper
                                          setTimeout(() => {
                                            document.getElementById(`model-card-${rm}`)?.scrollIntoView({ behavior: "smooth" });
                                          }, 100);
                                        }}
                                        className="text-[10px] bg-[#F6F8FA] hover:bg-blue-50 hover:text-blue-700 hover:border-blue-300 text-blue-600 px-2.5 py-1 rounded-md transition-all border border-[#D0D7DE] cursor-pointer font-bold shadow-xs"
                                      >
                                        展开：{modelObj.name.split(" ")[0]}
                                      </button>
                                    );
                                  })}
                                </div>
                              </div>
                            )}

                          </div>
                        </div>
                      ))}
                    </div>

                    {/* Level certification footer banner */}
                    <div className="p-6 bg-[#F8F9FA] border border-[#D0D7DE] rounded-2xl flex flex-col md:flex-row justify-between items-center gap-4 shadow-xs">
                      <div className="space-y-1">
                        <h4 className="text-sm font-bold text-gray-950 flex items-center gap-1.5">
                          <CheckCircle2 className="w-4 h-4 text-emerald-600 animate-pulse" />
                          <span>获得数模平台完赛见证证书认证</span>
                        </h4>
                        <p className="text-xs text-[#57606A] font-medium">
                          按照上述推荐路径，复现 10 个模型算法和一篇完整大奖论文 of 支撑求解包即可解锁定制化证书。
                        </p>
                      </div>
                      <button className="px-5 py-2 bg-gradient-to-r from-blue-600 to-teal-600 hover:from-blue-700 hover:to-teal-700 text-white text-xs font-bold rounded-lg transition-all flex items-center gap-1.5 shrink-0 shadow-sm cursor-not-allowed opacity-80">
                        <Lock className="w-3.5 h-3.5" />
                        <span>未解锁证书审核</span>
                      </button>
                    </div>

                  </div>
                );
              })()}

            </div>
          )}

          {/* SCREEN 5: USER PROFILE */}
          {activeTab === "profile" && (
            <div id="profile-parent-card" className="space-y-8 animate-fade-in">
              {/* Header Box */}
              <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-[#D0D7DE] pb-5">
                <div>
                  <h2 className="text-2xl font-bold text-gray-950 flex items-center gap-2">
                    <User className="w-6 h-6 text-[#0969DA]" />
                    <span>个人主页</span>
                  </h2>
                  <p className="text-xs text-[#57606A] mt-1">
                    定制化你的数模简历肖像、基本资料和团队角色职责。
                  </p>
                </div>
                
                {/* Save Feedback Toast */}
                <div className="flex items-center gap-3 w-full md:w-auto justify-end">
                  {saveStatus === "success" && (
                    <span className="text-xs text-emerald-700 bg-emerald-50 px-3 py-1.5 rounded-lg border border-emerald-200 font-bold flex items-center gap-1.5 animate-bounce shadow-xs">
                      <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                      保存成功！个人设置已应用并写入本地缓存。
                    </span>
                  )}
                  {saveStatus === "error" && (
                    <span className="text-xs text-red-700 bg-red-50 px-3 py-1.5 rounded-lg border border-red-200 font-bold flex items-center gap-1.5 shadow-xs">
                      <AlertTriangle className="w-4 h-4 text-red-600" />
                      写入失败，本地浏览器存储不可用！
                    </span>
                  )}
                  <button
                    id="profile-btn-save-master"
                    onClick={handleSaveProfile}
                    className="px-5 py-2.5 bg-[#0969DA] hover:bg-blue-700 text-white text-xs font-bold rounded-lg shadow-sm hover:shadow-md transition-all flex items-center gap-1.5 cursor-pointer"
                  >
                    <Check className="w-4 h-4" />
                    <span>保存个人资料</span>
                  </button>
                </div>
              </div>

              {/* Main Content Info Grid */}
              <div className="grid grid-cols-12 gap-6">
                
                {/* Left Column: Avatar management & Presets */}
                <div className="col-span-12 lg:col-span-4 space-y-6">
                  <div id="profile-avatar-card" className="bg-white border border-[#D0D7DE] rounded-xl p-6 shadow-xs space-y-6">
                    <h3 className="text-sm font-bold text-gray-950 flex items-center gap-2 border-b border-gray-100 pb-3">
                      <User className="w-4 h-4 text-[#0969DA]" />
                      <span>个人肖像简历</span>
                    </h3>

                    {/* Image display */}
                    <div className="flex flex-col items-center justify-center space-y-4">
                      <div className="relative group">
                        {editAvatar ? (
                          <img
                            src={editAvatar}
                            alt="avatar preview"
                            className="w-28 h-28 rounded-full object-cover border-4 border-blue-50 cursor-pointer shadow-md hover:scale-102 transition-all animate-fade-in"
                            referrerPolicy="no-referrer"
                          />
                        ) : (
                          <div className="w-28 h-28 rounded-full bg-gradient-to-tr from-[#0969DA] to-[#388BFD] flex items-center justify-center font-bold text-2xl text-white shadow-md border-4 border-blue-50 cursor-pointer hover:scale-102 transition-all">
                            {editName.substring(0, 2).toUpperCase()}
                          </div>
                        )}
                        {editAvatar && (
                          <button
                            onClick={() => setEditAvatar("")}
                            className="absolute -top-1 -right-1 p-1 bg-red-100 hover:bg-red-205 text-red-700 rounded-full border border-red-200 cursor-pointer transition-all shadow-xs"
                            title="清除头像"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        )}
                      </div>
                      <div className="text-center">
                        <h4 className="text-sm font-bold text-gray-950">{editName || "数模新星"}</h4>
                        <p className="text-[11px] text-[#57606A] font-medium font-mono mt-0.5">{editEmail || "未配置邮箱"}</p>
                      </div>
                    </div>

                    {/* Drag-and-drop and Click selector component */}
                    <div
                      onDragEnter={handleDrag}
                      onDragOver={handleDrag}
                      onDragLeave={handleDrag}
                      onDrop={handleDrop}
                      className={`border-2 border-dashed rounded-xl p-4 text-center transition-all cursor-pointer ${
                        dragActive
                          ? "border-[#0969DA] bg-blue-50/20"
                          : "border-[#D0D7DE] hover:border-[#0969DA] bg-[#F6F8FA] hover:bg-white"
                      }`}
                      onClick={() => document.getElementById("avatar-file-input")?.click()}
                    >
                      <input
                        id="avatar-file-input"
                        type="file"
                        accept="image/*"
                        className="hidden"
                        onChange={handleFileSelect}
                      />
                      <UploadCloud className="w-6 h-6 text-[#57606A] mx-auto mb-2 animate-bounce" />
                      <p className="text-[11px] font-bold text-gray-950">拖拽头像图片到这里</p>
                      <p className="text-[10px] text-[#57606A] mt-1">支持 PNG, JPG, JPEG, 大小限制在 2MB 内</p>
                    </div>

                    {/* Predesigned scientific avatars picklist */}
                    <div className="space-y-2.5">
                      <p className="text-[11px] font-bold text-[#57606A]">🎯 或者选择极客公式免签学术头像：</p>
                      <div className="grid grid-cols-3 gap-2">
                        {[
                          { name: "探险家", symbol: "∑", gradient: "from-blue-600 to-indigo-650 text-white bg-gradient-to-tr" },
                          { name: "极客", symbol: "π", gradient: "from-rose-500 to-orange-500 text-white bg-gradient-to-tr" },
                          { name: "极值捕手", symbol: "∫", gradient: "from-teal-500 to-emerald-600 text-white bg-gradient-to-tr" },
                          { name: "物理编程", symbol: "💻", gradient: "from-slate-700 to-slate-900 text-white bg-gradient-to-tr" },
                          { name: "论文笔杆", symbol: "📝", gradient: "from-purple-500 to-fuchsia-600 text-white bg-gradient-to-tr" },
                          { name: "数模先机", symbol: "📊", gradient: "from-cyan-500 to-blue-600 text-white bg-gradient-to-tr" }
                        ].map((preset, idx) => {
                          const selectPresetAvatar = () => {
                            const fillHex = preset.symbol === "∑" ? "4F46E5" : preset.symbol === "π" ? "F97316" : preset.symbol === "∫" ? "0D9488" : preset.symbol === "💻" ? "1E293B" : preset.symbol === "📝" ? "D946EF" : "0284C7";
                            const presetSvg = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120"><rect width="120" height="120" rx="60" fill="%23${fillHex}"/><text x="50%25" y="55%25" dominant-baseline="middle" text-anchor="middle" font-family="system-ui,sans-serif" font-size="54" font-weight="bold" fill="white">${preset.symbol}</text></svg>`;
                            setEditAvatar(presetSvg);
                          };

                          return (
                            <button
                              key={idx}
                              onClick={selectPresetAvatar}
                              className="p-1.5 border border-[#D0D7DE] hover:border-blue-500 rounded-lg flex flex-col items-center hover:bg-blue-50/10 cursor-pointer transition-all"
                            >
                              <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm ${preset.gradient}`}>
                                {preset.symbol}
                              </div>
                              <span className="text-[9px] text-[#57606A] mt-1 font-semibold">{preset.name}</span>
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Right Column: User detailed form */}
                <div className="col-span-12 lg:col-span-8 space-y-6">
                  <div id="profile-info-form" className="bg-white border border-[#D0D7DE] rounded-xl p-6 shadow-xs space-y-6">
                    <h3 className="text-sm font-bold text-gray-950 flex items-center gap-2 border-b border-gray-100 pb-3">
                      <Sliders className="w-4 h-4 text-[#0969DA]" />
                      <span>个人基本资料档案</span>
                    </h3>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                      
                      {/* Name input */}
                      <div className="space-y-1.5">
                        <label className="text-xs font-bold text-[#57606A] flex items-center gap-1">
                          昵称 / 姓名:
                        </label>
                        <input
                          id="profile-input-name"
                          type="text"
                          value={editName}
                          onChange={(e) => setEditName(e.target.value)}
                          placeholder="请输入用户名/笔名..."
                          className="w-full bg-[#F6F8FA] border border-[#D0D7DE] rounded-lg px-3.5 py-2 text-xs focus:outline-none focus:bg-white focus:border-[#0969DA] focus:ring-1 focus:ring-[#0969DA] font-semibold"
                        />
                      </div>

                      {/* Email input */}
                      <div className="space-y-1.5">
                        <label className="text-xs font-bold text-[#57606A]">
                          电子邮箱:
                        </label>
                        <input
                          id="profile-input-email"
                          type="email"
                          value={editEmail}
                          onChange={(e) => setEditEmail(e.target.value)}
                          placeholder="请输入邮箱地址..."
                          className="w-full bg-[#F6F8FA] border border-[#D0D7DE] rounded-lg px-3.5 py-2 text-xs focus:outline-none focus:bg-white focus:border-[#0969DA] focus:ring-1 focus:ring-[#0969DA] font-mono font-semibold"
                        />
                      </div>

                      {/* Gender input */}
                      <div className="space-y-1.5">
                        <label className="text-xs font-bold text-[#57606A]">
                          性别:
                        </label>
                        <select
                          id="profile-input-gender"
                          value={editGender}
                          onChange={(e) => setEditGender(e.target.value)}
                          className="w-full bg-[#F6F8FA] border border-[#D0D7DE] rounded-lg px-3 py-2 text-xs focus:outline-none focus:bg-white focus:border-[#0969DA] font-semibold"
                        >
                          <option value="男">男 (Male)</option>
                          <option value="女">女 (Female)</option>
                          <option value="保密">保密 (Undisclosed)</option>
                        </select>
                      </div>

                      {/* Age input */}
                      <div className="space-y-1.5">
                        <label className="text-xs font-bold text-[#57606A]">
                          年龄:
                        </label>
                        <input
                          id="profile-input-age"
                          type="number"
                          value={editAge}
                          onChange={(e) => setEditAge(e.target.value)}
                          placeholder="请输入年龄..."
                          className="w-full bg-[#F6F8FA] border border-[#D0D7DE] rounded-lg px-3.5 py-2 text-xs focus:outline-none focus:bg-white focus:border-[#0969DA] focus:ring-1 focus:ring-[#0969DA] font-mono font-semibold"
                        />
                      </div>

                      {/* University */}
                      <div className="space-y-1.5">
                        <label className="text-xs font-bold text-[#57606A]">
                          就读高校 / 单位:
                        </label>
                        <input
                          id="profile-input-school"
                          type="text"
                          value={editSchool}
                          onChange={(e) => setEditSchool(e.target.value)}
                          placeholder="名牌大学、研究所等..."
                          className="w-full bg-[#F6F8FA] border border-[#D0D7DE] rounded-lg px-3.5 py-2 text-xs focus:outline-none focus:bg-white focus:border-[#0969DA] focus:ring-1 focus:ring-[#0969DA] font-semibold"
                        />
                      </div>

                      {/* Major */}
                      <div className="space-y-1.5">
                        <label className="text-xs font-bold text-[#57606A]">
                          专业主修:
                        </label>
                        <input
                          id="profile-input-major"
                          type="text"
                          value={editMajor}
                          onChange={(e) => setEditMajor(e.target.value)}
                          placeholder="如 数学、计算机、物理等..."
                          className="w-full bg-[#F6F8FA] border border-[#D0D7DE] rounded-lg px-3.5 py-2 text-xs focus:outline-none focus:bg-white focus:border-[#0969DA] focus:ring-1 focus:ring-[#0969DA] font-semibold"
                        />
                      </div>

                      {/* Core Modeling Role */}
                      <div className="col-span-1 md:col-span-2 space-y-2">
                        <label className="text-xs font-bold text-[#57606A]">
                          团队数模竞赛角色定位 (选择最擅长的职责):
                        </label>
                        <div className="flex flex-col gap-2">
                          {[
                            "建模手 (负责主推数学算式、理清关系、构建物理图景)",
                            "编程手 (负责算法编写、方程求解、绘制科学可视化折线)",
                            "论文手 (负责LaTeX排版、摘要撰写、英美风图表布局)",
                            "全能手 & 队长 (主导全面进程、合理控制工作节奏与任务协作)"
                          ].map((roleOption) => {
                            const isSelected = editRole === roleOption;
                            return (
                              <button
                                key={roleOption}
                                type="button"
                                onClick={() => setEditRole(roleOption)}
                                className={`text-xs px-3.5 py-2 rounded-lg border font-semibold transition-all cursor-pointer text-left w-full ${
                                  isSelected
                                    ? "bg-blue-50/70 text-[#0969DA] border-[#0969DA] shadow-xs"
                                    : "bg-[#F6F8FA] hover:bg-[#EFF2F5] text-gray-700 border-[#D0D7DE]"
                                }`}
                              >
                                <span className="flex items-center gap-2">
                                  <span className={`w-3.5 h-3.5 rounded-full border flex items-center justify-center shrink-0 ${isSelected ? "border-[#0969DA] bg-[#0969DA]" : "border-gray-300"}`}>
                                    {isSelected && <span className="w-1.5 h-1.5 bg-white rounded-full"></span>}
                                  </span>
                                  <span>{roleOption}</span>
                                </span>
                              </button>
                            );
                          })}
                        </div>
                      </div>

                    </div>
                  </div>
                </div>

              </div>
            </div>
          )}

          {/* SCREEN 6: CONFIGURATION CENTER */}
          {activeTab === "settings" && (
            <div id="settings-parent-card" className="space-y-8 animate-fade-in">
              {/* Header Box */}
              <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-[#D0D7DE] pb-5">
                <div>
                  <h2 className="text-2xl font-bold text-gray-950 flex items-center gap-2">
                    <Sliders className="w-6 h-6 text-[#0969DA]" />
                    <span>配置中心</span>
                  </h2>
                  <p className="text-xs text-[#57606A] mt-1">
                    在此配置私有阿里云百炼大模型接口与算力托管设置。
                  </p>
                </div>
                
                {/* Save Feedback Toast */}
                <div className="flex items-center gap-3 w-full md:w-auto justify-end">
                  {saveStatus === "success" && (
                    <span className="text-xs text-emerald-700 bg-emerald-50 px-3 py-1.5 rounded-lg border border-emerald-200 font-bold flex items-center gap-1.5 animate-bounce shadow-xs">
                      <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                      保存成功！算力配置已应用并写入本地缓存。
                    </span>
                  )}
                  {saveStatus === "error" && (
                    <span className="text-xs text-red-700 bg-red-50 px-3 py-1.5 rounded-lg border border-red-200 font-bold flex items-center gap-1.5 shadow-xs">
                      <AlertTriangle className="w-4 h-4 text-red-600" />
                      写入失败，本地浏览器存储不可用！
                    </span>
                  )}
                  <button
                    id="settings-btn-save-master"
                    onClick={handleSaveProfile}
                    className="px-5 py-2.5 bg-[#0969DA] hover:bg-blue-700 text-white text-xs font-bold rounded-lg shadow-sm hover:shadow-md transition-all flex items-center gap-1.5 cursor-pointer"
                  >
                    <Check className="w-4 h-4" />
                    <span>保存接口配置</span>
                  </button>
                </div>
              </div>

              {/* Alibaba Bailian config card */}
              <div id="settings-bailian-config" className="bg-white border border-[#D0D7DE] rounded-xl p-6 shadow-xs space-y-6">
                <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-2 border-b border-gray-100 pb-3">
                  <h3 className="text-sm font-bold text-gray-950 flex items-center gap-2">
                    <Key className="w-4 h-4 text-amber-500 animate-pulse" />
                    <span>阿里云百炼通义千问 API 算力枢纽</span>
                    <span className="text-[10px] bg-amber-50 border border-amber-200 text-amber-600 px-1.5 py-0.5 rounded font-mono font-bold animate-pulse">
                      自主大模型托管
                    </span>
                  </h3>
                  <a
                    href="https://bailian.console.aliyun.com/"
                    target="_blank"
                    rel="noreferrer"
                    className="text-xs text-blue-600 hover:text-blue-700 font-bold flex items-center gap-1 cursor-pointer"
                  >
                    <span>阿里云百炼控制台入口 ↗</span>
                  </a>
                </div>

                {/* Instruction guide */}
                <div className="p-4 bg-amber-50/45 border border-amber-200/50 rounded-xl space-y-2">
                  <p className="text-xs text-amber-850 font-bold flex items-center gap-1.5">
                    <Sparkles className="w-4 h-4 text-amber-600" />
                    <span>为什么接入通义千问 API？</span>
                  </p>
                  <p className="text-[11px] text-[#57606A] leading-relaxed">
                    由于默认公共算力接口可能存在限流、拥堵。在下方配置您申请的<strong>阿里云百炼 API 密钥</strong>后，内置的提问助手聊天和模型解释将<strong>本地直连</strong>升级为通义千问 Qwen 大模型服务。我们已在 Express 服务端开辟了 API 极速代理通道。通义千问是中国数学建模赛题公认理解极其敏捷、甚至在代码生成能力上顶尖擅长的大语言模型。
                  </p>
                </div>

                {/* ===== 文本模型配置区 ===== */}
                <div className="border border-blue-200 rounded-xl p-5 bg-blue-50/20 space-y-4">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold text-blue-700 bg-blue-100 border border-blue-200 px-2 py-0.5 rounded">文本模型</span>
                    <span className="text-[11px] text-[#57606A]">用于聊天对话、模型解释、代码生成等文本交互</span>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Text API Key Input */}
                    <div className="space-y-1.5 relative">
                      <label className="text-xs font-bold text-[#57606A] flex items-center justify-between">
                        <span>百炼 API Key (sk-...)</span>
                        <span className="text-[10px] text-gray-400 font-mono">localStorage 加密存储</span>
                      </label>
                      <div className="relative">
                        <input
                          id="profile-input-apikey"
                          type={showApiKey ? "text" : "password"}
                          value={editApiKey}
                          onChange={(e) => setEditApiKey(e.target.value)}
                          placeholder="请输入 sk-************************"
                          className="w-full bg-[#F6F8FA] border border-[#D0D7DE] rounded-lg pl-10 pr-10 py-2 text-xs focus:outline-none focus:bg-white focus:border-[#0969DA] focus:ring-1 focus:ring-[#0969DA] font-mono font-semibold"
                        />
                        <Key className="w-4 h-4 text-gray-400 absolute left-3 top-2.5" />
                        <button
                          type="button"
                          onClick={() => setShowApiKey(!showApiKey)}
                          className="absolute right-3 top-2.5 text-gray-400 hover:text-[#24292F] cursor-pointer"
                        >
                          {showApiKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                        </button>
                      </div>
                    </div>

                    {/* Text Model Select */}
                    <div className="space-y-1.5">
                      <label className="text-xs font-bold text-[#57606A] flex justify-between">
                        <span>文本模型选项</span>
                        <span className="text-[10px] text-gray-400">纯文本推理引擎</span>
                      </label>
                      <select
                        id="profile-select-model"
                        value={editModel}
                        onChange={(e) => setEditModel(e.target.value)}
                        className="w-full bg-[#F6F8FA] border border-[#D0D7DE] rounded-lg px-3 py-2 text-xs focus:outline-none focus:bg-white focus:border-[#0969DA] font-mono font-bold"
                      >
                        <option value="qwen3.7-plus">qwen3.7-plus</option>
                        <option value="deepseek-v4-flash">deepseek-v4-flash</option>
                        <option value="qwen3.6-flash-2026-04-16">qwen3.6-flash-2026-04-16</option>
                        <option value="qwen3.6-35b-a3b">qwen3.6-35b-a3b</option>
                        <option value="qwen3.7-max-2026-05-17">qwen3.7-max-2026-05-17</option>
                        <option value="glm-5.1">glm-5.1</option>
                        <option value="qwen3.6-plus-2026-04-02">qwen3.6-plus-2026-04-02</option>
                        <option value="qwen3.7-max-preview">qwen3.7-max-preview</option>
                        <option value="qwen3.6-plus">qwen3.6-plus</option>
                        <option value="qwen3.5-plus-2026-04-20">qwen3.5-plus-2026-04-20</option>
                      </select>
                    </div>
                  </div>

                  {/* Text connection test */}
                  {testFeedback && (
                    <div className={`p-3 rounded-lg border text-xs flex items-start gap-2 leading-relaxed font-semibold ${
                      testStatus === "success" ? "bg-emerald-50 border-emerald-200 text-emerald-800" : "bg-red-50 border-red-200 text-red-800"
                    }`}>
                      {testStatus === "success" ? <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" /> : <AlertTriangle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />}
                      <div>
                        <p className="font-bold">{testStatus === "success" ? "文本模型连通性正常" : "文本模型连通性异常"}</p>
                        <p className="font-mono text-[11px] whitespace-pre-wrap">{testFeedback}</p>
                      </div>
                    </div>
                  )}

                  <div className="flex justify-end">
                    <button
                      id="profile-btn-test-connection"
                      type="button"
                      disabled={testStatus === "testing"}
                      onClick={handleTestBailianConnection}
                      className="px-4 py-2 bg-white border border-[#D0D7DE] hover:border-[#8C959F] hover:bg-gray-50 text-xs text-[#24292F] font-bold rounded-lg transition-all flex items-center gap-1.5 shadow-xs cursor-pointer"
                    >
                      {testStatus === "testing" ? (
                        <><span className="inline-block w-3 h-3 border-2 border-amber-600 border-t-transparent rounded-full animate-spin"></span><span>校验中...</span></>
                      ) : (
                        <><Sparkles className="w-4 h-4 text-amber-600" /><span>测试文本模型连通性</span></>
                      )}
                    </button>
                  </div>
                </div>

                {/* ===== 视觉模型配置区 ===== */}
                <div className="border border-purple-200 rounded-xl p-5 bg-purple-50/20 space-y-4">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold text-purple-700 bg-purple-100 border border-purple-200 px-2 py-0.5 rounded">视觉模型</span>
                    <span className="text-[11px] text-[#57606A]">用于解析上传的 PDF、图片文件内容（需独立配置 API Key）</span>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Vision API Key Input */}
                    <div className="space-y-1.5 relative">
                      <label className="text-xs font-bold text-[#57606A] flex items-center justify-between">
                        <span>视觉模型 API Key (sk-...)</span>
                        <span className="text-[10px] text-gray-400 font-mono">localStorage 加密存储</span>
                      </label>
                      <div className="relative">
                        <input
                          id="profile-input-vision-apikey"
                          type={showVisionApiKey ? "text" : "password"}
                          value={editVisionApiKey}
                          onChange={(e) => setEditVisionApiKey(e.target.value)}
                          placeholder="请输入视觉模型的 sk-************************"
                          className="w-full bg-[#F6F8FA] border border-[#D0D7DE] rounded-lg pl-10 pr-10 py-2 text-xs focus:outline-none focus:bg-white focus:border-[#7C3AED] focus:ring-1 focus:ring-[#7C3AED] font-mono font-semibold"
                        />
                        <Key className="w-4 h-4 text-gray-400 absolute left-3 top-2.5" />
                        <button
                          type="button"
                          onClick={() => setShowVisionApiKey(!showVisionApiKey)}
                          className="absolute right-3 top-2.5 text-gray-400 hover:text-[#24292F] cursor-pointer"
                        >
                          {showVisionApiKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                        </button>
                      </div>
                      <p className="text-[10px] text-gray-400">视觉模型可与文本模型使用不同的 API Key，两个 Key 的额度独立计算</p>
                    </div>

                    {/* Vision Model Select */}
                    <div className="space-y-1.5">
                      <label className="text-xs font-bold text-[#57606A] flex justify-between">
                        <span>视觉模型选项</span>
                        <span className="text-[10px] text-gray-400">支持图片 + PDF 文字识别</span>
                      </label>
                      <select
                        id="profile-select-vision-model"
                        value={editVisionModel}
                        onChange={(e) => setEditVisionModel(e.target.value)}
                        className="w-full bg-[#F6F8FA] border border-[#D0D7DE] rounded-lg px-3 py-2 text-xs focus:outline-none focus:bg-white focus:border-[#7C3AED] font-mono font-bold"
                      >
                        <option value="qwen3-vl-plus">qwen3-vl-plus</option>
                        <option value="qwen3-vl-plus-2025-12-19">qwen3-vl-plus-2025-12-19</option>
                        <option value="qwen3-vl-flash">qwen3-vl-flash</option>
                        <option value="qwen3-vl-flash-2026-01-22">qwen3-vl-flash-2026-01-22</option>
                        <option value="qwen-vl-plus-latest">qwen-vl-plus-latest</option>
                        <option value="qwen-vl-plus-2025-08-15">qwen-vl-plus-2025-08-15</option>
                        <option value="qwen-vl-max-latest">qwen-vl-max-latest</option>
                        <option value="qwen-vl-max-2025-08-13">qwen-vl-max-2025-08-13</option>
                        <option value="qwen-vl-ocr-latest">qwen-vl-ocr-latest</option>
                        <option value="qwen-vl-ocr-2025-08-28">qwen-vl-ocr-2025-08-28</option>
                      </select>
                      <p className="text-[10px] text-gray-400">qwen3-vl / qwen-vl 系列用于图片/PDF 视觉理解与 OCR 识别</p>
                    </div>
                  </div>

                  {/* Vision test result */}
                  {visionTestFeedback && (
                    <div className={`p-3 rounded-lg border text-xs flex items-start gap-2 leading-relaxed font-semibold ${
                      visionTestStatus === "success" ? "bg-emerald-50 border-emerald-200 text-emerald-800" : "bg-red-50 border-red-200 text-red-800"
                    }`}>
                      {visionTestStatus === "success" ? <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" /> : <AlertTriangle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />}
                      <div>
                        <p className="font-bold">{visionTestStatus === "success" ? "视觉模型连通性正常" : "视觉模型连通性异常"}</p>
                        <p className="font-mono text-[11px] whitespace-pre-wrap">{visionTestFeedback}</p>
                      </div>
                    </div>
                  )}

                  <div className="flex justify-end">
                    <button
                      id="profile-btn-test-vision-connection"
                      type="button"
                      disabled={visionTestStatus === "testing"}
                      onClick={handleTestVisionConnection}
                      className="px-4 py-2 bg-white border border-[#D0D7DE] hover:border-[#8C959F] hover:bg-gray-50 text-xs text-[#24292F] font-bold rounded-lg transition-all flex items-center gap-1.5 shadow-xs cursor-pointer"
                    >
                      {visionTestStatus === "testing" ? (
                        <><span className="inline-block w-3 h-3 border-2 border-purple-600 border-t-transparent rounded-full animate-spin"></span><span>校验中...</span></>
                      ) : (
                        <><Sparkles className="w-4 h-4 text-purple-600" /><span>测试视觉模型连通性</span></>
                      )}
                    </button>
                  </div>
                </div>

                {/* ===== 文生图 API 配置区 ===== */}
                <div className="border border-green-200 rounded-xl p-5 bg-green-50/20 space-y-4">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold text-green-700 bg-green-100 border border-green-200 px-2 py-0.5 rounded">文生图</span>
                    <span className="text-[11px] text-[#57606A]">用于纯文生图、图像编辑等（wan2.7 / wordart 系列，需独立配置 API Key）</span>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* ImageGen API Key */}
                    <div className="space-y-1.5 relative">
                      <label className="text-xs font-bold text-[#57606A] flex items-center justify-between">
                        <span>文生图 API Key (sk-...)</span>
                        <span className="text-[10px] text-gray-400 font-mono">localStorage 加密存储</span>
                      </label>
                      <div className="relative">
                        <input
                          id="profile-input-imagegen-apikey"
                          type={showImageGenApiKey ? "text" : "password"}
                          value={editImageGenApiKey}
                          onChange={(e) => setEditImageGenApiKey(e.target.value)}
                          placeholder="请输入文生图的 sk-************************"
                          className="w-full bg-[#F6F8FA] border border-[#D0D7DE] rounded-lg pl-10 pr-10 py-2 text-xs focus:outline-none focus:bg-white focus:border-[#7C3AED] focus:ring-1 focus:ring-[#7C3AED] font-mono font-semibold"
                        />
                        <Key className="w-4 h-4 text-gray-400 absolute left-3 top-2.5" />
                        <button
                          type="button"
                          onClick={() => setShowImageGenApiKey(!showImageGenApiKey)}
                          className="absolute right-3 top-2.5 text-gray-400 hover:text-[#24292F] cursor-pointer"
                        >
                          {showImageGenApiKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                        </button>
                      </div>
                      <p className="text-[10px] text-gray-400">文生图模型使用独立 API Key，与视觉模型 Key 额度独立</p>
                    </div>

                    {/* ImageGen Model Select */}
                    <div className="space-y-1.5">
                      <label className="text-xs font-bold text-[#57606A] flex justify-between">
                        <span>文生图模型选项</span>
                        <span className="text-[10px] text-gray-400">图像/视频生成</span>
                      </label>
                      <select
                        id="profile-select-imagegen-model"
                        value={editImageGenModel}
                        onChange={(e) => setEditImageGenModel(e.target.value)}
                        className="w-full bg-[#F6F8FA] border border-[#D0D7DE] rounded-lg px-3 py-2 text-xs focus:outline-none focus:bg-white focus:border-[#7C3AED] font-mono font-bold"
                      >
                        <option value="wan2.7-image">wan2.7-image</option>
                        <option value="wan2.7-image-pro">wan2.7-image-pro</option>
                        <option value="wordart-semantic">wordart-semantic</option>
                        <option value="wordart-texture">wordart-texture</option>
                      </select>
                      <p className="text-[10px] text-gray-400">纯图像生成模型（wan2.7 / wordart 系列）</p>
                    </div>
                  </div>

                  {/* ImageGen test result */}
                  {imageGenTestFeedback && (
                    <div className={`p-3 rounded-lg border text-xs flex items-start gap-2 leading-relaxed font-semibold ${
                      imageGenTestStatus === "success" ? "bg-emerald-50 border-emerald-200 text-emerald-800" : "bg-red-50 border-red-200 text-red-800"
                    }`}>
                      {imageGenTestStatus === "success" ? <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" /> : <AlertTriangle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />}
                      <div>
                        <p className="font-bold">{imageGenTestStatus === "success" ? "文生图模型连通性正常" : "文生图模型连通性异常"}</p>
                        <p className="font-mono text-[11px] whitespace-pre-wrap">{imageGenTestFeedback}</p>
                      </div>
                    </div>
                  )}

                  <div className="flex justify-end">
                    <button
                      id="profile-btn-test-imagegen-connection"
                      type="button"
                      disabled={imageGenTestStatus === "testing"}
                      onClick={handleTestImageGenConnection}
                      className="px-4 py-2 bg-white border border-[#D0D7DE] hover:border-[#8C959F] hover:bg-gray-50 text-xs text-[#24292F] font-bold rounded-lg transition-all flex items-center gap-1.5 shadow-xs cursor-pointer"
                    >
                      {imageGenTestStatus === "testing" ? (
                        <><span className="inline-block w-3 h-3 border-2 border-green-600 border-t-transparent rounded-full animate-spin"></span><span>校验中...</span></>
                      ) : (
                        <><Sparkles className="w-4 h-4 text-green-600" /><span>测试文生图连通性</span></>
                      )}
                    </button>
                  </div>
                </div>

                {/* ===== 文生视频 API 配置区 ===== */}
                <div className="border border-orange-200 rounded-xl p-5 bg-orange-50/20 space-y-4">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold text-orange-700 bg-orange-100 border border-orange-200 px-2 py-0.5 rounded">文生视频</span>
                    <span className="text-[11px] text-[#57606A]">用于文生视频、图生视频等（wan2.7 / happyhorse 系列，需独立配置 API Key）</span>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* VideoGen API Key */}
                    <div className="space-y-1.5 relative">
                      <label className="text-xs font-bold text-[#57606A] flex items-center justify-between">
                        <span>文生视频 API Key (sk-...)</span>
                        <span className="text-[10px] text-gray-400 font-mono">localStorage 加密存储</span>
                      </label>
                      <div className="relative">
                        <input
                          id="profile-input-videogen-apikey"
                          type={showVideoGenApiKey ? "text" : "password"}
                          value={editVideoGenApiKey}
                          onChange={(e) => setEditVideoGenApiKey(e.target.value)}
                          placeholder="请输入文生视频的 sk-************************"
                          className="w-full bg-[#F6F8FA] border border-[#D0D7DE] rounded-lg pl-10 pr-10 py-2 text-xs focus:outline-none focus:bg-white focus:border-orange-500 focus:ring-1 focus:ring-orange-200 font-mono font-semibold"
                        />
                        <Key className="w-4 h-4 text-gray-400 absolute left-3 top-2.5" />
                        <button
                          type="button"
                          onClick={() => setShowVideoGenApiKey(!showVideoGenApiKey)}
                          className="absolute right-3 top-2.5 text-gray-400 hover:text-[#24292F] cursor-pointer"
                        >
                          {showVideoGenApiKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                        </button>
                      </div>
                      <p className="text-[10px] text-gray-400">文生视频模型使用独立 API Key，与文生图 Key 额度独立</p>
                    </div>

                    {/* VideoGen Model Select */}
                    <div className="space-y-1.5">
                      <label className="text-xs font-bold text-[#57606A] flex justify-between">
                        <span>文生视频模型选项</span>
                        <span className="text-[10px] text-gray-400">wan2.7 / happyhorse</span>
                      </label>
                      <select
                        id="profile-select-videogen-model"
                        value={editVideoGenModel}
                        onChange={(e) => setEditVideoGenModel(e.target.value)}
                        className="w-full bg-[#F6F8FA] border border-[#D0D7DE] rounded-lg px-3 py-2 text-xs focus:outline-none focus:bg-white focus:border-orange-500 font-mono font-bold"
                      >
                        <option value="wan2.7-t2v-2026-04-25">wan2.7-t2v-2026-04-25</option>
                        <option value="wan2.7-t2v">wan2.7-t2v</option>
                        <option value="wan2.7-i2v-2026-04-25">wan2.7-i2v-2026-04-25</option>
                        <option value="wan2.7-i2v">wan2.7-i2v</option>
                        <option value="wan2.7-r2v">wan2.7-r2v</option>
                        <option value="wan2.7-videoedit">wan2.7-videoedit</option>
                        <option value="happyhorse-1.0-t2v">happyhorse-1.0-t2v</option>
                        <option value="happyhorse-1.0-i2v">happyhorse-1.0-i2v</option>
                        <option value="happyhorse-1.0-r2v">happyhorse-1.0-r2v</option>
                        <option value="happyhorse-1.0-video-edit">happyhorse-1.0-video-edit</option>
                      </select>
                      <p className="text-[10px] text-gray-400">t2v=文生视频；i2v=图生视频；r2v=参考视频</p>
                    </div>
                  </div>

                  {/* VideoGen test result */}
                  {videoGenTestFeedback && (
                    <div className={`p-3 rounded-lg border text-xs flex items-start gap-2 leading-relaxed font-semibold ${
                      videoGenTestStatus === "success" ? "bg-emerald-50 border-emerald-200 text-emerald-800" : "bg-red-50 border-red-200 text-red-800"
                    }`}>
                      {videoGenTestStatus === "success" ? <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" /> : <AlertTriangle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />}
                      <div>
                        <p className="font-bold">{videoGenTestStatus === "success" ? "文生视频模型连通性正常" : "文生视频模型连通性异常"}</p>
                        <p className="font-mono text-[11px] whitespace-pre-wrap">{videoGenTestFeedback}</p>
                      </div>
                    </div>
                  )}

                  <div className="flex justify-end">
                    <button
                      id="profile-btn-test-videogen-connection"
                      type="button"
                      disabled={videoGenTestStatus === "testing"}
                      onClick={handleTestVideoGenConnection}
                      className="px-4 py-2 bg-white border border-[#D0D7DE] hover:border-[#8C959F] hover:bg-gray-50 text-xs text-[#24292F] font-bold rounded-lg transition-all flex items-center gap-1.5 shadow-xs cursor-pointer"
                    >
                      {videoGenTestStatus === "testing" ? (
                        <><span className="inline-block w-3 h-3 border-2 border-orange-500 border-t-transparent rounded-full animate-spin"></span><span>校验中...</span></>
                      ) : (
                        <><Sparkles className="w-4 h-4 text-orange-500" /><span>测试文生视频连通性</span></>
                      )}
                    </button>
                  </div>
                </div>

                {/* Action panel */}
                <div className="flex flex-col sm:flex-row justify-between items-center gap-4 bg-[#F8F9FA] p-4 rounded-xl border border-[#D0D7DE]">
                  <div className="text-left space-y-0.5 w-full sm:w-auto">
                    <p className="text-xs font-bold text-gray-950 flex items-center gap-1.5">
                      <Sparkles className="w-4 h-4 text-amber-500" />
                      <span>保存全部配置</span>
                    </p>
                    <p className="text-[11px] text-[#57606A]">保存后将同时更新文本模型、视觉模型、文生图和文生视频的配置</p>
                  </div>
                  <div className="shrink-0">
                    <button
                      type="button"
                      onClick={handleSaveProfile}
                      className="px-6 py-2.5 bg-[#0969DA] hover:bg-blue-700 text-white text-xs font-bold rounded-lg transition-all flex items-center justify-center gap-1.5 shadow-sm hover:shadow-md cursor-pointer"
                    >
                      <Check className="w-4 h-4" />
                      <span>确认保存本地服务设置</span>
                    </button>
                  </div>
                </div>

              </div>
            </div>
          )}

          {/* SCREEN 7: SIGN IN / LOGIN VIEW */}
          {activeTab === "login" && (
            <div id="login-parent-card" className="max-w-md mx-auto my-12 animate-fade-in space-y-6">
              <div className="bg-white border border-[#D0D7DE] rounded-2xl p-8 shadow-md space-y-6 text-center">
                
                {/* Brand & Heading */}
                <div className="text-center space-y-2 select-none">
                  <div className="inline-flex items-center justify-center w-12 h-12 rounded-2xl bg-gradient-to-tr from-[#0969DA] to-[#388BFD] text-white shadow-sm">
                    <LogIn className="w-6 h-6 animate-pulse" />
                  </div>
                  <h2 className="text-xl font-bold text-gray-950">登录数模算力与算法平台</h2>
                  <p className="text-xs text-[#57606A]">
                    开启智能模型解析、竞赛解题与阿里云百炼极速算力通道
                  </p>
                </div>

                {/* Built-in Credentials Tip Alert Box */}
                <div className="p-4 bg-blue-50/70 border border-blue-200/80 rounded-xl space-y-1 text-xs text-left">
                  <div className="flex items-center gap-1.5 text-blue-900 font-bold">
                    <Info className="w-4 h-4 text-blue-600 shrink-0" />
                    <span>系统内置测试账号免申请</span>
                  </div>
                  <p className="text-[#57606A] leading-relaxed">
                    为了方便您快速登录和体验全部大算力配置及提问功能，请使用下方预设账户：
                  </p>
                  <div className="mt-2 grid grid-cols-2 gap-2 bg-white/80 p-2.5 rounded-lg border border-blue-100 font-mono font-bold text-[#24292F] tracking-wide">
                    <div>账号: <span className="text-blue-700 bg-blue-50 px-1 py-0.5 rounded">123456</span></div>
                    <div>密码: <span className="text-blue-700 bg-blue-50 px-1 py-0.5 rounded">123456</span></div>
                  </div>
                </div>

                {/* Error feedback */}
                {loginError && (
                  <div className="p-3 bg-red-50 border border-red-200 text-red-800 text-xs font-semibold rounded-lg flex items-start gap-1.5 text-left leading-relaxed">
                    <AlertTriangle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
                    <span>{loginError}</span>
                  </div>
                )}

                {/* Login Form */}
                <form onSubmit={handleLoginSubmit} className="space-y-4 text-left">
                  {/* Account Input */}
                  <div className="space-y-1.5">
                    <label className="text-xs font-bold text-[#57606A] block">
                      测试账号 (Username)
                    </label>
                    <div className="relative">
                      <input
                        type="text"
                        required
                        value={loginAccount}
                        onChange={(e) => setLoginAccount(e.target.value)}
                        placeholder="请输入账户 123456"
                        className="w-full bg-[#F6F8FA] border border-[#D0D7DE] rounded-xl pl-9 pr-4 py-2 text-xs focus:outline-none focus:bg-white focus:border-[#0969DA] focus:ring-1 focus:ring-[#0969DA] font-semibold"
                      />
                      <User className="w-4 h-4 text-[#57606A] absolute left-3 top-2.5" />
                    </div>
                  </div>

                  {/* Password Input */}
                  <div className="space-y-1.5">
                    <label className="text-xs font-bold text-[#57606A] block">
                      账户密码 (Password)
                    </label>
                    <div className="relative">
                      <input
                        type="password"
                        required
                        value={loginPassword}
                        onChange={(e) => setLoginPassword(e.target.value)}
                        placeholder="请输入密码 123456"
                        className="w-full bg-[#F6F8FA] border border-[#D0D7DE] rounded-xl pl-9 pr-4 py-2 text-xs focus:outline-none focus:bg-white focus:border-[#0969DA] focus:ring-1 focus:ring-[#0969DA] font-semibold"
                      />
                      <Lock className="w-4 h-4 text-[#57606A] absolute left-3 top-2.5" />
                    </div>
                  </div>

                  {/* Form Action Submit Button */}
                  <button
                    type="submit"
                    className="w-full py-2.5 bg-[#0969DA] hover:bg-blue-700 text-white font-bold rounded-xl text-xs shadow-sm hover:shadow-md transition-all flex items-center justify-center gap-2 cursor-pointer"
                  >
                    <LogIn className="w-4 h-4" />
                    <span>立即登录</span>
                  </button>
                </form>

                <div className="pt-4 border-t border-[#F6F8FA]">
                  <p className="text-xs text-[#57606A]">
                    还没有账号？{" "}
                    <button 
                      onClick={() => { setActiveTab("register"); setLoginError(""); }}
                      className="text-[#0969DA] font-bold hover:underline cursor-pointer"
                    >
                      立即注册
                    </button>
                  </p>
                </div>

              </div>
            </div>
          )}

          {/* SCREEN 8: REGISTER VIEW */}
          {activeTab === "register" && (
            <div id="register-parent-card" className="max-w-md mx-auto my-12 animate-fade-in space-y-6">
              <div className="bg-white border border-[#D0D7DE] rounded-2xl p-8 shadow-md space-y-6 text-center">
                
                <div className="text-center space-y-2">
                  <div className="inline-flex items-center justify-center w-12 h-12 rounded-2xl bg-gradient-to-tr from-[#0969DA] to-[#388BFD] text-white shadow-sm">
                    <User className="w-6 h-6" />
                  </div>
                  <h2 className="text-xl font-bold text-gray-950">创建新账号</h2>
                  <p className="text-xs text-[#57606A]">
                    加入数模社区，获取更多专属算法资源
                  </p>
                </div>

                {regError && (
                  <div className="p-3 bg-red-50 border border-red-200 text-red-800 text-xs font-semibold rounded-lg flex items-start gap-1.5 text-left">
                    <AlertTriangle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
                    <span>{regError}</span>
                  </div>
                )}

                <form onSubmit={handleRegisterSubmit} className="space-y-4 text-left">
                  <div className="space-y-1.5">
                    <label className="text-xs font-bold text-[#57606A] block">用户名 / 手机号</label>
                    <div className="relative">
                      <input
                        type="text"
                        required
                        value={regAccount}
                        onChange={(e) => setRegAccount(e.target.value)}
                        placeholder="请输入您的账号"
                        className="w-full bg-[#F6F8FA] border border-[#D0D7DE] rounded-xl pl-9 pr-4 py-2 text-xs focus:outline-none focus:bg-white focus:border-[#0969DA] focus:ring-1 focus:ring-[#0969DA] font-semibold"
                      />
                      <User className="w-4 h-4 text-[#57606A] absolute left-3 top-2.5" />
                    </div>
                  </div>

                  <div className="space-y-1.5">
                    <label className="text-xs font-bold text-[#57606A] block">设置密码</label>
                    <div className="relative">
                      <input
                        type="password"
                        required
                        value={regPassword}
                        onChange={(e) => setRegPassword(e.target.value)}
                        placeholder="请输入至少6位密码"
                        className="w-full bg-[#F6F8FA] border border-[#D0D7DE] rounded-xl pl-9 pr-4 py-2 text-xs focus:outline-none focus:bg-white focus:border-[#0969DA] focus:ring-1 focus:ring-[#0969DA] font-semibold"
                      />
                      <Lock className="w-4 h-4 text-[#57606A] absolute left-3 top-2.5" />
                    </div>
                  </div>

                  <div className="space-y-1.5">
                    <label className="text-xs font-bold text-[#57606A] block">确认密码</label>
                    <div className="relative">
                      <input
                        type="password"
                        required
                        value={regConfirmPassword}
                        onChange={(e) => setRegConfirmPassword(e.target.value)}
                        placeholder="请再次通过密码验证"
                        className="w-full bg-[#F6F8FA] border border-[#D0D7DE] rounded-xl pl-9 pr-4 py-2 text-xs focus:outline-none focus:bg-white focus:border-[#0969DA] focus:ring-1 focus:ring-[#0969DA] font-semibold"
                      />
                      <Lock className="w-4 h-4 text-[#57606A] absolute left-3 top-2.5" />
                    </div>
                  </div>

                  <button
                    type="submit"
                    className="w-full py-2.5 bg-[#0969DA] hover:bg-blue-700 text-white font-bold rounded-xl text-xs shadow-sm hover:shadow-md transition-all flex items-center justify-center gap-2 cursor-pointer mt-2"
                  >
                    <span>立即注册并登录</span>
                  </button>
                </form>

                <div className="pt-4 border-t border-[#F6F8FA]">
                  <p className="text-xs text-[#57606A]">
                    已有账号？{" "}
                    <button 
                      onClick={() => { setActiveTab("login"); setRegError(""); }}
                      className="text-[#0969DA] font-bold hover:underline cursor-pointer"
                    >
                      返回登录
                    </button>
                  </p>
                </div>
              </div>
            </div>
          )}

        </div>
      </main>

      {/* FLOATING AI ASSISTANT PANEL */}
      <div
        id="ai-assistant-widget"
        className={`fixed bottom-6 right-6 flex flex-col bg-white border rounded-2xl shadow-2xl transition-all duration-300 z-50 ${
          chatOpen
            ? "w-[420px] h-[580px] border-[#0969DA]/60"
            : "w-14 h-14 rounded-full border-[#D0D7DE] overflow-hidden cursor-pointer hover:scale-105"
        }`}
      >
        {chatOpen ? (
          <>
            {/* Header toolbar */}
            <div className="bg-gradient-to-r from-blue-700 to-[#0969DA] px-4 py-3.5 text-white flex justify-between items-center select-none rounded-t-2xl">
              <div className="flex items-center gap-2">
                <div className="relative">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 absolute border-2 border-blue-700 -right-0.5 -bottom-0.5"></span>
                  <div className="w-7 h-7 bg-white/20 rounded-lg flex items-center justify-center font-bold text-xs">
                    AI
                  </div>
                </div>
                <div>
                  <h3 className="text-xs font-semibold tracking-wide">数学建模 AI Copilot</h3>
                  <p className="text-[9px] text-blue-100/80 font-mono">
                    {bailianApiKey ? `Powered by 阿里云百炼 (${bailianModel})` : "Powered by Gemini 1.5 Flash"}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => {
                    setChatMessages([
                      {
                        role: "assistant",
                        content: "对话历史已清空。我有任何可以帮您解决的建模问题吗？",
                        timestamp: new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })
                      }
                    ]);
                  }}
                  className="text-[9px] bg-white/10 hover:bg-white/20 text-white rounded px-2 py-0.5 transition-all text-[9px] font-bold"
                  title="清空对话"
                >
                  清空
                </button>
                <button
                  onClick={() => setChatOpen(false)}
                  className="rounded p-1 hover:bg-white/10 text-white transition-all cursor-pointer text-xs"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Chat message content box */}
            <div className="flex-1 overflow-y-auto p-4 space-y-3.5 bg-[#F6F8FA] flex flex-col">
              {chatMessages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex gap-2 max-w-[85%] ${msg.role === "user" ? "self-end flex-row-reverse" : "self-start"}`}
                >
                  {msg.role !== "user" && (
                    <div className="w-6 h-6 rounded bg-[#0969DA] text-white flex-shrink-0 flex items-center justify-center text-[10px] font-bold">
                      Σ
                    </div>
                  )}
                  {msg.role === "user" && profileAvatar && (
                    <img
                      src={profileAvatar}
                      alt={profileName}
                      className="w-6 h-6 rounded object-cover flex-shrink-0"
                      referrerPolicy="no-referrer"
                    />
                  )}
                  <div className="space-y-1">
                    <div
                      className={`p-3 text-xs leading-relaxed ${
                        msg.role === "user"
                          ? "bg-[#0969DA] text-white rounded-l-xl rounded-br-xl shadow-xs font-semibold"
                          : "bg-white border border-[#D0D7DE] text-[#24292F] rounded-r-xl rounded-bl-xl shadow-xs font-medium"
                      }`}
                    >
                      {msg.role === "assistant" ? (
                        <div className="prose prose-sm max-w-none prose-blue markdown-body">
                          {renderMarkdownWithMath(msg.content)}
                        </div>
                      ) : (
                        <p className="whitespace-pre-wrap">{msg.content}</p>
                      )}
                    </div>
                    <p className={`text-[9px] text-[#57606A] font-mono ${msg.role === "user" ? "text-right" : "text-left"}`}>
                      {msg.timestamp}
                    </p>
                  </div>
                </div>
              ))}

              {chatLoading && (
                <div className="flex gap-2 max-w-[80%] self-start">
                  <div className="w-6 h-6 rounded bg-[#0969DA] text-white flex-shrink-0 flex items-center justify-center text-[10px] font-bold">
                    Σ
                  </div>
                  <div className="p-3 text-xs bg-white border border-[#D0D7DE] text-[#1F2328] rounded-r-xl rounded-bl-xl flex items-center gap-2 shadow-xs">
                    <span className="inline-block w-2 h-2 bg-[#57606A] rounded-full animate-bounce"></span>
                    <span className="inline-block w-2 h-2 bg-[#57606A] rounded-full animate-bounce" style={{ animationDelay: "0.2s" }}></span>
                    <span className="inline-block w-2 h-2 bg-[#57606A] rounded-full animate-bounce" style={{ animationDelay: "0.4s" }}></span>
                    <span className="ml-1 text-[10px] text-[#57606A] font-medium">正在运算推算模型思路中...</span>
                  </div>
                </div>
              )}
              <div ref={chatBottomRef} />
            </div>

            {/* Suggested quick triggers list */}
            <div className="p-2 border-t border-[#D0D7DE] bg-[#F6F8FA] flex flex-wrap gap-1.5 overflow-x-auto shrink-0 select-none">
              <button
                onClick={() => handleSendMessage("怎么挑选最适合的评价类模型？AHP和TOPSIS优缺点对比")}
                className="text-[10px] bg-white hover:bg-blue-50 border border-[#D0D7DE] hover:border-blue-400 text-[#57606A] hover:text-[#0969DA] px-2.5 py-1 rounded-full whitespace-nowrap font-bold shadow-xs cursor-pointer transition-all"
              >
                如何选评价模型？
              </button>
              <button
                onClick={() => handleSendMessage("编写一段求解遗传规划算法TSP的Python代码")}
                className="text-[10px] bg-white hover:bg-blue-50 border border-[#D0D7DE] hover:border-blue-400 text-[#57606A] hover:text-[#0969DA] px-2.5 py-1 rounded-full whitespace-nowrap font-bold shadow-xs cursor-pointer transition-all"
              >
                GA 求解 TSP 算法
              </button>
              <button
                onClick={() => handleSendMessage("数学建模国赛摘要书写黄金公式")}
                className="text-[10px] bg-white hover:bg-blue-50 border border-[#D0D7DE] hover:border-blue-400 text-[#57606A] hover:text-[#0969DA] px-2.5 py-1 rounded-full whitespace-nowrap font-bold shadow-xs cursor-pointer transition-all"
              >
                国赛摘要怎么写？
              </button>
            </div>

            {/* Input send bar footer */}
            <div className="bg-white p-2.5 border-t border-[#D0D7DE] rounded-b-2xl flex items-center gap-2 select-none">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSendMessage();
                }}
                disabled={chatLoading}
                placeholder="在此向 AI Copilot 提报或追问数模问题..."
                className="bg-[#F6F8FA] border border-[#D0D7DE] text-[#24292F] placeholder-[#57606A] text-xs w-full focus:outline-none focus:border-[#0969DA] px-3 py-2 rounded-xl font-medium"
              />
              <button
                disabled={chatLoading || !chatInput.trim()}
                onClick={() => handleSendMessage()}
                className={`p-2 rounded-xl text-white transition-all cursor-pointer ${
                  chatInput.trim() && !chatLoading
                    ? "bg-[#0969DA] hover:bg-blue-700 opacity-100 shadow-sm"
                    : "bg-[#F6F8FA] border border-[#D0D7DE] text-gray-400 cursor-not-allowed opacity-60"
                }`}
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
          </>
        ) : (
          <div
            onClick={() => setChatOpen(true)}
            className="w-full h-full bg-gradient-to-tr from-blue-600 to-teal-500 flex items-center justify-center text-white relative group"
            title="点击展开 AI 数模 Copilot 助手"
          >
            <MessageSquare className="w-5 h-5 absolute transform transition-all group-hover:scale-110" />
            <span className="absolute inline-flex h-3 w-3 rounded-full bg-teal-300 opacity-75 animate-ping top-1 right-1"></span>
          </div>
        )}
      </div>

    </div>
  );
}
