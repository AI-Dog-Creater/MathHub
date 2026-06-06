export interface ModelDoc {
  id: string; // e.g., 'linear-programming'
  name: string; // e.g., '线性规划'
  category: "preprocess" | "reduction" | "prediction" | "evaluation" | "machine-learning" | "stats-analysis" | "econometrics" | "programming-solvers";
  categoryName: string; // e.g., '基础建模'
  summary: string; // 📌 模型简介
  principles: string; // 🧠 基本原理（含有公式说明）
  scenarios: string[]; // 📊 适用场景
  limitations: string[]; // ⚠️ 使用条件与限制
  caseStudy: {
    title: string;
    description: string;
    solution: string;
  }; // 🧪 示例案例
  code: {
    python?: string;
    matlab?: string;
  }; // 💻 示例代码
  resources: {
    title: string;
    url: string;
    type: "paper" | "video" | "book" | "link";
  }[]; // 📎 延伸学习资源
}

export interface Competition {
  id: string;
  name: string;
  alias: string; // e.g. MCM/ICM
  logoChar: string; // Letter or emoji
  month: number; // 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, etc. (1-12)
  timeline: {
    signup: string;
    contest: string;
    results: string;
  }; // 📅 时间线
  description: string; // 🏆 比赛简介
  requirements: string; // 👥 参赛要求
  valueAnalysis: string; // 📈 含金量分析
  pastPapersUrl: string; // 📚 历年题目入口
  targetAudience: "新手" | "进阶" | "全部人员" | "研究生"; // 🎯 适合人群
  difficulty: number; // 1-5 stars
}

export interface PathStep {
  title: string;
  duration: string;
  description: string;
  skills: string[];
  recommendedModels: string[];
}

export interface LearningPath {
  id: string;
  name: string;
  description: string;
  steps: PathStep[];
}

export interface Message {
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  imageUrl?: string; // URL of generated image/video
  mediaType?: "image" | "video"; // media type for the imageUrl
}

export interface AgentSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

export interface AssessmentQuestion {
  id: number;
  question: string;
  options: string[];
  dimension: "abstraction" | "selection" | "foundation" | "algorithm" | "programming" | "expression";
  weight: number;
  correctOptionIndex: number;
}
