// Imports

import { ChatWindowMessage } from "@/schema/ChatWindowMessage";
import { Voy as VoyClient } from "voy-search";

import {
  Annotation,
  MessagesAnnotation,
  StateGraph,
} from "@langchain/langgraph/web";

import {
  WebPDFLoader,
  HuggingFaceTransformersEmbeddings,
  VoyVectorStore,
  ChatWebLLM,
  ChromeAI,
} from "@langchain/community";

import {
  ChatPromptTemplate,
  PromptTemplate,
} from "@langchain/core/prompts";

import {
  BaseMessage,
  Document,
  RunnableConfig,
} from "@langchain/core";

import {
  BaseChatModel,
  BaseLLM,
  LanguageModelLike,
} from "@langchain/core/language_models";

import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { LangChainTracer } from "@langchain/core/tracers/tracer_langchain";
import { Client } from "langsmith";
import { ChatOllama } from "@langchain/ollama";

// Embeddings and Vector Store Setup

const embeddings = new HuggingFaceTransformersEmbeddings({
  modelName: "Xenova/all-MiniLM-L6-v2",
});

const voyClient = new VoyClient();
const vectorstore = new VoyVectorStore(voyClient, embeddings);

// Prompt Templates

const SYSTEM_TEMPLATES = {
  ollama: `You are an experienced researcher...<context>\n{context}\n<context/>`,
  webllm: `You are an experienced researcher...`,
  chrome_ai: `You are an AI assistant...<context>\n{context}\n</context>\n\n{conversation_turns}\nassistant: `,
};

// PDF Embedding

const embedPDF = async (pdfBlob: Blob) => {
  const loader = new WebPDFLoader(pdfBlob, { parsedItemSeparator: " " });
  const docs = await loader.load();
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
  });
  const splitDocs = await splitter.splitDocuments(docs);
  self.postMessage({ type: "log", data: splitDocs });
  await vectorstore.addDocuments(splitDocs);
};

// Main RAG Pipeline

const generateRAGResponse = async (
  messages: ChatWindowMessage[],
  {
    model,
    modelProvider,
    devModeTracer,
  }: {
    model: LanguageModelLike;
    modelProvider: "ollama" | "webllm" | "chrome_ai";
    devModeTracer?: LangChainTracer;
  },
) => {
  const AnnotationSpec = Annotation.Root({
    ...MessagesAnnotation.spec,
    rephrasedQuestion: Annotation<string>,
    sourceDocuments: Annotation<Document[]>,
  });

  // Step 1: Rephrase
  const rephraseQuestion = async (
    state: typeof AnnotationSpec.State,
    config: RunnableConfig,
  ) => {
    const input = state.messages.at(-1)?.content as string;

    let prompt;
    if (modelProvider === "chrome_ai") {
      const template = PromptTemplate.fromTemplate(SYSTEM_TEMPLATES.chrome_ai);
      const turns = state.messages
        .map((msg) => `${msg.getType() === "ai" ? "assistant" : "user"}: ${msg.content}`)
        .join("\n");
      prompt = await template.invoke({ input, conversation_turns: turns }, config);
    } else {
      const userPrompt =
        modelProvider === "ollama"
          ? "Given the above conversation, generate a natural language search query..."
          : "Given the above conversation, rephrase the following question...";
      const template = ChatPromptTemplate.fromMessages([
        ["placeholder", "{messages}"],
        ["user", userPrompt],
      ]);
      prompt = await template.invoke({ messages: state.messages, input }, config);
    }

    const response = await model.invoke(prompt, config);
    return {
      rephrasedQuestion: typeof response === "string" ? response : response.content,
    };
  };

  // Step 2: Retrieve
  const retrieveSourceDocuments = async (
    state: typeof AnnotationSpec.State,
    config: RunnableConfig,
  ) => {
    const query = state.rephrasedQuestion ?? (state.messages.at(-1)?.content as string);
    const docs = await vectorstore.asRetriever().invoke(query, config);
    return { sourceDocuments: docs };
  };

  // Step 3: Respond
  const generateResponse = async (
    state: typeof AnnotationSpec.State,
    config: RunnableConfig,
  ) => {
    const context = state.sourceDocuments
      .map((doc) => `<doc>\n${doc.pageContent}\n</doc>`)
      .join("\n\n");

    let prompt;
    if (modelProvider === "chrome_ai") {
      const turns = state.messages
        .map((msg) => `${msg.getType() === "ai" ? "assistant" : "user"}: ${msg.content}`)
        .join("\n");
      prompt = await PromptTemplate.fromTemplate(SYSTEM_TEMPLATES.chrome_ai).invoke(
        { context, conversation_turns: turns },
        config,
      );
    } else {
      const baseMessages = [
        ["system", SYSTEM_TEMPLATES[modelProvider]],
        ["placeholder", "{messages}"],
      ];
      if (modelProvider === "webllm") {
        baseMessages.splice(1, 0,
          ["user", "When responding, use the following documents...\n<context>\n{context}\n</context>"],
          ["assistant", "Understood! I will use the context."]
        );
      }
      prompt = await ChatPromptTemplate.fromMessages(baseMessages).invoke(
        { context, messages: state.messages },
        config,
      );
    }

    const response = await model.withConfig({ tags: ["response_generator"] }).invoke(prompt, config);
    return { messages: [{ role: "assistant", content: typeof response === "string" ? response : response.content }] };
  };

  // LangGraph Execution
  const graph = new StateGraph(AnnotationSpec)
    .addNode("rephraseQuestion", rephraseQuestion)
    .addNode("retrieveSourceDocuments", retrieveSourceDocuments)
    .addNode("generateResponse", generateResponse)
    .addConditionalEdges("__start__", async (state) =>
      state.messages.length > 1 ? "rephraseQuestion" : "retrieveSourceDocuments",
    )
    .addEdge("rephraseQuestion", "retrieveSourceDocuments")
    .addEdge("retrieveSourceDocuments", "generateResponse")
    .compile();

  const eventStream = await graph.streamEvents(
    { messages },
    { version: "v2", callbacks: devModeTracer ? [devModeTracer] : [] },
  );

  for await (const { event, data, tags } of eventStream) {
    if (tags?.includes("response_generator")) {
      const chunk = event === "on_chat_model_stream" ? data.chunk.content : data.chunk.text;
      self.postMessage({ type: "chunk", data: chunk });
    }
  }

  self.postMessage({ type: "complete", data: "OK" });
};

// Worker Message Listener

self.addEventListener("message", async ({ data }) => {
  self.postMessage({ type: "log", data: `Received data!` });

  try {
    let tracer;
    if (typeof data.DEV_LANGCHAIN_TRACING === "object") {
      tracer = new LangChainTracer({
        projectName: data.DEV_LANGCHAIN_TRACING.LANGCHAIN_PROJECT,
        client: new Client({ apiKey: data.DEV_LANGCHAIN_TRACING.LANGCHAIN_API_KEY }),
      });
    }

    if (data.pdf) {
      await embedPDF(data.pdf);
    } else {
      let model: LanguageModelLike;
      const provider = data.modelProvider;
      if (provider === "webllm") {
        const webllm = new ChatWebLLM(data.modelConfig);
        await webllm.initialize((event) => self.postMessage({ type: "init_progress", data: event }));
        model = webllm.bind({ stop: ["\nInstruct:", "Instruct:", "<hr>", "\n<hr>"] });
      } else if (provider === "chrome_ai") {
        model = new ChromeAI(data.modelConfig);
      } else {
        model = new ChatOllama(data.modelConfig);
      }

      await generateRAGResponse(data.messages, {
        model,
        modelProvider: provider,
        devModeTracer: tracer,
      });
    }
  } catch (e: any) {
    self.postMessage({
      type: "error",
      error:
        data.modelProvider === "ollama"
          ? `${e.message}. Make sure you are running Ollama.`
          : `${e.message}. Make sure your browser supports WebLLM/WebGPU.`,
    });
    throw e;
  }

  self.postMessage({ type: "complete", data: "OK" });
});
