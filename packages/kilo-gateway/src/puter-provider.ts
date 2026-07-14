import { createOpenAICompatible } from "@ai-sdk/openai-compatible"
import type { Provider as SDK } from "ai"
import type { KiloProviderOptions } from "./types.js"
import { getApiKey } from "./auth/token.js"
import { buildKiloHeaders, getDefaultHeaders } from "./headers.js"
import { ANONYMOUS_API_KEY } from "./api/constants.js"

// ============================================================================
// Puter.js Provider
// ============================================================================
//
// Puter.js provides free AI model access through a custom API.
// This provider wraps Puter's /drivers/call endpoint with an
// OpenAI-compatible interface using a custom fetch transformer.
//
// Authentication: Bearer token from Puter.com dashboard
// Models: Claude, GPT-4o, Gemini, DeepSeek, Mistral, and more
// ============================================================================

const PUTER_API_BASE = "https://api.puter.com"

/**
 * Map model IDs to Puter driver names
 */
function getDriverForModel(modelId: string): string {
  const id = modelId.toLowerCase()
  if (id.includes("claude")) return "claude"
  if (id.includes("gpt") || id.includes("o3") || id.includes("o4")) return "openai-completion"
  if (id.includes("gemini")) return "google-ai"
  if (id.includes("mistral")) return "mistral"
  if (id.includes("deepseek")) return "deepseek"
  if (id.includes("llama")) return "openai-completion"
  return "openai-completion"
}

/**
 * Transform OpenAI-compatible messages to Puter format
 */
function transformMessages(messages: any[]): any[] {
  return messages
    .filter((m) => m.role && m.content)
    .map((m) => ({
      role: m.role === "system" ? "system" : m.role === "assistant" ? "assistant" : "user",
      content: String(m.content),
    }))
}

/**
 * Transform Puter response to OpenAI-compatible format
 */
function transformPuterResponse(data: any): any {
  // Check for error
  if (data.error) {
    const errMsg = typeof data.error === "string" ? data.error : data.error?.message || JSON.stringify(data.error)
    throw new Error(`Puter API error: ${errMsg}`)
  }

  // Format: { message: { role, content, tool_calls } }
  if (data.message && typeof data.message === "object") {
    return {
      choices: [
        {
          message: {
            role: data.message.role || "assistant",
            content: data.message.content || "",
            tool_calls: data.message.tool_calls || undefined,
          },
        },
      ],
    }
  }

  // Format: { result: { message: {...} } }
  if (data.result && typeof data.result === "object") {
    if (data.result.message && typeof data.result.message === "object") {
      return {
        choices: [
          {
            message: {
              role: data.result.message.role || "assistant",
              content: data.result.message.content || "",
              tool_calls: data.result.message.tool_calls || undefined,
            },
          },
        ],
      }
    }
    if (typeof data.result.content === "string") {
      return { choices: [{ message: { role: "assistant", content: data.result.content } }] }
    }
    if (typeof data.result.text === "string") {
      return { choices: [{ message: { role: "assistant", content: data.result.text } }] }
    }
  }

  // Format: { choices: [{ message: {...} }] } (OpenAI-style)
  if (Array.isArray(data.choices) && data.choices.length > 0) {
    return data
  }

  // Format: { text: "..." }
  if (typeof data.text === "string") {
    return { choices: [{ message: { role: "assistant", content: data.text } }] }
  }

  // Format: { content: "..." }
  if (typeof data.content === "string") {
    return { choices: [{ message: { role: "assistant", content: data.content } }] }
  }

  // Format: driver call response { success: true, result: ... }
  if (data.success && data.result) {
    return transformPuterResponse(data.result)
  }

  // Fallback: return raw data as text
  return { choices: [{ message: { role: "assistant", content: JSON.stringify(data) } }] }
}

/**
 * Create a custom fetch that transforms between OpenAI-compatible and Puter formats
 */
function createPuterFetch(token: string) {
  return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url

    // Only intercept Puter API calls
    if (!url.includes("api.puter.com")) {
      return fetch(input, init)
    }

    // Parse the OpenAI-compatible request body
    let requestBody: any = {}
    if (init?.body) {
      if (typeof init.body === "string") {
        try {
          requestBody = JSON.parse(init.body)
        } catch {
          // Not JSON, pass through
          return fetch(input, init)
        }
      } else if (init.body instanceof ReadableStream) {
        // Read the stream
        const reader = init.body.getReader()
        const chunks: Uint8Array[] = []
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          chunks.push(value)
        }
        const text = new TextDecoder().decode(new Uint8Array(chunks.reduce((acc, chunk) => [...acc, ...chunk], [] as number[])))
        try {
          requestBody = JSON.parse(text)
        } catch {
          return fetch(input, init)
        }
      }
    }

    const model = requestBody.model || "gpt-4o-mini"
    const messages = requestBody.messages || []
    const tools = requestBody.tools || []

    // Transform to Puter format
    const driver = getDriverForModel(model)
    const puterBody = {
      interface: "puter-chat-completion",
      driver: driver,
      method: "complete",
      args: {
        messages: transformMessages(messages),
        model: model,
        ...(tools.length > 0 && { tools }),
        ...(requestBody.max_tokens && { max_tokens: requestBody.max_tokens }),
      },
    }

    // Make request to Puter API
    const puterUrl = `${PUTER_API_BASE}/drivers/call`
    const response = await fetch(puterUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
        Origin: "https://puter.com",
      },
      body: JSON.stringify(puterBody),
    })

    if (!response.ok) {
      const errorText = await response.text()
      let errorMessage = `HTTP ${response.status}: ${errorText}`
      if (response.status === 401) {
        errorMessage = "Unauthorized (401). Your Puter token is expired or invalid. Get a new one from puter.com."
      } else if (response.status === 403) {
        errorMessage = "Forbidden (403). Access denied."
      } else if (response.status === 404) {
        errorMessage = `Endpoint not found (404): /drivers/call`
      }
      return new Response(JSON.stringify({ error: { message: errorMessage } }), {
        status: response.status,
        headers: { "Content-Type": "application/json" },
      })
    }

    const responseText = await response.text()
    let data: any
    try {
      data = JSON.parse(responseText)
    } catch {
      data = { text: responseText }
    }

    // Transform Puter response to OpenAI-compatible format
    const transformed = transformPuterResponse(data)

    return new Response(JSON.stringify(transformed), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })
  }
}

/**
 * Create a Puter.js provider instance
 *
 * @example
 * ```typescript
 * const provider = createPuter({
 *   puterToken: "your-puter-token-here"
 * })
 *
 * const model = provider.languageModel("claude-sonnet-4-6")
 * ```
 */
export function createPuter(options: KiloProviderOptions = {}): SDK {
  // Get API key from options or environment
  const apiKey = getApiKey(options)
  const puterToken = options.puterToken || apiKey || ""

  if (!puterToken) {
    console.warn("[PuterProvider] No token provided. Set puterToken option or PUTER_TOKEN env var.")
  }

  // Merge custom headers with defaults
  const customHeaders = {
    ...getDefaultHeaders(),
    ...buildKiloHeaders(undefined, {
      kilocodeOrganizationId: options.kilocodeOrganizationId,
      kilocodeTesterWarningsDisabledUntil: undefined,
    }),
    ...options.headers,
  }

  // Create custom fetch wrapper
  const originalFetch = options.fetch ?? fetch
  const wrappedFetch = async (input: string | URL | Request, init?: RequestInit) => {
    const headers = new Headers(customHeaders)
    new Headers(init?.headers).forEach((value, key) => {
      headers.set(key, value)
    })

    // Use the Puter-specific fetch transformer
    const puterFetch = createPuterFetch(puterToken)

    return puterFetch(input, {
      ...init,
      headers,
    })
  }

  const sdkOptions = {
    baseURL: PUTER_API_BASE,
    apiKey: puterToken || ANONYMOUS_API_KEY,
    headers: customHeaders,
    fetch: wrappedFetch as typeof fetch,
  }

  const openaiCompatible = createOpenAICompatible({ ...sdkOptions, name: "puter" })

  return {
    languageModel(modelId: string) {
      return openaiCompatible.languageModel(modelId)
    },
    embeddingModel(modelId: string) {
      return openaiCompatible.textEmbeddingModel?.(modelId) || openaiCompatible.languageModel(modelId)
    },
    rerankingModel(_modelId: string): never {
      throw new Error(`Reranking model not supported: ${_modelId}`)
    },
    imageModel(_modelId: string) {
      return openaiCompatible.imageModel?.(_modelId) || openaiCompatible.languageModel(_modelId)
    },
  }
}

/**
 * Default Puter models available through the free tier
 */
export const PUTER_DEFAULT_MODELS = [
  // Claude (Anthropic)
  { id: "claude-opus-4-7", name: "Claude Opus 4.7", provider: "anthropic" },
  { id: "claude-opus-4-6-fast", name: "Claude Opus 4.6 Fast", provider: "anthropic" },
  { id: "claude-sonnet-4-6", name: "Claude Sonnet 4.6", provider: "anthropic" },
  { id: "claude-opus-4-6", name: "Claude Opus 4.6", provider: "anthropic" },
  { id: "claude-opus-4-5", name: "Claude Opus 4.5", provider: "anthropic" },
  { id: "claude-haiku-4-5", name: "Claude Haiku 4.5", provider: "anthropic" },
  { id: "claude-sonnet-4-5", name: "Claude Sonnet 4.5", provider: "anthropic" },
  { id: "claude-opus-4-1", name: "Claude Opus 4.1", provider: "anthropic" },
  { id: "claude-opus-4", name: "Claude Opus 4", provider: "anthropic" },
  { id: "claude-sonnet-4", name: "Claude Sonnet 4", provider: "anthropic" },
  // OpenAI
  { id: "gpt-4o-mini", name: "GPT-4o Mini", provider: "openai" },
  { id: "gpt-4o", name: "GPT-4o", provider: "openai" },
  // Google
  { id: "gemini-2.5-pro", name: "Gemini 2.5 Pro", provider: "google" },
  { id: "gemini-2.5-flash", name: "Gemini 2.5 Flash", provider: "google" },
  // DeepSeek
  { id: "deepseek-chat", name: "DeepSeek Chat", provider: "deepseek" },
  { id: "deepseek-reasoner", name: "DeepSeek Reasoner", provider: "deepseek" },
  // Mistral
  { id: "mistral-large-latest", name: "Mistral Large", provider: "mistral" },
] as const
