import type { Plugin } from "@kilocode/plugin"

/**
 * Puter.js Authentication Plugin
 *
 * Provides API key authentication for Puter.js provider.
 * Users can authenticate by pasting their Puter auth token
 * from the Puter.com dashboard.
 */
export const PuterAuthPlugin: Plugin = async (ctx) => {
  return {
    auth: {
      provider: "puter",
      async loader(getAuth, _provider) {
        // Get the stored auth
        const auth = await getAuth()
        if (!auth) return {}

        // For API auth, the key is the token directly
        if (auth.type === "api") {
          return {
            puterToken: auth.key,
          }
        }

        return {}
      },
      methods: [
        {
          type: "api",
          label: "Puter.js (API Token)",
          prompts: [
            {
              type: "text",
              key: "puterToken",
              message: "Enter your Puter auth token",
              placeholder: "Paste your Puter token here...",
              validate: (value: string) => {
                if (!value || value.trim().length === 0) {
                  return "Token is required"
                }
                if (value.trim().length < 10) {
                  return "Token appears to be too short"
                }
                return undefined
              },
            },
          ],
          async authorize(inputs?: Record<string, string>) {
            const token = inputs?.puterToken?.trim()
            if (!token) {
              return { type: "failed" as const }
            }

            return {
              type: "success" as const,
              key: token,
              provider: "puter",
              metadata: {
                provider: "puter",
              },
            }
          },
        },
      ],
    },
  }
}

export default PuterAuthPlugin
