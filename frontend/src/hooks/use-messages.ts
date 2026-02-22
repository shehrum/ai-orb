import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "../lib/api";
import type { Citation, Message } from "../types";

export function useMessages(conversationId: string | null) {
	const [messages, setMessages] = useState<Message[]>([]);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [streaming, setStreaming] = useState(false);
	const [streamingContent, setStreamingContent] = useState("");
	const [statusMessage, setStatusMessage] = useState<string | null>(null);
	const [searchSteps, setSearchSteps] = useState<string[]>([]);
	const [citations, setCitations] = useState<Citation[]>([]);
	const [docLabelMap, setDocLabelMap] = useState<Record<string, string>>({});
	const abortRef = useRef<AbortController | null>(null);

	const refresh = useCallback(async () => {
		if (!conversationId) {
			setMessages([]);
			return;
		}
		try {
			setLoading(true);
			setError(null);
			const data = await api.fetchMessages(conversationId);
			setMessages(data);
		} catch (err) {
			setError(err instanceof Error ? err.message : "Failed to load messages");
		} finally {
			setLoading(false);
		}
	}, [conversationId]);

	useEffect(() => {
		refresh();
		return () => {
			if (abortRef.current) {
				abortRef.current.abort();
			}
		};
	}, [refresh]);

	const send = useCallback(
		async (content: string) => {
			if (!conversationId || streaming) return;

			const userMessage: Message = {
				id: `temp-${Date.now()}`,
				conversation_id: conversationId,
				role: "user",
				content,
				sources_cited: 0,
				created_at: new Date().toISOString(),
			};

			setMessages((prev) => [...prev, userMessage]);
			setStreaming(true);
			setStreamingContent("");
			setStatusMessage(null);
			setSearchSteps([]);
			setCitations([]);
			setError(null);

			try {
				const response = await api.sendMessage(conversationId, content);

				if (!response.body) {
					throw new Error("No response body");
				}

				const reader = response.body.getReader();
				const decoder = new TextDecoder();
				let accumulated = "";
				let buffer = "";

				while (true) {
					const { done, value } = await reader.read();
					if (done) break;

					buffer += decoder.decode(value, { stream: true });
					const lines = buffer.split("\n");
					buffer = lines.pop() ?? "";

					for (const line of lines) {
						const trimmed = line.trim();
						if (!trimmed || !trimmed.startsWith("data: ")) continue;

						const data = trimmed.slice(6);
						if (data === "[DONE]") continue;

						try {
							const parsed = JSON.parse(data) as {
								type?: string;
								content?: string;
								delta?: string;
								message?: Message;
								citations?: Citation[];
								doc_label_map?: Record<string, string>;
							};

							if (parsed.type === "status" && parsed.content) {
								setStatusMessage(parsed.content);
								setSearchSteps((prev) => [...prev, parsed.content as string]);
							} else if (parsed.type === "delta" && parsed.delta) {
								accumulated += parsed.delta;
								setStreamingContent(accumulated);
								setStatusMessage(null);
							} else if (parsed.type === "content" && parsed.content) {
								accumulated += parsed.content;
								setStreamingContent(accumulated);
								setStatusMessage(null);
							} else if (parsed.type === "message" && parsed.message) {
								setMessages((prev) => [...prev, parsed.message as Message]);
								accumulated = "";
							} else if (parsed.type === "done") {
								if (parsed.citations) {
									setCitations(parsed.citations);
								}
								if (parsed.doc_label_map) {
									setDocLabelMap(parsed.doc_label_map);
								}
							} else if (parsed.content && !parsed.type) {
								accumulated += parsed.content;
								setStreamingContent(accumulated);
							}
						} catch {
							// Skip invalid JSON lines
						}
					}
				}

				if (accumulated) {
					const assistantMessage: Message = {
						id: `stream-${Date.now()}`,
						conversation_id: conversationId,
						role: "assistant",
						content: accumulated,
						sources_cited: 0,
						created_at: new Date().toISOString(),
					};
					setMessages((prev) => [...prev, assistantMessage]);
				}

				const freshMessages = await api.fetchMessages(conversationId);
				setMessages(freshMessages);
			} catch (err) {
				if (err instanceof DOMException && err.name === "AbortError") return;
				setError(err instanceof Error ? err.message : "Failed to send message");
			} finally {
				setStreaming(false);
				setStreamingContent("");
				setStatusMessage(null);
				setSearchSteps([]);
			}
		},
		[conversationId, streaming],
	);

	return {
		messages,
		loading,
		error,
		streaming,
		streamingContent,
		statusMessage,
		searchSteps,
		citations,
		docLabelMap,
		send,
		refresh,
	};
}
