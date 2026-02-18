import { useCallback, useEffect, useState } from "react";
import * as api from "../lib/api";
import type { Conversation } from "../types";

export function useConversations() {
	const [conversations, setConversations] = useState<Conversation[]>([]);
	const [selectedId, setSelectedId] = useState<string | null>(null);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);

	const refresh = useCallback(async () => {
		try {
			setError(null);
			const data = await api.fetchConversations();
			setConversations(data);
		} catch (err) {
			setError(
				err instanceof Error ? err.message : "Failed to load conversations",
			);
		} finally {
			setLoading(false);
		}
	}, []);

	useEffect(() => {
		refresh();
	}, [refresh]);

	const create = useCallback(async () => {
		try {
			setError(null);
			const conversation = await api.createConversation();
			setConversations((prev) => [conversation, ...prev]);
			setSelectedId(conversation.id);
			return conversation;
		} catch (err) {
			setError(
				err instanceof Error ? err.message : "Failed to create conversation",
			);
			return null;
		}
	}, []);

	const select = useCallback((id: string | null) => {
		setSelectedId(id);
	}, []);

	const remove = useCallback(
		async (id: string) => {
			try {
				setError(null);
				await api.deleteConversation(id);
				setConversations((prev) => prev.filter((c) => c.id !== id));
				if (selectedId === id) {
					setSelectedId(null);
				}
			} catch (err) {
				setError(
					err instanceof Error ? err.message : "Failed to delete conversation",
				);
			}
		},
		[selectedId],
	);

	const selected = conversations.find((c) => c.id === selectedId) ?? null;

	return {
		conversations,
		selected,
		selectedId,
		loading,
		error,
		create,
		select,
		remove,
		refresh,
	};
}
