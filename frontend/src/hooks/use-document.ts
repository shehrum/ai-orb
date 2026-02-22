import { useCallback, useEffect, useState } from "react";
import * as api from "../lib/api";
import type { Document } from "../types";

export function useDocument(conversationId: string | null) {
	const [document, setDocument] = useState<Document | null>(null);
	const [uploading, setUploading] = useState(false);
	const [error, setError] = useState<string | null>(null);

	const refresh = useCallback(async () => {
		if (!conversationId) {
			setDocument(null);
			return;
		}
		try {
			setError(null);
			const detail = await api.fetchConversation(conversationId);
			const docs = detail.documents ?? [];
			setDocument(docs[0] ?? null);
		} catch (err) {
			setError(err instanceof Error ? err.message : "Failed to load document");
		}
	}, [conversationId]);

	useEffect(() => {
		refresh();
	}, [refresh]);

	const upload = useCallback(
		async (file: File) => {
			if (!conversationId) return null;
			try {
				setUploading(true);
				setError(null);
				const doc = await api.uploadDocument(conversationId, file);
				setDocument(doc);
				return doc;
			} catch (err) {
				setError(
					err instanceof Error ? err.message : "Failed to upload document",
				);
				return null;
			} finally {
				setUploading(false);
			}
		},
		[conversationId],
	);

	return {
		document,
		uploading,
		error,
		upload,
		refresh,
	};
}
