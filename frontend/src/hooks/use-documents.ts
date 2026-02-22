import { useCallback, useEffect, useState } from "react";
import * as api from "../lib/api";
import type { Document } from "../types";

export interface UploadTask {
	id: string;
	filename: string;
	status: "pending" | "uploading" | "processing" | "done" | "error";
	error?: string;
}

export function useDocuments(conversationId: string | null) {
	const [documents, setDocuments] = useState<Document[]>([]);
	const [activeDocumentId, setActiveDocumentId] = useState<string | null>(null);
	const [targetPage, setTargetPage] = useState<number | null>(null);
	const [uploadTasks, setUploadTasks] = useState<UploadTask[]>([]);
	const [error, setError] = useState<string | null>(null);

	const uploading = uploadTasks.some(
		(t) => t.status === "pending" || t.status === "uploading" || t.status === "processing",
	);

	const refresh = useCallback(async () => {
		if (!conversationId) {
			setDocuments([]);
			setActiveDocumentId(null);
			return;
		}
		try {
			setError(null);
			const detail = await api.fetchConversation(conversationId);
			const docs = detail.documents ?? [];
			setDocuments(docs);
			if (docs.length > 0 && !activeDocumentId) {
				setActiveDocumentId(docs[0].id);
			}
		} catch (err) {
			setError(err instanceof Error ? err.message : "Failed to load documents");
		}
	}, [conversationId, activeDocumentId]);

	useEffect(() => {
		refresh();
	}, [refresh]);

	const updateTask = useCallback((id: string, update: Partial<UploadTask>) => {
		setUploadTasks((prev) =>
			prev.map((t) => (t.id === id ? { ...t, ...update } : t)),
		);
	}, []);

	const uploadSingleFile = useCallback(
		async (file: File, taskId: string) => {
			if (!conversationId) return null;
			try {
				updateTask(taskId, { status: "uploading" });
				const doc = await api.uploadDocument(conversationId, file);
				// Upload done, now it's processing (RAG pipeline runs server-side)
				updateTask(taskId, { status: "processing" });
				setDocuments((prev) => [...prev, doc]);
				setActiveDocumentId(doc.id);
				// Mark done after a brief delay (processing happens server-side during upload)
				updateTask(taskId, { status: "done" });
				return doc;
			} catch (err) {
				updateTask(taskId, {
					status: "error",
					error: err instanceof Error ? err.message : "Upload failed",
				});
				return null;
			}
		},
		[conversationId, updateTask],
	);

	const uploadFiles = useCallback(
		async (files: File[]) => {
			if (!conversationId || files.length === 0) return;

			// Create task entries for all files
			const tasks: UploadTask[] = files.map((f, i) => ({
				id: `upload-${Date.now()}-${i}`,
				filename: f.name,
				status: "pending" as const,
			}));
			setUploadTasks(tasks);
			setError(null);

			// Upload sequentially (server processes each one)
			for (let i = 0; i < files.length; i++) {
				await uploadSingleFile(files[i], tasks[i].id);
			}
		},
		[conversationId, uploadSingleFile],
	);

	// Legacy single-file upload (kept for compatibility)
	const upload = useCallback(
		async (file: File) => {
			await uploadFiles([file]);
		},
		[uploadFiles],
	);

	const clearUploadTasks = useCallback(() => {
		setUploadTasks((prev) => prev.filter((t) => t.status !== "done" && t.status !== "error"));
	}, []);

	const navigateTo = useCallback((docId: string, page?: number) => {
		setActiveDocumentId(docId);
		if (page) {
			setTargetPage(page);
		}
	}, []);

	const clearTargetPage = useCallback(() => {
		setTargetPage(null);
	}, []);

	const activeDocument = documents.find((d) => d.id === activeDocumentId) ?? null;

	return {
		documents,
		activeDocument,
		activeDocumentId,
		targetPage,
		uploading,
		uploadTasks,
		error,
		upload,
		uploadFiles,
		refresh,
		navigateTo,
		clearTargetPage,
		clearUploadTasks,
		setActiveDocumentId,
	};
}
