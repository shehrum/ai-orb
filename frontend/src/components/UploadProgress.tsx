import { CheckCircle2, ChevronDown, ChevronUp, FileText, Loader2, XCircle } from "lucide-react";
import { useState } from "react";
import type { UploadTask } from "../hooks/use-documents";

interface UploadProgressProps {
	tasks: UploadTask[];
	onDismiss: () => void;
}

function statusIcon(status: UploadTask["status"]) {
	switch (status) {
		case "pending":
			return <Loader2 className="h-3.5 w-3.5 animate-spin text-neutral-400" />;
		case "uploading":
			return <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-500" />;
		case "processing":
			return <Loader2 className="h-3.5 w-3.5 animate-spin text-amber-500" />;
		case "done":
			return <CheckCircle2 className="h-3.5 w-3.5 text-green-500" />;
		case "error":
			return <XCircle className="h-3.5 w-3.5 text-red-500" />;
	}
}

function statusLabel(status: UploadTask["status"]) {
	switch (status) {
		case "pending":
			return "Waiting...";
		case "uploading":
			return "Uploading...";
		case "processing":
			return "Processing (chunking, embedding)...";
		case "done":
			return "Ready";
		case "error":
			return "Failed";
	}
}

export function UploadProgress({ tasks, onDismiss }: UploadProgressProps) {
	const [expanded, setExpanded] = useState(true);

	if (tasks.length === 0) return null;

	const done = tasks.filter((t) => t.status === "done").length;
	const hasErrors = tasks.some((t) => t.status === "error");
	const allDone = tasks.every((t) => t.status === "done" || t.status === "error");
	const activeTask = tasks.find(
		(t) => t.status === "uploading" || t.status === "processing",
	);

	const summaryText = allDone
		? hasErrors
			? `${done}/${tasks.length} uploaded`
			: `${tasks.length} document${tasks.length > 1 ? "s" : ""} ready`
		: activeTask
			? statusLabel(activeTask.status)
			: `Uploading ${tasks.length} document${tasks.length > 1 ? "s" : ""}...`;

	return (
		<div className="mx-4 mt-2 overflow-hidden rounded-lg border border-neutral-200 bg-white text-sm shadow-sm">
			{/* Summary bar — always visible */}
			<button
				type="button"
				onClick={() => setExpanded(!expanded)}
				className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-neutral-50"
			>
				{allDone ? (
					hasErrors ? (
						<XCircle className="h-4 w-4 flex-shrink-0 text-red-500" />
					) : (
						<CheckCircle2 className="h-4 w-4 flex-shrink-0 text-green-500" />
					)
				) : (
					<Loader2 className="h-4 w-4 flex-shrink-0 animate-spin text-blue-500" />
				)}
				<span className="flex-1 text-neutral-700">{summaryText}</span>
				{expanded ? (
					<ChevronUp className="h-3.5 w-3.5 text-neutral-400" />
				) : (
					<ChevronDown className="h-3.5 w-3.5 text-neutral-400" />
				)}
			</button>

			{/* Expandable detail log */}
			{expanded && (
				<div className="border-t border-neutral-100 px-3 py-1.5">
					{tasks.map((task) => (
						<div key={task.id} className="flex items-center gap-2 py-1">
							{statusIcon(task.status)}
							<FileText className="h-3.5 w-3.5 flex-shrink-0 text-neutral-400" />
							<span className="min-w-0 flex-1 truncate text-xs text-neutral-600">
								{task.filename}
							</span>
							<span className="flex-shrink-0 text-xs text-neutral-400">
								{task.error || statusLabel(task.status)}
							</span>
						</div>
					))}
					{allDone && (
						<button
							type="button"
							onClick={onDismiss}
							className="mt-1 w-full text-center text-xs text-neutral-400 hover:text-neutral-600"
						>
							Dismiss
						</button>
					)}
				</div>
			)}
		</div>
	);
}
