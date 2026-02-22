import { Loader2 } from "lucide-react";
import { useEffect, useRef } from "react";
import type { UploadTask } from "../hooks/use-documents";
import type { Message } from "../types";
import { ChatInput } from "./ChatInput";
import { EmptyState } from "./EmptyState";
import { MessageBubble, StreamingBubble } from "./MessageBubble";
import { UploadProgress } from "./UploadProgress";

interface ChatWindowProps {
	messages: Message[];
	loading: boolean;
	error: string | null;
	streaming: boolean;
	streamingContent: string;
	statusMessage?: string | null;
	searchSteps?: string[];
	hasDocuments: boolean;
	conversationId: string | null;
	uploadTasks: UploadTask[];
	onSend: (content: string) => void;
	onUpload: (files: File[]) => void;
	onCitationClick?: (docLabel: string, page: number) => void;
	onDismissUpload: () => void;
}

export function ChatWindow({
	messages,
	loading,
	error,
	streaming,
	streamingContent,
	statusMessage,
	searchSteps,
	hasDocuments,
	conversationId,
	uploadTasks,
	onSend,
	onUpload,
	onCitationClick,
	onDismissUpload,
}: ChatWindowProps) {
	const scrollRef = useRef<HTMLDivElement>(null);

	// Auto-scroll to bottom when new messages arrive or during streaming
	const messagesLength = messages.length;
	// biome-ignore lint/correctness/useExhaustiveDependencies: messages and streamingContent are intentional triggers for auto-scroll
	useEffect(() => {
		if (scrollRef.current) {
			scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
		}
	}, [messagesLength, streamingContent]);

	// No conversation selected
	if (!conversationId) {
		return (
			<div className="flex flex-1 items-center justify-center bg-neutral-50">
				<div className="text-center">
					<p className="text-sm text-neutral-400">
						Select a conversation or create a new one
					</p>
				</div>
			</div>
		);
	}

	// Loading messages
	if (loading) {
		return (
			<div className="flex flex-1 items-center justify-center bg-white">
				<Loader2 className="h-6 w-6 animate-spin text-neutral-400" />
			</div>
		);
	}

	// Empty conversation - show upload prompt
	if (messages.length === 0 && !streaming) {
		return (
			<div className="flex flex-1 flex-col bg-white">
				{uploadTasks.length > 0 && (
					<UploadProgress tasks={uploadTasks} onDismiss={onDismissUpload} />
				)}
				<div className="flex flex-1 items-center justify-center">
					{hasDocuments ? (
						<div className="text-center">
							<p className="text-sm text-neutral-500">
								Documents uploaded. Ask a question to get started.
							</p>
						</div>
					) : (
						<EmptyState onUpload={onUpload} />
					)}
				</div>
				<ChatInput
					onSend={onSend}
					onUpload={onUpload}
					disabled={streaming}
				/>
			</div>
		);
	}

	return (
		<div className="flex flex-1 flex-col bg-white">
			{uploadTasks.length > 0 && (
				<UploadProgress tasks={uploadTasks} onDismiss={onDismissUpload} />
			)}

			{error && (
				<div className="mx-4 mt-2 rounded-lg bg-red-50 px-4 py-2 text-sm text-red-600">
					{error}
				</div>
			)}

			<div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-4">
				<div className="mx-auto max-w-2xl space-y-1">
					{messages.map((message) => (
						<MessageBubble
							key={message.id}
							message={message}
							onCitationClick={onCitationClick}
						/>
					))}
					{streaming && (
						<StreamingBubble
							content={streamingContent}
							statusMessage={statusMessage}
							searchSteps={searchSteps}
						/>
					)}
				</div>
			</div>

			<ChatInput
				onSend={onSend}
				onUpload={onUpload}
				disabled={streaming}
			/>
		</div>
	);
}
