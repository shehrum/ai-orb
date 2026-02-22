import { useCallback } from "react";
import { ChatSidebar } from "./components/ChatSidebar";
import { ChatWindow } from "./components/ChatWindow";
import { DocumentViewer } from "./components/DocumentViewer";
import { TooltipProvider } from "./components/ui/tooltip";
import { useConversations } from "./hooks/use-conversations";
import { useDocuments } from "./hooks/use-documents";
import { useMessages } from "./hooks/use-messages";

export default function App() {
	const {
		conversations,
		selectedId,
		loading: conversationsLoading,
		create,
		select,
		remove,
		refresh: refreshConversations,
	} = useConversations();

	const {
		messages,
		loading: messagesLoading,
		error: messagesError,
		streaming,
		streamingContent,
		statusMessage,
		searchSteps,
		docLabelMap,
		send,
	} = useMessages(selectedId);

	const {
		documents,
		activeDocumentId,
		targetPage,
		uploadTasks,
		uploadFiles,
		refresh: refreshDocuments,
		navigateTo,
		clearTargetPage,
		clearUploadTasks,
		setActiveDocumentId,
	} = useDocuments(selectedId);

	const handleSend = useCallback(
		async (content: string) => {
			await send(content);
			refreshConversations();
		},
		[send, refreshConversations],
	);

	const handleUpload = useCallback(
		async (files: File[]) => {
			await uploadFiles(files);
			refreshDocuments();
			refreshConversations();
		},
		[uploadFiles, refreshDocuments, refreshConversations],
	);

	const handleCreate = useCallback(async () => {
		await create();
	}, [create]);

	const handleCitationClick = useCallback(
		(docLabel: string, page: number) => {
			const docId = docLabelMap[docLabel];
			if (docId) {
				navigateTo(docId, page);
			} else {
				const doc = documents.find((d) => d.label === docLabel);
				if (doc) {
					navigateTo(doc.id, page);
				}
			}
		},
		[docLabelMap, documents, navigateTo],
	);

	return (
		<TooltipProvider delayDuration={200}>
			<div className="flex h-screen bg-neutral-50">
				<ChatSidebar
					conversations={conversations}
					selectedId={selectedId}
					loading={conversationsLoading}
					onSelect={select}
					onCreate={handleCreate}
					onDelete={remove}
				/>

				<ChatWindow
					messages={messages}
					loading={messagesLoading}
					error={messagesError}
					streaming={streaming}
					streamingContent={streamingContent}
					statusMessage={statusMessage}
					searchSteps={searchSteps}
					hasDocuments={documents.length > 0}
					conversationId={selectedId}
					uploadTasks={uploadTasks}
					onSend={handleSend}
					onUpload={handleUpload}
					onCitationClick={handleCitationClick}
					onDismissUpload={clearUploadTasks}
				/>

				<DocumentViewer
					documents={documents}
					activeDocumentId={activeDocumentId}
					onSelectDocument={setActiveDocumentId}
					targetPage={targetPage}
					onTargetPageHandled={clearTargetPage}
				/>
			</div>
		</TooltipProvider>
	);
}
