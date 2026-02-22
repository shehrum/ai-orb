import { motion } from "framer-motion";
import { Bot, ChevronDown, ChevronUp, Globe, Search } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Streamdown } from "streamdown";
import "streamdown/styles.css";
import { hydrateCitationBadges, stripCiteTags, transformCitations } from "../lib/citation-parser";
import type { Message } from "../types";

interface MessageBubbleProps {
	message: Message;
	onCitationClick?: (docLabel: string, page: number) => void;
}

export function MessageBubble({ message, onCitationClick }: MessageBubbleProps) {
	const proseRef = useRef<HTMLDivElement>(null);

	// After Streamdown renders, hydrate [Doc X, p.N] text into clickable pills
	useEffect(() => {
		if (!proseRef.current || !onCitationClick || message.role !== "assistant") return;

		// Small delay to let Streamdown finish rendering
		const timer = setTimeout(() => {
			if (proseRef.current) {
				hydrateCitationBadges(proseRef.current, onCitationClick);
			}
		}, 100);
		return () => clearTimeout(timer);
	}, [message.content, message.role, onCitationClick]);

	if (message.role === "system") {
		return (
			<motion.div
				initial={{ opacity: 0 }}
				animate={{ opacity: 1 }}
				transition={{ duration: 0.2 }}
				className="flex justify-center py-2"
			>
				<p className="text-xs text-neutral-400">{message.content}</p>
			</motion.div>
		);
	}

	if (message.role === "user") {
		return (
			<motion.div
				initial={{ opacity: 0, y: 8 }}
				animate={{ opacity: 1, y: 0 }}
				transition={{ duration: 0.2 }}
				className="flex justify-end py-1.5"
			>
				<div className="max-w-[75%] rounded-2xl rounded-br-md bg-neutral-100 px-4 py-2.5">
					<p className="whitespace-pre-wrap text-sm text-neutral-800">
						{message.content}
					</p>
				</div>
			</motion.div>
		);
	}

	// Assistant message — transform citations to readable text, then hydrate pills via useEffect
	const displayContent = transformCitations(message.content);

	return (
		<motion.div
			initial={{ opacity: 0, y: 8 }}
			animate={{ opacity: 1, y: 0 }}
			transition={{ duration: 0.2 }}
			className="flex gap-3 py-1.5"
		>
			<div className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full bg-neutral-900">
				<Bot className="h-4 w-4 text-white" />
			</div>
			<div className="min-w-0 max-w-[80%]">
				<div ref={proseRef} className="prose">
					<Streamdown>{displayContent}</Streamdown>
				</div>
				{message.sources_cited > 0 && (
					<p className="mt-1.5 text-xs text-neutral-400">
						{message.sources_cited} source
						{message.sources_cited !== 1 ? "s" : ""} cited
					</p>
				)}
			</div>
		</motion.div>
	);
}

interface StreamingBubbleProps {
	content: string;
	statusMessage?: string | null;
	searchSteps?: string[];
}

function isWebStep(step: string): boolean {
	return step.startsWith("Web search:") || step.includes("web results");
}

function isDocStep(step: string): boolean {
	return step.startsWith("Searching:") || step.startsWith("Found") && step.includes("results across");
}

export function StreamingBubble({ content, statusMessage, searchSteps = [] }: StreamingBubbleProps) {
	const [logExpanded, setLogExpanded] = useState(false);
	const displayContent = content ? stripCiteTags(content) : "";
	const hasSteps = searchSteps.length > 0;
	const isSearching = statusMessage && !displayContent;
	const isWebStatus = statusMessage ? isWebStep(statusMessage) : false;
	const webStepCount = searchSteps.filter(isWebStep).length;
	const docStepCount = searchSteps.filter(isDocStep).length;

	const summaryParts = [
		docStepCount > 0 ? `${docStepCount} doc search${docStepCount !== 1 ? "es" : ""}` : "",
		webStepCount > 0 ? `${webStepCount} web search${webStepCount !== 1 ? "es" : ""}` : "",
	].filter(Boolean);

	return (
		<div className="flex gap-3 py-1.5">
			<div className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full bg-neutral-900">
				<Bot className="h-4 w-4 text-white" />
			</div>
			<div className="min-w-0 max-w-[80%]">
				{/* Activity log — collapsible */}
				{hasSteps && (
					<div className="mb-2 overflow-hidden rounded-lg border border-neutral-100 bg-neutral-50 text-xs">
						<button
							type="button"
							onClick={() => setLogExpanded(!logExpanded)}
							className="flex w-full items-center gap-1.5 px-2.5 py-1.5 text-left text-neutral-500 hover:bg-neutral-100"
						>
							{isSearching && isWebStatus ? (
								<Globe className="h-3 w-3 flex-shrink-0 animate-pulse text-emerald-500" />
							) : (
								<Search className={`h-3 w-3 flex-shrink-0 ${isSearching ? "animate-pulse text-blue-500" : "text-neutral-400"}`} />
							)}
							<span className="flex-1">
								{isSearching
									? statusMessage
									: `${summaryParts.join(", ")} completed`}
							</span>
							{logExpanded ? (
								<ChevronUp className="h-3 w-3 text-neutral-400" />
							) : (
								<ChevronDown className="h-3 w-3 text-neutral-400" />
							)}
						</button>
						{logExpanded && (
							<div className="border-t border-neutral-100 px-2.5 py-1.5">
								{searchSteps.map((step, i) => (
									<div key={`step-${i}`} className="flex items-start gap-1.5 py-0.5 text-neutral-500">
										{isWebStep(step) ? (
											<Globe className="mt-0.5 h-3 w-3 flex-shrink-0 text-emerald-400" />
										) : isDocStep(step) ? (
											<Search className="mt-0.5 h-3 w-3 flex-shrink-0 text-blue-300" />
										) : (
											<span className="mt-0.5 h-3 w-3 flex-shrink-0 rounded-full bg-neutral-200" />
										)}
										<span>{step}</span>
									</div>
								))}
							</div>
						)}
					</div>
				)}

				{/* Thinking dots — only when no content and no search steps yet */}
				{!displayContent && !hasSteps && (
					<div className="flex items-center gap-1 py-2">
						<span className="h-1.5 w-1.5 animate-pulse rounded-full bg-neutral-400" />
						<span
							className="h-1.5 w-1.5 animate-pulse rounded-full bg-neutral-400"
							style={{ animationDelay: "0.15s" }}
						/>
						<span
							className="h-1.5 w-1.5 animate-pulse rounded-full bg-neutral-400"
							style={{ animationDelay: "0.3s" }}
						/>
					</div>
				)}

				{/* Streaming content */}
				{displayContent && (
					<>
						<div className="prose">
							<Streamdown mode="streaming">{displayContent}</Streamdown>
						</div>
						<span className="inline-block h-4 w-0.5 animate-pulse bg-neutral-400" />
					</>
				)}
			</div>
		</div>
	);
}
