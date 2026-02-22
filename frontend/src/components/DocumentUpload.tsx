import { ScanLine, Upload } from "lucide-react";
import { type DragEvent, useCallback, useRef, useState } from "react";

interface DocumentUploadProps {
	onUpload: (files: File[], ocr: boolean) => void;
	uploading?: boolean;
}

export function DocumentUpload({
	onUpload,
	uploading = false,
}: DocumentUploadProps) {
	const [dragOver, setDragOver] = useState(false);
	const [ocrMode, setOcrMode] = useState(false);
	const fileInputRef = useRef<HTMLInputElement>(null);

	const handleDragOver = useCallback((e: DragEvent) => {
		e.preventDefault();
		setDragOver(true);
	}, []);

	const handleDragLeave = useCallback((e: DragEvent) => {
		e.preventDefault();
		setDragOver(false);
	}, []);

	const handleDrop = useCallback(
		(e: DragEvent) => {
			e.preventDefault();
			setDragOver(false);
			const files = Array.from(e.dataTransfer.files).filter(
				(f) => f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf"),
			);
			if (files.length > 0) {
				onUpload(files, ocrMode);
			}
		},
		[onUpload, ocrMode],
	);

	const handleClick = useCallback(() => {
		fileInputRef.current?.click();
	}, []);

	const handleFileChange = useCallback(
		(e: React.ChangeEvent<HTMLInputElement>) => {
			const files = Array.from(e.target.files ?? []);
			if (files.length > 0) {
				onUpload(files, ocrMode);
			}
			if (fileInputRef.current) {
				fileInputRef.current.value = "";
			}
		},
		[onUpload, ocrMode],
	);

	return (
		<div className="flex flex-col items-center gap-3">
			<button
				type="button"
				className={`w-full max-w-md cursor-pointer rounded-xl border-2 border-dashed px-8 py-10 text-center transition-colors ${
					dragOver
						? "border-neutral-400 bg-neutral-100"
						: "border-neutral-200 bg-white hover:border-neutral-300 hover:bg-neutral-50"
				}`}
				onDragOver={handleDragOver}
				onDragLeave={handleDragLeave}
				onDrop={handleDrop}
				onClick={handleClick}
				disabled={uploading}
			>
				<input
					ref={fileInputRef}
					type="file"
					accept=".pdf"
					multiple
					className="hidden"
					onChange={handleFileChange}
				/>

				<div className="flex flex-col items-center">
					<Upload className="mb-3 h-10 w-10 text-neutral-400" />
					<p className="text-sm font-medium text-neutral-600">
						Upload PDF documents
					</p>
					<p className="mt-1 text-xs text-neutral-400">
						Click or drag and drop — select multiple files
					</p>
				</div>
			</button>

			<button
				type="button"
				onClick={() => setOcrMode(!ocrMode)}
				className={`flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition-colors ${
					ocrMode
						? "bg-amber-100 text-amber-700 border border-amber-200"
						: "bg-neutral-100 text-neutral-500 border border-neutral-200 hover:bg-neutral-200"
				}`}
			>
				<ScanLine className="h-3 w-3" />
				{ocrMode ? "Scanned docs mode ON" : "Scanned docs? Click here"}
			</button>
		</div>
	);
}
