/**
 * Parse and transform <cite> tags in LLM responses.
 */

const CITE_RE = /<cite\s+doc="([^"]+)"\s+page="(\d+)"(?:\s+section="([^"]*)")?\s*>(.*?)<\/cite>/gs;

/**
 * Convert <cite> tags to readable inline format with section info.
 * The [Doc X, Section, p.N] badges are later turned into clickable pills via DOM post-processing.
 */
export function transformCitations(text: string): string {
	return text.replace(CITE_RE, (_match, doc: string, page: string, section: string | undefined, quoted: string) => {
		if (section) {
			return `${quoted.trim()} [${doc}, ${section}, p.${page}]`;
		}
		return `${quoted.trim()} [${doc}, p.${page}]`;
	});
}

/**
 * For streaming: same as transformCitations but also strips incomplete tags.
 */
export function stripCiteTags(text: string): string {
	let result = transformCitations(text);
	result = result.replace(/<cite\s+[^>]*$/, "");
	return result;
}

/**
 * Regex that matches [Doc X, p.N] or [Doc X, Section, p.N] badges in rendered text.
 */
const BADGE_RE = /\[(Doc\s+[A-Z]+),\s*((?:(?!p\.\d).)+,\s*)?p\.(\d+)\]/g;

/**
 * Walk the DOM tree inside a container and replace [Doc X, ...] text nodes
 * with styled clickable <span> elements.
 */
export function hydrateCitationBadges(
	container: HTMLElement,
	onClick: (docLabel: string, page: number) => void,
): void {
	const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
	const nodesToProcess: Text[] = [];

	let node: Text | null;
	while ((node = walker.nextNode() as Text | null)) {
		if (BADGE_RE.test(node.textContent ?? "")) {
			nodesToProcess.push(node);
		}
		BADGE_RE.lastIndex = 0;
	}

	for (const textNode of nodesToProcess) {
		const text = textNode.textContent ?? "";
		const fragment = document.createDocumentFragment();
		let lastIndex = 0;

		BADGE_RE.lastIndex = 0;
		let match: RegExpExecArray | null;
		while ((match = BADGE_RE.exec(text)) !== null) {
			// Add text before the match
			if (match.index > lastIndex) {
				fragment.appendChild(document.createTextNode(text.slice(lastIndex, match.index)));
			}

			const docLabel = match[1];
			const sectionPart = match[2] ? match[2].replace(/,\s*$/, "").trim() : null;
			const page = Number.parseInt(match[3], 10);

			// Create the citation pill
			const pill = document.createElement("span");
			pill.className = "citation-pill";
			pill.dataset.doc = docLabel;
			pill.dataset.page = String(page);

			if (sectionPart) {
				pill.textContent = `${docLabel} · ${sectionPart} · p.${page}`;
				pill.dataset.section = sectionPart;
			} else {
				pill.textContent = `${docLabel} · p.${page}`;
			}

			pill.addEventListener("click", (e) => {
				e.preventDefault();
				e.stopPropagation();
				onClick(docLabel, page);
			});

			fragment.appendChild(pill);
			lastIndex = match.index + match[0].length;
		}

		// Add remaining text
		if (lastIndex < text.length) {
			fragment.appendChild(document.createTextNode(text.slice(lastIndex)));
		}

		textNode.parentNode?.replaceChild(fragment, textNode);
	}
}
