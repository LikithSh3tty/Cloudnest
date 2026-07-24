# Premium CloudNest Support UI Design

## Goal

Replace the current generic chat-page presentation with a premium support workspace that feels deliberately designed by a product team: calm, editorial, precise, and focused on the customer’s question.

## Direction

Use a dark, editorial SaaS workspace rather than a glassy, neon, or mascot-led AI-chat interface.

- **Palette:** deep ink and blue-black surfaces; warm off-white text; restrained sage-green for status and primary actions.
- **Typography:** a distinctive display face for the welcome moment, paired with a highly legible sans-serif UI face and small monospaced labels for operational metadata.
- **Character:** ample whitespace, low visual noise, soft but purposeful corner radii, and limited card treatment. Avoid gradients, decorative sparkles, fake performance statistics, and excessive status badges.

## Information Architecture

### Desktop

Use a three-column support workspace:

1. **Left navigation** — CloudNest Support identity, a small set of workspace actions, and a concise support-context note.
2. **Conversation** — primary reading and replying area; it must receive the strongest visual emphasis.
3. **Context panel** — only relevant topic and related-help links. It is quiet, optional context rather than a dashboard of generic AI features.

The conversation header displays the product/support identity and a subtle availability signal. The empty state uses an editorial greeting and curated help prompts. Active turns use a modest customer bubble and an unboxed, human support response with a small identity marker.

### Mobile

Collapse the left navigation and context panel. Preserve the conversation header, generous reading width, prominent composer, and the same information order. The main interaction must not require horizontal scrolling.

## Components

- **Brand mark and navigation:** restrained, text-led product identity; no large illustration.
- **Header:** compact topic/context label, support title, and availability indicator.
- **Empty state:** product-like welcome copy with three editorially styled support prompts.
- **Conversation:** clear distinction between user and support messages without making every message a floating card.
- **Composer:** dark contained input with a single sage send action, remaining visually anchored at the bottom of the conversation.
- **Context panel:** relevant guidance only; hide it on narrow screens.

## Interaction and State

- Preserve the existing API contract and current React message state behavior.
- Keep the existing loading state, error state, Markdown rendering, example prompts, health badge behavior, and clarify-response styling, but restyle them to the approved direction.
- Use familiar hover, focus, disabled, and keyboard-visible states. The primary send action is the only strongly filled control in its immediate area.
- Do not expose routing, confidence, retrieval, or other implementation details in the interface.

## Accessibility and Quality

- Maintain sufficient contrast for body text, muted labels, status indicators, and interactive controls on dark surfaces.
- Use semantic buttons and inputs, descriptive labels for icon-only controls, and visible keyboard focus.
- Respect reduced-motion preferences; do not add ambient or looping animation.
- Ensure the layout works from 320px upward and does not clip message content or composer controls.

## Out of Scope

- No backend, LangGraph, retrieval, authentication, or data-model changes.
- No new support ticket system, account panel, or persistence feature.
- No new external imagery or stock illustrations.
