# Editing the frontend

- `index.html`: site header, navigation, compose form, comparison container and footer. Look for HTML section comments.
- `style.css`: layout, typography, colors, responsive rules and animation. Section comments separate the main areas. CSS follows source order: later matching rules can override earlier ones; keep responsive rules with their section.
- `app.js`: model definitions and state, navigation, prediction requests, reusable rendering helpers, scores, sentence explanations, charts, detailed results and title animation. Search the section headings to find the part you need.

## Common changes

| Change | Location |
| --- | --- |
| Form labels and introductory text | `index.html` |
| Model names and descriptions | `modelInfo` in `app.js` |
| Prediction request and error handling | Prediction request and form events in `app.js` |
| Model cards and agreement | Prediction cards and model agreement in `app.js` |
| Model identity colors | Model identity colors section near the end of `style.css` |
| Fonts | Typography sections in `style.css` |
| Sentence highlighting | Highlighted email text in `app.js` |
| Glitch animation | Optional title animation in `app.js`, glitch keyframes in `style.css` |

Keep HTML IDs in sync with JavaScript lookups. The app uses plain HTML, CSS and JavaScript; no frontend build step is required. Start the Python server from the repository root and refresh the browser after editing. The local Prettier configuration keeps indentation and line wrapping consistent.
