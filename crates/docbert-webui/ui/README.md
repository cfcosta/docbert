# React + TypeScript + Vite

## Docbert UI surface tokens

The shared design-system entrypoint is `src/index.css`. Use the semantic surface tokens in that file before you add new gradients for one page in the page CSS.

At this time, the design system has these extracted tokens:

- `--surface-control`: the default surface for interactive controls
- `--surface-control-hover`: the hover state for text inputs and neutral controls
- `--surface-emphasis-hover`: the hover state with more emphasis for toggles and secondary actions
- `--surface-selected`: the surface for a selected or active neutral card
- `--surface-accent-strong`: the surface for primary accent actions
- `--surface-accent-hover`: the hover state for primary accent actions
- `--surface-danger-soft`: the soft error surface for inline alert cards.

At this time, the consumers of these tokens include Search, Settings, Chat, and the shared `SearchResults` styles.

This template gives a minimum setup that makes React operate in Vite with HMR and some ESLint rules.

At this time, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs).
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/).

## React Compiler

The React Compiler is not active on this template because the compiler decreases dev and build performance. To add the compiler, refer to [this documentation](https://react.dev/learn/react-compiler/installation).

## How to expand the ESLint configuration

For a production application, we recommend that you update the configuration to activate the type-aware lint rules:

```js
export default defineConfig([
  globalIgnores(["dist"]),
  {
    files: ["**/*.{ts,tsx}"],
    extends: [
      // Other configs...

      // Remove tseslint.configs.recommended and replace with this
      tseslint.configs.recommendedTypeChecked,
      // Alternatively, use this for stricter rules
      tseslint.configs.strictTypeChecked,
      // Optionally, add this for stylistic rules
      tseslint.configs.stylisticTypeChecked,

      // Other configs...
    ],
    languageOptions: {
      parserOptions: {
        project: ["./tsconfig.node.json", "./tsconfig.app.json"],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
]);
```

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from "eslint-plugin-react-x";
import reactDom from "eslint-plugin-react-dom";

export default defineConfig([
  globalIgnores(["dist"]),
  {
    files: ["**/*.{ts,tsx}"],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs["recommended-typescript"],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ["./tsconfig.node.json", "./tsconfig.app.json"],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
]);
```
