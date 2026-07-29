/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        base: {
          950: "#0a0c11",
          900: "#0e1117",
          800: "#141822",
          700: "#1b202c",
          600: "#242a38",
          500: "#323a4b",
        },
        ink: {
          100: "#eef0f4",
          300: "#c7cbd6",
          500: "#8b93a7",
          700: "#5b6478",
        },
        accent: {
          blue: "#4c6ef5",
          bluedim: "#2b3a72",
          purple: "#8b6ff0",
          amber: "#f59f00",
          green: "#37b24d",
          red: "#f0554c",
        },
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      boxShadow: {
        panel: "0 1px 0 rgba(255,255,255,0.03) inset, 0 8px 24px rgba(0,0,0,0.35)",
      },
    },
  },
  plugins: [],
};
