export default defineNuxtConfig({
  compatibilityDate: '2025-05-14',
  css: ['@/assets/css/tailwind.css'],
  postcss: {
    plugins: {
      tailwindcss: {},
      autoprefixer: {},
    },
  },
})
