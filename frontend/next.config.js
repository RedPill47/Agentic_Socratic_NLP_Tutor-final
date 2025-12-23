/** @type {import('next').NextConfig} */
const path = require('path')

// Load environment variables from root .env file (parent directory)
// This runs before Next.js processes the config, so variables are available
try {
  require('dotenv').config({ path: path.join(__dirname, '..', '.env') })
} catch (e) {
  // dotenv not available, Next.js will use its own env loading
  console.warn('Could not load .env from parent directory. Make sure dotenv is installed or use .env.local in frontend/')
}

const nextConfig = {
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  },
}

module.exports = nextConfig

